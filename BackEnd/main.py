#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import requests
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import os
import sys
import json

OCM_API_KEY = os.environ.get("OPENCHARGEMAP_API_KEY")

# ----------------------------
# Configuration dataclasses
# ----------------------------
@dataclass
class VehicleSpec:
    battery_kwh: float = 60.0
    base_kwh_per_km: float = 0.16          # nominal traction (adjust as needed)
    min_soc_reserve: float = 0.10          # reserve SOC (10%)
    max_soc: float = 0.90                  # stop charging at 90% for speed
    charge_power_kw: float = 50.0          # assumed DCFC power for prototype
    target_cabin_temp_c: float = 22.0
    current_soc: float = 0.80              # start SOC (0.0 - 1.0)


# ----------------------------
# Small helpers
# ----------------------------
def input_with_default(prompt: str, default, cast=str):
    """
    Prompt user; if empty input, return default (casted appropriately).
    cast is a function (str->type)
    """
    if default is None:
        raw = input(f"{prompt}: ")
        return cast(raw)
    # show default in prompt
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except Exception:
        print(f"Invalid input, using default {default}", file=sys.stderr)
        return default


# ----------------------------
# Data sources
# ----------------------------
def get_weather_open_meteo(lat: float, lon: float) -> Tuple[float, float]:
    """
    Returns (temperature_c, windspeed_kmh) using Open-Meteo.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current_weather=true"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    # open-meteo current_weather: {"temperature":..,"windspeed":..}
    cw = data.get("current_weather", {}) or {}
    t = float(cw.get("temperature", 25.0))
    w = float(cw.get("windspeed", 5.0))
    return t, w


def get_chargers_open_charge_map(lat: float, lon: float, radius_km: float = 25.0,
                                max_results: int = 100, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch chargers from Open Charge Map.
    Returns empty DataFrame safely if no chargers or missing fields.
    """
    if not api_key:
        print("Warning: OpenChargeMap API key missing — skipping charger fetch.")
        return pd.DataFrame(columns=["station_id", "title", "lat", "lon"])

    base = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "latitude": lat,
        "longitude": lon,
        "distance": radius_km,
        "distanceunit": "KM",
        "maxresults": max_results,
        "compact": "true",
        "verbose": "false",
    }
    headers = {
        "User-Agent": "WARCH-P-Student-Prototype/1.0",
        "Accept": "application/json",
        "X-API-Key": api_key
    }

    try:
        r = requests.get(base, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        pois = r.json()
    except Exception as e:
        print(f"Warning: Charger API failed: {e}")
        return pd.DataFrame(columns=["station_id", "title", "lat", "lon"])

    rows = []
    for p in pois:
        addr = p.get("AddressInfo") or {}
        lat_ = addr.get("Latitude")
        lon_ = addr.get("Longitude")

        # skip invalid coordinates
        if lat_ is None or lon_ is None:
            continue

        rows.append({
            "station_id": p.get("ID"),
            "title": addr.get("Title", ""),
            "lat": float(lat_),
            "lon": float(lon_),
        })

    if not rows:
        return pd.DataFrame(columns=["station_id", "title", "lat", "lon"])

    return pd.DataFrame(rows)



# ----------------------------
# Models
# ----------------------------
def wind_chill_index(temp_c: float, wind_kmh: float) -> float:
    v = max(wind_kmh, 0.0)
    return temp_c - 0.7 * (v / 10.0)


def hvac_kwh_per_km(temp_c: float, wind_kmh: float, target_cabin_c: float) -> float:
    delta = max(target_cabin_c - temp_c, 0.0)
    wc = wind_chill_index(temp_c, wind_kmh)
    a1 = 0.0020
    a2 = 0.0006
    hvac = a1 * delta + a2 * max((target_cabin_c - wc), 0.0)
    return max(hvac, 0.0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def snap_points_to_graph(G: nx.MultiDiGraph, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    return np.array(nodes, dtype=np.int64)


def compute_kwh_per_km(temp_c: float, wind_kmh: float, vehicle: VehicleSpec) -> float:
    hvac_term = hvac_kwh_per_km(temp_c, wind_kmh, vehicle.target_cabin_temp_c)
    return vehicle.base_kwh_per_km + hvac_term


def plan_single_charge_route(
    G: nx.MultiDiGraph,
    source_node: int,
    target_node: int,
    chargers_df: pd.DataFrame,
    temp_c: float,
    wind_kmh: float,
    vehicle: VehicleSpec,
) -> Dict:
    """
    Plan: if destination is within current range, go direct.
    Otherwise, find all reachable chargers and pick the optimal one
    (shortest total distance) then drive -> charge -> drive.
    """
    kwh_per_km = compute_kwh_per_km(temp_c, wind_kmh, vehicle)
    reserve_kwh = vehicle.battery_kwh * vehicle.min_soc_reserve
    current_kwh = vehicle.battery_kwh * float(np.clip(vehicle.current_soc, 0.0, 1.0))
    usable_kwh = max(current_kwh - reserve_kwh, 0.0)
    range_km = usable_kwh / max(kwh_per_km, 1e-9)

    len_from_source = nx.single_source_dijkstra_path_length(G, source_node, weight="length")
    if target_node not in len_from_source:
        return {"ok": False, "reason": "No path to destination on road network."}

    dist_src_to_target_km = len_from_source[target_node] / 1000.0
    time_from_source = nx.single_source_dijkstra_path_length(G, source_node, weight="travel_time")
    time_to_target = nx.single_source_dijkstra_path_length(G.reverse(copy=False), target_node, weight="travel_time")

    print(f"Estimated range from current SOC: {range_km:.2f} km")
    print(f"Shortest path distance (source -> destination): {dist_src_to_target_km:.2f} km")

    def route_nodes(u: int, v: int) -> List[int]:
        if u == v:
            return [u]
        return nx.shortest_path(G, u, v, weight="length")

    def nodes_to_coords(nodes: List[int]) -> List[Tuple[float, float]]:
        coords = []
        for n in nodes:
            data = G.nodes[n]
            coords.append((float(data["y"]), float(data["x"])))
        return coords

    def edge_length_km(u: int, v: int) -> float:
        data = G.get_edge_data(u, v)
        if not data:
            return 0.0
        best = None
        for _, attrs in data.items():
            length_m = float(attrs.get("length", 0.0))
            if best is None or length_m < best:
                best = length_m
        return (best or 0.0) / 1000.0

    def drive_actions_from_nodes(nodes: List[int]) -> List[Tuple[str, int, int, float]]:
        actions = []
        for i in range(len(nodes) - 1):
            u = nodes[i]
            v = nodes[i + 1]
            dist_km = edge_length_km(u, v)
            actions.append(("DRIVE", u, v, dist_km))
        return actions

    if dist_src_to_target_km <= range_km:
        time_min = (time_from_source.get(target_node, 0.0) or 0.0) / 60.0
        nodes = route_nodes(source_node, target_node)
        drive_actions = drive_actions_from_nodes(nodes)
        return {
            "ok": True,
            "cost_minutes": time_min,
            "actions": drive_actions,
            "route_nodes": nodes,
            "route_coords": nodes_to_coords(nodes),
        }

    if chargers_df.empty:
        return {"ok": False, "reason": "Destination out of range and no chargers available."}

    len_to_target = nx.single_source_dijkstra_path_length(G.reverse(copy=False), target_node, weight="length")

    max_soc_kwh = vehicle.max_soc * vehicle.battery_kwh
    usable_kwh_after = max(max_soc_kwh - reserve_kwh, 0.0)
    range_after_km = usable_kwh_after / max(kwh_per_km, 1e-9)

    best = None
    charger_nodes = chargers_df["graph_node"].dropna().astype(int).tolist()

    # Minimum distance from start before allowing a charge stop. Adjust or remove as needed.
    min_dist_from_start_km = 1.0
    for c in charger_nodes:
        if c not in len_from_source or c not in len_to_target:
            continue
        dist1_km = len_from_source[c] / 1000.0
        dist2_km = len_to_target[c] / 1000.0
        if dist1_km <= min_dist_from_start_km:
            continue
        if dist1_km > range_km:
            continue
        if dist2_km > range_after_km:
            continue

        time1_min = (time_from_source.get(c, 0.0) or 0.0) / 60.0
        time2_min = (time_to_target.get(c, 0.0) or 0.0) / 60.0

        kwh_after_drive = current_kwh - dist1_km * kwh_per_km
        add_kwh = max(max_soc_kwh - kwh_after_drive, 0.0)
        charge_time_min = 60.0 * add_kwh / max(vehicle.charge_power_kw, 1e-6)

        total_cost = time1_min + time2_min + charge_time_min
        total_dist = dist1_km + dist2_km

        cand = {
            "charger_node": c,
            "dist1_km": dist1_km,
            "dist2_km": dist2_km,
            "add_kwh": add_kwh,
            "cost_minutes": total_cost,
            "total_dist_km": total_dist,
        }

        if best is None or cand["cost_minutes"] < best["cost_minutes"]:
            best = cand

    if best is None:
        return {"ok": False, "reason": "No reachable charger found within current range."}

    nodes_a = route_nodes(source_node, best["charger_node"])
    nodes_b = route_nodes(best["charger_node"], target_node)
    full_nodes = nodes_a + nodes_b[1:] if nodes_b else nodes_a
    full_coords = nodes_to_coords(full_nodes)
    drive_actions_a = drive_actions_from_nodes(nodes_a)
    drive_actions_b = drive_actions_from_nodes(nodes_b)
    charge_coord = nodes_to_coords([best["charger_node"]])[0]

    print(
        "Chosen charger node: "
        f"{best['charger_node']} | dist1={best['dist1_km']:.2f} km | "
        f"dist2={best['dist2_km']:.2f} km | charge={best['add_kwh']:.2f} kWh"
    )

    charge_action = ("CHARGE", best["charger_node"], best["add_kwh"], charge_time_min, charge_coord)
    return {
        "ok": True,
        "cost_minutes": best["cost_minutes"],
        "actions": drive_actions_a + [charge_action] + drive_actions_b,
        "route_nodes": full_nodes,
        "route_coords": full_coords,
    }


# ----------------------------
# Demo runner (interactive with defaults)
# ----------------------------
def main():
    print("EV Risk-Aware Router — interactive mode (press Enter to accept default shown)\n")

    # Defaults chosen: VIT-AP (Amaravati) -> IIT Madras
    default_place = "Amaravati, Andhra Pradesh, India"
    default_src_lat, default_src_lon = 16.5730, 80.3570   # Amaravati approximate
    default_dst_lat, default_dst_lon = 13.0100, 80.2400   # IIT Madras approximate

    place = input_with_default("Enter city or place name (used for info only; graph built from coords)", default_place, str)

    src_lat = input_with_default("Enter source latitude", default_src_lat, float)
    src_lon = input_with_default("Enter source longitude", default_src_lon, float)

    dst_lat = input_with_default("Enter destination latitude", default_dst_lat, float)
    dst_lon = input_with_default("Enter destination longitude", default_dst_lon, float)

    # Vehicle / battery inputs
    print("\nVehicle / battery parameters (defaults shown):")
    battery_kwh = input_with_default(" Battery capacity (kWh)", 60.0, float)
    current_soc = input_with_default(" Current SOC at departure (0.0-1.0)", 0.80, float)
    min_soc_reserve = input_with_default(" Min SOC reserve (0.0-1.0)", 0.10, float)
    base_kwh_per_km = input_with_default(" Base kWh per km (vehicle efficiency)", 0.16, float)
    charge_power_kw = input_with_default(" Max charging power (kW)", 50.0, float)
    target_cabin_temp = input_with_default(" Cabin target temperature (C)", 22.0, float)

    # Routing / environment
    print("\nRouting & environment:")
    charger_search_radius_km = input_with_default(" Charger search radius (km) (for initial fetch)", 30.0, float)
    weather_override = input_with_default(" Skip live weather? (y/n)", "n", str).lower().startswith("y")

    # Build graph for bounding box around src/dst
    buffer_km = input_with_default(" Graph bbox buffer (km)", 10.0, float)
    straight_km = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
    min_buffer_km = straight_km / 2.0 + 10.0
    if buffer_km < min_buffer_km:
        print(f"Auto-adjusting buffer_km from {buffer_km:.1f} to {min_buffer_km:.1f} km")
        buffer_km = min_buffer_km
    print("\nConstructing graph (this may take some time)...")
    print("\nConstructing graph using point-radius...")
    cache_name = (
        f"graph_{src_lat:.3f}_{src_lon:.3f}_to_{dst_lat:.3f}_{dst_lon:.3f}"
        f"_r{buffer_km:.0f}km.graphml"
    )
    if os.path.exists(cache_name):
        print(f"Loading cached graph: {cache_name}")
        G = ox.load_graphml(cache_name)
    else:
        center_lat = (src_lat + dst_lat) / 2.0
        center_lon = (src_lon + dst_lon) / 2.0
        dist_m = max(500, int(buffer_km * 1000))
        G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="drive")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, cache_name)
        print(f"Graph saved to {cache_name}")


    # Snap source/target to nearest graph nodes
    source_node = ox.distance.nearest_nodes(G, X=src_lon, Y=src_lat)
    target_node = ox.distance.nearest_nodes(G, X=dst_lon, Y=dst_lat)

    # Weather
    if not weather_override:
        try:
            temp_c, wind_kmh = get_weather_open_meteo(src_lat, src_lon)
        except Exception as e:
            print(f"Warning: weather fetch failed ({e}); using defaults.", file=sys.stderr)
            temp_c, wind_kmh = 25.0, 5.0
    else:
        temp_c = input_with_default(" Ambient temperature (°C)", 25.0, float)
        wind_kmh = input_with_default(" Wind speed (km/h)", 5.0, float)

    print(f"Weather: T={temp_c:.1f}C, wind={wind_kmh:.1f} km/h")

    # Charger fetch (near source for demo). If no API key, skip but continue.
    chargers = get_chargers_open_charge_map(src_lat, src_lon, radius_km=charger_search_radius_km,
                                           max_results=200, api_key=OCM_API_KEY)
    
    # Fetch chargers near mid-point (IMPORTANT for inter-city trips)
    mid_lat = (src_lat + dst_lat) / 2
    mid_lon = (src_lon + dst_lon) / 2

    chargers_mid = get_chargers_open_charge_map(
        mid_lat,
        mid_lon,
        radius_km=charger_search_radius_km,
        max_results=80,
        api_key=OCM_API_KEY
    )

    # Combine source + mid-point chargers
    chargers = pd.concat([chargers, chargers_mid], ignore_index=True)

    print(f"Chargers fetched total: {len(chargers)}")

    if chargers.empty:
        print("No chargers found (or API key missing). Router will run without charger stops.")
    else:
        # snap chargers to graph nodes (only those inside graph bbox will snap)
        # Clean charger coordinates (CRITICAL FIX)
        chargers["lat"] = pd.to_numeric(chargers["lat"], errors="coerce")
        chargers["lon"] = pd.to_numeric(chargers["lon"], errors="coerce")

        chargers = chargers.dropna(subset=["lat", "lon"]).reset_index(drop=True)

        print(f"Valid chargers after cleaning: {len(chargers)}")

        # Snap chargers to graph
        chargers["graph_node"] = snap_points_to_graph(
            G,
            chargers["lat"].values.astype(float),
            chargers["lon"].values.astype(float)
        )

        print(f"Fetched {len(chargers)} chargers (snapped to graph nodes).")

    # Build vehicle & routing params from user inputs
    vehicle = VehicleSpec(
        battery_kwh=battery_kwh,
        base_kwh_per_km=base_kwh_per_km,
        min_soc_reserve=min_soc_reserve,
        max_soc=0.90,
        charge_power_kw=charge_power_kw,
        target_cabin_temp_c=target_cabin_temp,
        current_soc=current_soc,
    )

    print("\nRunning range-aware routing...")
    result = plan_single_charge_route(
        G=G,
        source_node=source_node,
        target_node=target_node,
        chargers_df=chargers,
        temp_c=temp_c,
        wind_kmh=wind_kmh,
        vehicle=vehicle,
    )

    if not result["ok"]:
        print("Routing failed:", result["reason"])
        return

    print(f"\nEstimated travel time: {result['cost_minutes']:.1f} minutes")
    total_km = sum(a[3] for a in result["actions"] if a[0] == "DRIVE")
    print(f"Approx total distance (sum of edges): {total_km:.2f} km")
    for act in result["actions"]:
        if act[0] == "CHARGE":
            print(f"Estimated charging time: {act[3]:.1f} minutes")
            break
    if "route_coords" in result:
        print(f"Route nodes count: {len(result['route_coords'])}")
        route_out = {
            "route_coords": result["route_coords"],
            "actions": result["actions"],
        }
        with open("route_output.json", "w", encoding="utf-8") as f:
            json.dump(route_out, f, ensure_ascii=False)
        print("Saved route_output.json")
    print("Actions (first 200 shown):")
    for act in result["actions"][:200]:
        print(act)

    # done
    print("\nDone.")

if __name__ == "__main__":
    main()
