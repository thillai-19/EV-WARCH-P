#!/usr/bin/env python3
import math
import heapq
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import requests
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import os
import sys

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


@dataclass
class RoutingParams:
    soc_bins: int = 30
    lambda_risk: float = 60.0  # minutes penalty per unit risk (tune for demos)
    max_route_km: float = 500.0


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


def station_success_probability(success: int, total: int, degradation: float = 1.0,
                                alpha: float = 2.0, beta: float = 2.0) -> float:
    total = max(total, 0)
    success = min(max(success, 0), total) if total > 0 else 0
    p = (success + alpha) / (total + alpha + beta)
    p *= float(np.clip(degradation, 0.0, 1.0))
    return float(np.clip(p, 0.0, 1.0))


# ----------------------------
# Routing helpers
# ----------------------------
def discretize_soc(soc: float, bins: int) -> int:
    soc = float(np.clip(soc, 0.0, 1.0))
    return int(round(soc * (bins - 1)))


def undiscretize_soc(bin_id: int, bins: int) -> float:
    return bin_id / (bins - 1)


def build_graph_bbox(src_lat: float, src_lon: float, dst_lat: float, dst_lon: float,
                     buffer_km: float = 20.0) -> nx.MultiDiGraph:
    """
    Build OSMnx driving graph that covers the box containing source and destination + a buffer.
    buffer_km: how many kilometers to extend the bounding box on all sides (default 20 km)
    """
    # approx degrees for buffer
    deg_per_km_lat = 1.0 / 111.0
    avg_lat = (src_lat + dst_lat) / 2.0
    deg_per_km_lon = 1.0 / (111.320 * math.cos(math.radians(avg_lat)) + 1e-9)

    bdeg_lat = buffer_km * deg_per_km_lat
    bdeg_lon = buffer_km * deg_per_km_lon

    north = max(src_lat, dst_lat) + bdeg_lat
    south = min(src_lat, dst_lat) - bdeg_lat
    east = max(src_lon, dst_lon) + bdeg_lon
    west = min(src_lon, dst_lon) - bdeg_lon

    print(f"Building graph bounding box: north={north:.4f} south={south:.4f} east={east:.4f} west={west:.4f}")
    G = ox.graph_from_bbox(
    bbox=(north, south, east, west),
    network_type="drive"
    )
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def snap_points_to_graph(G: nx.MultiDiGraph, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    return np.array(nodes, dtype=np.int64)


def risk_aware_route(
    G: nx.MultiDiGraph,
    source_node: int,
    target_node: int,
    chargers_df: pd.DataFrame,
    charger_logs_df: pd.DataFrame,
    temp_c: float,
    wind_kmh: float,
    vehicle: VehicleSpec,
    params: RoutingParams,
) -> Dict:
    """
    Returns best path info using risk-aware objective with SOC + charging.
    """
    charger_nodes = set(chargers_df["graph_node"].tolist()) if not chargers_df.empty else set()

    # Map station_id -> success probability from logs
    log_map = {row.station_id: row for row in charger_logs_df.itertuples(index=False)} if not charger_logs_df.empty else {}
    station_p = {}
    for row in chargers_df.itertuples(index=False):
        log = log_map.get(row.station_id, None)
        if log is None:
            p = station_success_probability(5, 10, degradation=0.9)
        else:
            p = station_success_probability(int(log.success), int(log.total), float(log.degradation))
        station_p[row.station_id] = p

    node_p = {}
    for row in chargers_df.itertuples(index=False):
        node_p[row.graph_node] = max(node_p.get(row.graph_node, 0.0), station_p[row.station_id])

    hvac_term = hvac_kwh_per_km(temp_c, wind_kmh, vehicle.target_cabin_temp_c)

    bins = params.soc_bins
    start_soc = float(np.clip(vehicle.current_soc, 0.0, 1.0))
    start_bin = discretize_soc(start_soc, bins)

    INF = 1e18
    dist = {}
    prev = {}
    pq = []

    start_state = (source_node, start_bin)
    dist[start_state] = 0.0
    heapq.heappush(pq, (0.0, start_state))

    reserve_kwh = vehicle.battery_kwh * vehicle.min_soc_reserve

    def soc_bin_to_kwh(soc_bin: int) -> float:
        soc = undiscretize_soc(soc_bin, bins)
        return soc * vehicle.battery_kwh

    def kwh_to_soc_bin(kwh: float) -> int:
        soc = float(np.clip(kwh / vehicle.battery_kwh, 0.0, 1.0))
        return discretize_soc(soc, bins)

    while pq:
        cur_cost, (u, soc_bin) = heapq.heappop(pq)
        if cur_cost != dist.get((u, soc_bin), INF):
            continue

        if u == target_node:
            break

        u_kwh = soc_bin_to_kwh(soc_bin)

        # DRIVE transitions
        for _, v, key, data in G.out_edges(u, keys=True, data=True):
            dist_m = float(data.get("length", 0.0))
            dist_km = dist_m / 1000.0
            travel_time_s = float(data.get("travel_time", 0.0) or 0.0)
            travel_time_min = travel_time_s / 60.0

            seg_kwh = dist_km * (vehicle.base_kwh_per_km + hvac_term)

            if (u_kwh - seg_kwh) < reserve_kwh:
                continue

            v_kwh = u_kwh - seg_kwh
            v_soc_bin = kwh_to_soc_bin(v_kwh)

            new_cost = cur_cost + travel_time_min

            state2 = (v, v_soc_bin)
            if new_cost < dist.get(state2, INF):
                dist[state2] = new_cost
                prev[state2] = ((u, soc_bin), ("DRIVE", u, v, dist_km))
                heapq.heappush(pq, (new_cost, state2))

                # Force charge if SOC < 40% and charger exists
        if u in charger_nodes and u_kwh < 0.4 * vehicle.battery_kwh:
            p = node_p.get(u, 0.8)
            risk_penalty = params.lambda_risk * (1.0 - p)

            target_kwh = vehicle.max_soc * vehicle.battery_kwh
            add_kwh = target_kwh - u_kwh

            charge_time_min = 60 * add_kwh / vehicle.charge_power_kw
            new_cost = cur_cost + charge_time_min + risk_penalty

            new_bin = kwh_to_soc_bin(target_kwh)
            state2 = (u, new_bin)

            if new_cost < dist.get(state2, INF):
                dist[state2] = new_cost
                prev[state2] = ((u, soc_bin), ("CHARGE", u, add_kwh, p))
                heapq.heappush(pq, (new_cost, state2))


        # CHARGE action (if charger available at node)
        if u in charger_nodes:
            p = node_p.get(u, 0.70)
            risk_penalty = params.lambda_risk * (1.0 - p)

            target_soc = vehicle.max_soc
            target_kwh = target_soc * vehicle.battery_kwh

            if target_kwh > u_kwh + 1e-9:
                add_kwh = target_kwh - u_kwh
                charge_time_h = add_kwh / max(vehicle.charge_power_kw, 1e-6)
                charge_time_min = 60.0 * charge_time_h

                new_kwh = target_kwh
                new_bin = kwh_to_soc_bin(new_kwh)
                new_cost = cur_cost + charge_time_min + risk_penalty

                state2 = (u, new_bin)
                if new_cost < dist.get(state2, INF):
                    dist[state2] = new_cost
                    prev[state2] = ((u, soc_bin), ("CHARGE", u, add_kwh, p))
                    heapq.heappush(pq, (new_cost, state2))

    end_states = [(cost, state) for state, cost in dist.items() if state[0] == target_node]
    if not end_states:
        return {"ok": False, "reason": "No feasible route found with SOC constraints."}

    best_cost, best_state = min(end_states, key=lambda x: x[0])

    actions = []
    cur = best_state
    while cur != start_state:
        pr = prev.get(cur)
        if pr is None:
            break
        cur_prev, act = pr
        actions.append(act)
        cur = cur_prev
    actions.reverse()

    return {"ok": True, "cost_minutes": best_cost, "actions": actions}


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
    lambda_risk = input_with_default(" Risk penalty lambda (minutes per unit failure)", 60.0, float)
    soc_bins = input_with_default(" Number of SOC bins (discretization)", 30, int)
    charger_search_radius_km = input_with_default(" Charger search radius (km) (for initial fetch)", 30.0, float)
    weather_override = input_with_default(" Skip live weather? (y/n)", "n", str).lower().startswith("y")

    # Build graph for bounding box around src/dst
    buffer_km = input_with_default(" Graph bbox buffer (km)", 10.0, float)
    print("\nConstructing graph (this may take some time)...")
    print("\nConstructing graph using district-level regions...")
    if os.path.exists("amaravati_vizag_corridor.graphml"):
        print("Loading cached road graph...")
        G = ox.load_graphml("amaravati_vizag_corridor.graphml")
    else:
        print("Building road graph from Overpass...")
        G = ox.graph_from_place(
        [
            "Krishna district, Andhra Pradesh, India",
            "Guntur district, Andhra Pradesh, India",
            "Bapatla district, Andhra Pradesh, India",

            "West Godavari district, Andhra Pradesh, India",
            "Eluru district, Andhra Pradesh, India",
            "East Godavari district, Andhra Pradesh, India",
            "Kakinada district, Andhra Pradesh, India",

            "Visakhapatnam district, Andhra Pradesh, India"

        ],
        network_type="drive"
    )
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, "amaravati_vizag_corridor.graphml")
    ox.save_graphml(G, "amaravati_vizag_corridor.graphml")
    print("Graph saved to amaravati_vizag_corridor.graphml")


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

    # Synthetic charger logs (if chargers exist)
    if not chargers.empty:
        rng = np.random.default_rng(7)
        logs = chargers[["station_id"]].copy()
        logs["total"] = rng.integers(5, 60, size=len(logs))
        logs["success"] = (logs["total"] * rng.uniform(0.6, 0.95, size=len(logs))).astype(int)
        logs["degradation"] = rng.uniform(0.85, 1.0, size=len(logs))
        charger_logs_df = logs
    else:
        charger_logs_df = pd.DataFrame(columns=["station_id", "total", "success", "degradation"])

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
    params = RoutingParams(soc_bins=soc_bins, lambda_risk=lambda_risk)

    print("\nRunning risk-aware routing...")
    result = risk_aware_route(
        G=G,
        source_node=source_node,
        target_node=target_node,
        chargers_df=chargers,
        charger_logs_df=charger_logs_df,
        temp_c=temp_c,
        wind_kmh=wind_kmh,
        vehicle=vehicle,
        params=params,
    )

    if not result["ok"]:
        print("Routing failed:", result["reason"])
        return

    print(f"\nBest risk-adjusted cost: {result['cost_minutes']:.1f} minutes")
    total_km = sum(a[3] for a in result["actions"] if a[0] == "DRIVE")
    print(f"Approx total distance (sum of edges): {total_km:.2f} km")
    print("Actions (first 200 shown):")
    for act in result["actions"][:200]:
        print(act)

    # done
    print("\nDone.")

if __name__ == "__main__":
    main()
