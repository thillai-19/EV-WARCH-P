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
from http.server import HTTPServer, SimpleHTTPRequestHandler

OCM_API_KEY = os.environ.get("OPENCHARGEMAP_API_KEY")
DEFAULT_OCM_API_KEY = "86ff4b4a-e7f8-42bd-a259-0f326f4a8d83"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Configuration dataclasses
# ----------------------------
@dataclass
class VehicleSpec:
    battery_kwh: float = 60.0
    base_kwh_per_km: float = 0.16          # nominal traction (adjust as needed)
    min_soc_reserve: float = 0.10          # reserve SOC (10%)
    max_soc: float = 1.00                  # stop charging at 100%
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


def _slugify(name: str) -> str:
    keep = []
    for ch in name.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in [" ", "-", "_"]:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:60] if slug else "unknown"


def cache_key(lat1, lon1, lat2, lon2, buffer_km, src_name=None, dst_name=None):
    buffer_bucket = int(round(buffer_km / 25.0) * 25)
    if src_name and dst_name:
        return f"graph_{_slugify(src_name)}_to_{_slugify(dst_name)}_r{buffer_bucket}km.graphml"
    mid_lat = (lat1 + lat2) / 2.0
    mid_lon = (lon1 + lon2) / 2.0
    q = 0.1
    mid_lat = round(mid_lat / q) * q
    mid_lon = round(mid_lon / q) * q
    return f"graph_mid_{mid_lat:.2f}_{mid_lon:.2f}_r{buffer_bucket}km.graphml"


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

    def battery_profile_for_nodes(nodes: List[int], start_soc: float) -> List[float]:
        socs = [float(np.clip(start_soc, 0.0, 1.0))]
        cur_kwh = vehicle.battery_kwh * socs[0]
        for i in range(len(nodes) - 1):
            dist_km = edge_length_km(nodes[i], nodes[i + 1])
            cur_kwh -= dist_km * kwh_per_km
            socs.append(float(np.clip(cur_kwh / vehicle.battery_kwh, 0.0, 1.0)))
        return socs

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
            "route_soc": battery_profile_for_nodes(nodes, start_soc),
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
    start_soc = float(np.clip(vehicle.current_soc, 0.0, 1.0))
    soc_a = battery_profile_for_nodes(nodes_a, start_soc)
    soc_at_charge = soc_a[-1] if soc_a else start_soc
    soc_after_charge = vehicle.max_soc
    soc_b = battery_profile_for_nodes(nodes_b, soc_after_charge)
    full_soc = soc_a + (soc_b[1:] if soc_b else [])
    return {
        "ok": True,
        "cost_minutes": best["cost_minutes"],
        "actions": drive_actions_a + [charge_action] + drive_actions_b,
        "route_nodes": full_nodes,
        "route_coords": full_coords,
        "route_soc": full_soc,
    }


def _sanitize_json(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json(v) for v in obj]
    return obj


def run_route(
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    current_soc: float,
    src_name: Optional[str] = None,
    dst_name: Optional[str] = None,
    temp_c: Optional[float] = None,
    wind_kmh: Optional[float] = None,
    battery_kwh: float = 60.0,
    min_soc_reserve: float = 0.10,
    base_kwh_per_km: float = 0.16,
    charge_power_kw: float = 50.0,
    target_cabin_temp: float = 22.0,
    charger_search_radius_km: float = 30.0,
    buffer_km: float = 10.0,
) -> Dict:
    straight_km = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
    min_buffer_km = straight_km / 2.0 + 10.0
    if buffer_km < min_buffer_km:
        buffer_km = min_buffer_km

    cache_name = cache_key(src_lat, src_lon, dst_lat, dst_lon, buffer_km, src_name, dst_name)
    cache_path = os.path.join(BASE_DIR, cache_name)
    if os.path.exists(cache_path):
        G = ox.load_graphml(cache_path)
    else:
        center_lat = (src_lat + dst_lat) / 2.0
        center_lon = (src_lon + dst_lon) / 2.0
        dist_m = max(500, int(buffer_km * 1000))
        G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="drive")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, cache_path)

    source_node = ox.distance.nearest_nodes(G, X=src_lon, Y=src_lat)
    target_node = ox.distance.nearest_nodes(G, X=dst_lon, Y=dst_lat)

    if temp_c is None or wind_kmh is None:
        try:
            temp_c, wind_kmh = get_weather_open_meteo(src_lat, src_lon)
        except Exception:
            temp_c, wind_kmh = 25.0, 5.0

    api_key = OCM_API_KEY or DEFAULT_OCM_API_KEY
    chargers = get_chargers_open_charge_map(
        src_lat, src_lon, radius_km=charger_search_radius_km, max_results=200, api_key=api_key
    )
    mid_lat = (src_lat + dst_lat) / 2
    mid_lon = (src_lon + dst_lon) / 2
    chargers_mid = get_chargers_open_charge_map(
        mid_lat, mid_lon, radius_km=charger_search_radius_km, max_results=80, api_key=api_key
    )
    chargers = pd.concat([chargers, chargers_mid], ignore_index=True)

    if not chargers.empty:
        chargers["lat"] = pd.to_numeric(chargers["lat"], errors="coerce")
        chargers["lon"] = pd.to_numeric(chargers["lon"], errors="coerce")
        chargers = chargers.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        chargers["graph_node"] = snap_points_to_graph(
            G, chargers["lat"].values.astype(float), chargers["lon"].values.astype(float)
        )

    vehicle = VehicleSpec(
        battery_kwh=battery_kwh,
        base_kwh_per_km=base_kwh_per_km,
        min_soc_reserve=min_soc_reserve,
        max_soc=0.90,
        charge_power_kw=charge_power_kw,
        target_cabin_temp_c=target_cabin_temp,
        current_soc=current_soc,
    )

    result = plan_single_charge_route(
        G=G,
        source_node=source_node,
        target_node=target_node,
        chargers_df=chargers,
        temp_c=temp_c,
        wind_kmh=wind_kmh,
        vehicle=vehicle,
    )
    print(f"Routing ok: {result.get('ok')}")
    if not result.get("ok"):
        print("Routing failed:", result.get("reason"))

    if result.get("ok"):
        if "route_soc" not in result or not result.get("route_soc"):
            # Fallback: compute SOC from actions if missing
            kwh_per_km = compute_kwh_per_km(temp_c, wind_kmh, vehicle)
            reserve_kwh = vehicle.battery_kwh * vehicle.min_soc_reserve
            cur_kwh = vehicle.battery_kwh * float(np.clip(vehicle.current_soc, 0.0, 1.0))
            socs = [float(np.clip(cur_kwh / vehicle.battery_kwh, 0.0, 1.0))]
            for act in result.get("actions", []):
                if act[0] == "DRIVE":
                    dist_km = float(act[3])
                    cur_kwh -= dist_km * kwh_per_km
                    socs.append(float(np.clip(cur_kwh / vehicle.battery_kwh, 0.0, 1.0)))
                elif act[0] == "CHARGE":
                    cur_kwh = vehicle.battery_kwh * vehicle.max_soc
                    socs.append(float(np.clip(cur_kwh / vehicle.battery_kwh, 0.0, 1.0)))
            result["route_soc"] = socs

        route_out = _sanitize_json(
            {
                "route_coords": result.get("route_coords", []),
                "actions": result.get("actions", []),
                "route_soc": result.get("route_soc", []),
                "meta": {
                    "src": [src_lat, src_lon],
                    "dst": [dst_lat, dst_lon],
                    "temp_c": temp_c,
                    "wind_kmh": wind_kmh,
                    "src_name": src_name,
                    "dst_name": dst_name,
                },
            }
        )
        with open(os.path.join(BASE_DIR, "route_output.json"), "w", encoding="utf-8") as f:
            json.dump(route_out, f, ensure_ascii=False)
        print("Saved route_output.json")

    return result


class RouteHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/route":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON")
            return

        try:
            src_lat = float(payload["src_lat"])
            src_lon = float(payload["src_lon"])
            dst_lat = float(payload["dst_lat"])
            dst_lon = float(payload["dst_lon"])
        except Exception:
            self.send_error(400, "Missing or invalid coordinates")
            return

        current_soc = float(payload.get("current_soc", 0.80))
        src_name = payload.get("src_name")
        dst_name = payload.get("dst_name")
        temp_c = payload.get("temp_c", None)
        wind_kmh = payload.get("wind_kmh", None)
        buffer_km = float(payload.get("buffer_km", 10.0))
        charger_search_radius_km = float(payload.get("charger_search_radius_km", 30.0))

        with open(os.path.join(BASE_DIR, "inputs.json"), "w", encoding="utf-8") as f:
            json.dump(_sanitize_json(payload), f, ensure_ascii=False)
        print("Saved inputs.json")

        print(
            f"Route request: src=({src_lat},{src_lon}) dst=({dst_lat},{dst_lon}) "
            f"soc={current_soc} temp_c={temp_c} wind_kmh={wind_kmh}"
        )

        result = run_route(
            src_lat=src_lat,
            src_lon=src_lon,
            dst_lat=dst_lat,
            dst_lon=dst_lon,
            current_soc=current_soc,
            src_name=src_name,
            dst_name=dst_name,
            temp_c=float(temp_c) if temp_c is not None else None,
            wind_kmh=float(wind_kmh) if wind_kmh is not None else None,
            charger_search_radius_km=charger_search_radius_km,
            buffer_km=buffer_km,
        )
        body = json.dumps(_sanitize_json(result)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if not result.get("ok"):
            print("Stopping server due to routing failure.")
            os._exit(1)


def serve(host: str = "127.0.0.1", port: int = 8000):
    def handler(*args, **kwargs):
        return RouteHandler(*args, directory=BASE_DIR, **kwargs)
    httpd = HTTPServer((host, port), handler)
    print(f"Serving on http://{host}:{port}")
    httpd.serve_forever()


# ----------------------------
# Demo runner (interactive with defaults)
# ----------------------------
def main():
    print("EV Risk-Aware Router — CLI mode\n")

    inputs_path = os.path.join(BASE_DIR, "inputs.json")
    if not os.path.exists(inputs_path):
        print("inputs.json not found. Use the web UI with `python3 main.py --serve`.")
        return

    with open(inputs_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    src_lat = float(payload["src_lat"])
    src_lon = float(payload["src_lon"])
    dst_lat = float(payload["dst_lat"])
    dst_lon = float(payload["dst_lon"])

    # Vehicle defaults (UI provides these in serve mode).
    battery_kwh = float(payload.get("battery_kwh", 60.0))          # kWh
    current_soc = float(payload.get("current_soc", 0.80))          # 0.0 - 1.0
    min_soc_reserve = float(payload.get("min_soc_reserve", 0.10))  # 0.0 - 1.0
    base_kwh_per_km = float(payload.get("base_kwh_per_km", 0.16))  # kWh/km
    charge_power_kw = float(payload.get("charge_power_kw", 50.0))  # kW
    target_cabin_temp = float(payload.get("target_cabin_temp", 22.0))  # Celsius

    # Routing defaults (UI provides these in serve mode).
    charger_search_radius_km = float(payload.get("charger_search_radius_km", 30.0))
    weather_override = False
    buffer_km = float(payload.get("buffer_km", 10.0))
    straight_km = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
    min_buffer_km = straight_km / 2.0 + 10.0
    if buffer_km < min_buffer_km:
        print(f"Auto-adjusting buffer_km from {buffer_km:.1f} to {min_buffer_km:.1f} km")
        buffer_km = min_buffer_km
    print("\nConstructing graph (this may take some time)...")
    print("\nConstructing graph using point-radius...")
    cache_name = cache_key(src_lat, src_lon, dst_lat, dst_lon, buffer_km, payload.get("src_name"), payload.get("dst_name"))
    cache_path = os.path.join(BASE_DIR, cache_name)
    if os.path.exists(cache_path):
        print(f"Loading cached graph: {cache_name}")
        G = ox.load_graphml(cache_path)
    else:
        center_lat = (src_lat + dst_lat) / 2.0
        center_lon = (src_lon + dst_lon) / 2.0
        dist_m = max(500, int(buffer_km * 1000))
        G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="drive")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, cache_path)
        print(f"Graph saved to {cache_name}")


    # Snap source/target to nearest graph nodes
    source_node = ox.distance.nearest_nodes(G, X=src_lon, Y=src_lat)
    target_node = ox.distance.nearest_nodes(G, X=dst_lon, Y=dst_lat)

    # Weather
    temp_c = payload.get("temp_c")
    wind_kmh = payload.get("wind_kmh")
    if temp_c is None or wind_kmh is None:
        try:
            temp_c, wind_kmh = get_weather_open_meteo(src_lat, src_lon)
        except Exception as e:
            print(f"Warning: weather fetch failed ({e}); using defaults.", file=sys.stderr)
            temp_c, wind_kmh = 25.0, 5.0

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
            "route_soc": result.get("route_soc", []),
        }
        with open(os.path.join(BASE_DIR, "route_output.json"), "w", encoding="utf-8") as f:
            json.dump(route_out, f, ensure_ascii=False)
        print("Saved route_output.json")
    print("Actions (first 200 shown):")
    for act in result["actions"][:200]:
        print(act)

    # done
    print("\nDone.")

if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
    else:
        main()
