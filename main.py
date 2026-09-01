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

DWPT_POWER_KW   = 20.0   # Realistic wireless charging power (ElectReon/eRoadArlanda spec)
DWPT_SPEED_KMH  = 40.0   # Speed limit on DWPT road segments
OVERPASS_ENDPOINTS = [
    url
    for url in dict.fromkeys(
        [
            os.environ.get("OVERPASS_URL", "").strip(),
            "https://overpass-api.de/api",
            "https://lz4.overpass-api.de/api",
            "https://z.overpass-api.de/api",
        ]
    )
    if url
]

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

def build_dwpt_schedule(total_route_km: float, dwpt_km: float) -> list:
    """
    Scatter dwpt_km across the route in equal chunks with random 10-15km gaps.
    Returns a list of (start_km, end_km) tuples representing DWPT active segments.
    One chunk per 50km of route. If route too short, fit as many as possible.
    """
    import random
    if dwpt_km <= 0.0 or total_route_km <= 0.0:
        return []

    # auto-calculate number of chunks based on route length
    num_chunks = max(1, int(total_route_km / 50.0))
    chunk_size_km = dwpt_km / num_chunks

    segments = []
    cursor_km = 5.0  # start first chunk 5km into the route (not at the very start)

    for i in range(num_chunks):
        start_km = cursor_km
        end_km = start_km + chunk_size_km

        # if this chunk goes beyond the route, stop here
        if start_km >= total_route_km:
            break
        end_km = min(end_km, total_route_km)
        segments.append((start_km, end_km))

        # random gap of 10-15km before next chunk
        gap_km = random.uniform(10.0, 15.0)
        cursor_km = end_km + gap_km

    return segments

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


def load_or_build_graph(cache_path: str, center_lat: float, center_lon: float, dist_m: int):
    if os.path.exists(cache_path):
        return ox.load_graphml(cache_path)

    last_error = None
    ox.settings.requests_timeout = max(getattr(ox.settings, "requests_timeout", 180), 180)
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            ox.settings.overpass_url = endpoint
            print(f"Trying Overpass endpoint: {endpoint}")
            G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="drive")
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            ox.save_graphml(G, cache_path)
            print(f"Graph saved to {cache_path}")
            return G
        except Exception as e:
            last_error = e
            print(f"Overpass endpoint failed: {endpoint} ({e})", file=sys.stderr)

    raise RuntimeError(f"Unable to download road graph from Overpass: {last_error}")


def plan_single_charge_route(
    G: nx.MultiDiGraph,
    source_node: int,
    target_node: int,
    chargers_df: pd.DataFrame,
    temp_c: float,
    wind_kmh: float,
    vehicle: VehicleSpec,
    dwpt_km: float = 0.0,
    verbose: bool = True,
) -> Dict:
    """
    Plan: if destination is within current range, go direct.
    Otherwise, find all reachable chargers and pick the optimal one
    (shortest total distance) then drive -> charge -> drive.

    dwpt_km: total km of Dynamic Wireless Power Transfer road along the route.
    Since we only know the total DWPT distance and not the exact segments, we
    model it as being encountered progressively from the start of the route.
    """
    def log(message: str) -> None:
        if verbose:
            print(message)

    # --- Core vehicle energy variables (must be defined first) ---
    kwh_per_km   = compute_kwh_per_km(temp_c, wind_kmh, vehicle)
    reserve_kwh  = vehicle.battery_kwh * vehicle.min_soc_reserve
    start_kwh    = vehicle.battery_kwh * float(np.clip(vehicle.current_soc, 0.0, 1.0))
    max_soc_kwh  = vehicle.max_soc * vehicle.battery_kwh

    # --- DWPT energy-rate constant ---
    dwpt_km          = max(dwpt_km, 0.0)
    dwpt_kwh_per_km  = DWPT_POWER_KW / max(DWPT_SPEED_KMH, 1e-9)   # kWh per km at 40 km/h
    dwpt_kwh_gained  = dwpt_km * dwpt_kwh_per_km                    # theoretical max gain

    log(
        f"DWPT: {dwpt_km:.1f} km available "
        f"→ up to +{dwpt_kwh_gained:.2f} kWh while driving (scattered chunks)"
    )

    # Base range from actual SOC — DWPT credit applied per-edge later
    base_usable_kwh = max(start_kwh - reserve_kwh, 0.0)
    range_km        = base_usable_kwh / max(kwh_per_km, 1e-9)

    # DWPT schedule built after first pass gives us total_route_km
    dwpt_segments = []

    len_from_source = nx.single_source_dijkstra_path_length(G, source_node, weight="length")
    if target_node not in len_from_source:
        return {"ok": False, "reason": "No path to destination on road network."}

    dist_src_to_target_km = len_from_source[target_node] / 1000.0
    time_from_source = nx.single_source_dijkstra_path_length(G, source_node, weight="travel_time")
    time_to_target = nx.single_source_dijkstra_path_length(G.reverse(copy=False), target_node, weight="travel_time")

    log(f"Estimated range from current SOC: {range_km:.2f} km")
    log(f"Shortest path distance (source -> destination): {dist_src_to_target_km:.2f} km")

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

    def battery_profile_for_nodes(nodes: List[int], start_kwh_value: float, dwpt_segments: list, leg_offset_km: float = 0.0) -> Dict:
        """
        dwpt_segments: list of (start_km, end_km) tuples from build_dwpt_schedule
                       these are absolute positions from route start.
        leg_offset_km: how many km into the full route this leg starts at.
        """
        cur_kwh = float(np.clip(start_kwh_value, 0.0, max_soc_kwh))
        socs = [float(np.clip(cur_kwh / vehicle.battery_kwh, 0.0, 1.0))]
        gained_kwh = 0.0
        distance_km = 0.0

        for i in range(len(nodes) - 1):
            dist_km = edge_length_km(nodes[i], nodes[i + 1])
            edge_start = leg_offset_km + distance_km
            edge_end = edge_start + dist_km

            # check overlap of this edge with every DWPT segment
            dwpt_on_edge_km = 0.0
            for seg_start, seg_end in dwpt_segments:
                overlap_start = max(edge_start, seg_start)
                overlap_end = min(edge_end, seg_end)
                if overlap_end > overlap_start:
                    dwpt_on_edge_km += overlap_end - overlap_start

            gained_here_kwh = dwpt_on_edge_km * dwpt_kwh_per_km
            cur_kwh = min(cur_kwh + gained_here_kwh, max_soc_kwh)
            cur_kwh = max(cur_kwh - dist_km * kwh_per_km, 0.0)
            gained_kwh += gained_here_kwh
            distance_km += dist_km
            socs.append(float(np.clip(cur_kwh / vehicle.battery_kwh, 0.0, 1.0)))

        return {
            "socs": socs,
            "end_kwh": cur_kwh,
            "dwpt_gained_kwh": gained_kwh,
            "distance_km": distance_km,
        }

    direct_nodes = route_nodes(source_node, target_node)

    # first pass with no DWPT to get total route distance
    bare_profile = battery_profile_for_nodes(direct_nodes, start_kwh, [], 0.0)
    total_route_km = bare_profile["distance_km"]

    # now build scattered DWPT schedule based on real route length
    dwpt_segments = build_dwpt_schedule(total_route_km, dwpt_km)
    log(f"DWPT schedule: {len(dwpt_segments)} chunks across {total_route_km:.1f} km route")
    for seg in dwpt_segments:
        log(f"  DWPT active: {seg[0]:.1f} km – {seg[1]:.1f} km")

    def dwpt_coords_from_nodes(nodes: List[int], seg_list: list, leg_offset_km: float = 0.0) -> List[List[List[float]]]:
        """
        Convert DWPT segment km-positions into lists of [lat, lon] coordinate
        sequences that can be drawn as polylines on the map.

        Returns a list of segments, each segment being a list of [lat, lon] pairs.
        """
        if not seg_list:
            return []

        # Walk every edge, accumulate cumulative distance, collect coords for
        # any portion of an edge that overlaps a DWPT segment.
        result_segments: List[List[List[float]]] = []
        cursor_km = leg_offset_km

        for i in range(len(nodes) - 1):
            u = nodes[i]
            v = nodes[i + 1]
            dist_km = edge_length_km(u, v)
            edge_start = cursor_km
            edge_end   = cursor_km + dist_km

            u_lat = float(G.nodes[u]["y"])
            u_lon = float(G.nodes[u]["x"])
            v_lat = float(G.nodes[v]["y"])
            v_lon = float(G.nodes[v]["x"])

            for seg_start, seg_end in seg_list:
                overlap_start = max(edge_start, seg_start)
                overlap_end   = min(edge_end,   seg_end)
                if overlap_end <= overlap_start:
                    continue

                # Interpolate start/end coords along the edge
                if dist_km < 1e-9:
                    t_start, t_end = 0.0, 1.0
                else:
                    t_start = (overlap_start - edge_start) / dist_km
                    t_end   = (overlap_end   - edge_start) / dist_km

                p_start_lat = u_lat + t_start * (v_lat - u_lat)
                p_start_lon = u_lon + t_start * (v_lon - u_lon)
                p_end_lat   = u_lat + t_end   * (v_lat - u_lat)
                p_end_lon   = u_lon + t_end   * (v_lon - u_lon)

                # Try to merge with the last segment if it's the same DWPT band
                # and we're on consecutive edges (gap < 0.001 km)
                if result_segments:
                    last_seg = result_segments[-1]
                    last_pt  = last_seg[-1]
                    gap = abs(last_pt[0] - p_start_lat) + abs(last_pt[1] - p_start_lon)
                    if gap < 0.0001:          # ~10 m — same continuous segment
                        last_seg.append([p_end_lat, p_end_lon])
                        continue

                result_segments.append([
                    [p_start_lat, p_start_lon],
                    [p_end_lat,   p_end_lon],
                ])

            cursor_km = edge_end

        return result_segments

    direct_profile = battery_profile_for_nodes(direct_nodes, start_kwh, dwpt_segments, 0.0)
    if direct_profile["end_kwh"] >= reserve_kwh:
        time_min = (time_from_source.get(target_node, 0.0) or 0.0) / 60.0
        drive_actions = drive_actions_from_nodes(direct_nodes)
        dwpt_road_coords = dwpt_coords_from_nodes(direct_nodes, dwpt_segments, 0.0)
        return {
            "ok": True,
            "cost_minutes": time_min,
            "actions": drive_actions,
            "route_nodes": direct_nodes,
            "route_coords": nodes_to_coords(direct_nodes),
            "route_soc": direct_profile["socs"],
            "dwpt_kwh_gained": direct_profile["dwpt_gained_kwh"],
            "dwpt_segments": dwpt_segments,
            "dwpt_road_coords": dwpt_road_coords,
        }

    if chargers_df.empty:
        return {"ok": False, "reason": "Destination out of range and no chargers available."}

    len_to_target = nx.single_source_dijkstra_path_length(G.reverse(copy=False), target_node, weight="length")

    best = None
    charger_nodes = chargers_df["graph_node"].dropna().astype(int).tolist()

    # Minimum distance from start before allowing a charge stop.
    min_dist_from_start_km = 1.0

    # Helper: total DWPT kWh collected on a leg that spans [leg_start_km, leg_start_km+leg_dist_km]
    # of the full route.  Defined once here — not inside the loop.
    def kwh_from_segments_on_leg(seg_list, leg_start_km, leg_dist_km):
        gained = 0.0
        leg_end_km = leg_start_km + leg_dist_km
        for seg_start, seg_end in seg_list:
            overlap_start = max(leg_start_km, seg_start)
            overlap_end   = min(leg_end_km,   seg_end)
            if overlap_end > overlap_start:
                gained += (overlap_end - overlap_start) * dwpt_kwh_per_km
        return gained

    # Safe default: used on line 505 even if charger loop never executes
    dwpt_gained_leg2 = 0.0

    for c in charger_nodes:
        if c not in len_from_source or c not in len_to_target:
            continue
        dist1_km = len_from_source[c] / 1000.0
        dist2_km = len_to_target[c] / 1000.0
        if dist1_km <= min_dist_from_start_km:
            continue
        time1_min = (time_from_source.get(c, 0.0) or 0.0) / 60.0
        time2_min = (time_to_target.get(c, 0.0) or 0.0) / 60.0

        dwpt_gained_leg1 = kwh_from_segments_on_leg(dwpt_segments, 0.0, dist1_km)
        dwpt_gained_leg2 = kwh_from_segments_on_leg(dwpt_segments, dist1_km, dist2_km)

        kwh_after_drive = min(start_kwh + dwpt_gained_leg1, max_soc_kwh)
        kwh_after_drive = max(kwh_after_drive - dist1_km * kwh_per_km, 0.0)
        if kwh_after_drive < reserve_kwh:
            continue

        required_departure_kwh = reserve_kwh + max(
            dist2_km * kwh_per_km - dwpt_gained_leg2,
            0.0,
        )
        if required_departure_kwh > max_soc_kwh + 1e-9:
            continue

        add_kwh = max(required_departure_kwh - kwh_after_drive, 0.0)
        charge_time_min = 60.0 * add_kwh / max(vehicle.charge_power_kw, 1e-6)

        total_cost = time1_min + time2_min + charge_time_min
        total_dist = dist1_km + dist2_km

        cand = {
            "charger_node": c,
            "dist1_km": dist1_km,
            "dist2_km": dist2_km,
            "add_kwh": add_kwh,
            "required_departure_kwh": required_departure_kwh,
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

    drive_profile_a = battery_profile_for_nodes(nodes_a, start_kwh, dwpt_segments, 0.0)
    kwh_at_charge = drive_profile_a["end_kwh"]
    leg_a_dist_km = drive_profile_a["distance_km"]
    leg_b_dist_km = sum(float(act[3]) for act in drive_actions_b)

    # leg B starts at leg_a_dist_km into the route — segments apply correctly
    required_departure_kwh = reserve_kwh + max(
        leg_b_dist_km * kwh_per_km - dwpt_gained_leg2,
        0.0,
    )
    required_departure_kwh = min(max(required_departure_kwh, kwh_at_charge), max_soc_kwh)
    add_kwh = max(required_departure_kwh - kwh_at_charge, 0.0)
    charge_time_min = 60.0 * add_kwh / max(vehicle.charge_power_kw, 1e-6)
    charged_kwh = min(kwh_at_charge + add_kwh, max_soc_kwh)
    drive_profile_b = battery_profile_for_nodes(nodes_b, charged_kwh, dwpt_segments, leg_a_dist_km)
    time1_min = (time_from_source.get(best["charger_node"], 0.0) or 0.0) / 60.0
    time2_min = (time_to_target.get(best["charger_node"], 0.0) or 0.0) / 60.0

    log(
        "Chosen charger node: "
        f"{best['charger_node']} | dist1={best['dist1_km']:.2f} km | "
        f"dist2={best['dist2_km']:.2f} km | charge={add_kwh:.2f} kWh"
    )

    charge_action = ("CHARGE", best["charger_node"], add_kwh, charge_time_min, charge_coord)
    soc_a = drive_profile_a["socs"]
    soc_b = drive_profile_b["socs"]
    full_soc = soc_a + (soc_b[1:] if soc_b else [])

    # Build DWPT road coords for both legs combined
    dwpt_road_coords_a = dwpt_coords_from_nodes(nodes_a, dwpt_segments, 0.0)
    dwpt_road_coords_b = dwpt_coords_from_nodes(nodes_b, dwpt_segments, leg_a_dist_km)
    dwpt_road_coords   = dwpt_road_coords_a + dwpt_road_coords_b

    return {
        "ok": True,
        "cost_minutes": time1_min + time2_min + charge_time_min,
        "actions": drive_actions_a + [charge_action] + drive_actions_b,
        "route_nodes": full_nodes,
        "route_coords": full_coords,
        "route_soc": full_soc,
        "dwpt_kwh_gained": drive_profile_a["dwpt_gained_kwh"] + drive_profile_b["dwpt_gained_kwh"],
        "dwpt_road_coords": dwpt_road_coords,
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


def total_charge_time_minutes(route_result: Dict) -> float:
    return sum(
        float(action[3])
        for action in route_result.get("actions", [])
        if action and action[0] == "CHARGE"
    )


def total_charge_added_kwh(route_result: Dict) -> float:
    return sum(
        float(action[2])
        for action in route_result.get("actions", [])
        if action and action[0] == "CHARGE"
    )


def build_dwpt_summary_meta(
    route_result: Dict,
    baseline_result: Optional[Dict],
    vehicle: VehicleSpec,
    dwpt_km: float,
) -> Dict:
    charge_time_with_dwpt = total_charge_time_minutes(route_result)
    charge_kwh_with_dwpt = total_charge_added_kwh(route_result)
    charge_pct_with_dwpt = (
        charge_kwh_with_dwpt / max(vehicle.battery_kwh, 1e-9)
    ) * 100.0

    dwpt_kwh_gained = float(route_result.get("dwpt_kwh_gained", 0.0))
    dwpt_kwh_per_km = DWPT_POWER_KW / max(DWPT_SPEED_KMH, 1e-9)
    dwpt_km_used = dwpt_kwh_gained / max(dwpt_kwh_per_km, 1e-9)

    meta = {
        "dwpt_km": dwpt_km,
        "dwpt_km_used": round(min(dwpt_km_used, max(dwpt_km, 0.0)), 2),
        "dwpt_kwh_gained": round(dwpt_kwh_gained, 3),
        "charge_time_with_dwpt_min": round(charge_time_with_dwpt, 1),
        "charge_added_kwh_with_dwpt": round(charge_kwh_with_dwpt, 3),
        "charge_added_pct_with_dwpt": round(charge_pct_with_dwpt, 1),
        "dwpt_time_saved_min": 0.0,
        "dwpt_enabled_route": False,
    }

    if baseline_result and baseline_result.get("ok"):
        charge_time_without_dwpt = total_charge_time_minutes(baseline_result)
        charge_kwh_without_dwpt = total_charge_added_kwh(baseline_result)
        charge_pct_without_dwpt = (
            charge_kwh_without_dwpt / max(vehicle.battery_kwh, 1e-9)
        ) * 100.0

        meta.update(
            {
                "charge_time_without_dwpt_min": round(charge_time_without_dwpt, 1),
                "charge_added_kwh_without_dwpt": round(charge_kwh_without_dwpt, 3),
                "charge_added_pct_without_dwpt": round(charge_pct_without_dwpt, 1),
                "dwpt_time_saved_min": round(
                    max(charge_time_without_dwpt - charge_time_with_dwpt, 0.0),
                    1,
                ),
            }
        )
    elif dwpt_km > 0.0:
        # baseline failed (no charger reachable without DWPT) but DWPT made it work
        baseline_failed = baseline_result is not None and not baseline_result.get("ok")
        meta["dwpt_enabled_route"] = baseline_failed

    return meta


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
    dwpt_km: float = 0.0,
) -> Dict:
    straight_km = haversine_km(src_lat, src_lon, dst_lat, dst_lon)
    min_buffer_km = straight_km / 2.0 + 10.0
    if buffer_km < min_buffer_km:
        buffer_km = min_buffer_km

    cache_name = cache_key(src_lat, src_lon, dst_lat, dst_lon, buffer_km, src_name, dst_name)
    cache_path = os.path.join(BASE_DIR, cache_name)
    center_lat = (src_lat + dst_lat) / 2.0
    center_lon = (src_lon + dst_lon) / 2.0
    dist_m = max(500, int(buffer_km * 1000))
    try:
        G = load_or_build_graph(cache_path, center_lat, center_lon, dist_m)
    except Exception as e:
        return {
            "ok": False,
            "reason": (
                "Road network download failed. The route is large and the Overpass server "
                f"is unavailable right now. Details: {e}"
            ),
        }

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
        max_soc=0.99,
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
        dwpt_km=dwpt_km,
    )
    print(f"Routing ok: {result.get('ok')}")
    if not result.get("ok"):
        print("Routing failed:", result.get("reason"))

    if result.get("ok"):
        baseline_result = None
        if dwpt_km > 0.0:
            baseline_result = plan_single_charge_route(
                G=G,
                source_node=source_node,
                target_node=target_node,
                chargers_df=chargers,
                temp_c=temp_c,
                wind_kmh=wind_kmh,
                vehicle=vehicle,
                dwpt_km=0.0,
                verbose=False,
            )
        dwpt_meta = build_dwpt_summary_meta(result, baseline_result, vehicle, dwpt_km)

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
        result["meta"] = {
            "src": [src_lat, src_lon],
            "dst": [dst_lat, dst_lon],
            "temp_c": temp_c,
            "wind_kmh": wind_kmh,
            "src_name": src_name,
            "dst_name": dst_name,
            "battery_kwh": vehicle.battery_kwh,
            "charge_power_kw": vehicle.charge_power_kw,
            **dwpt_meta,
        }

        route_out = _sanitize_json(
            {
                "route_coords": result.get("route_coords", []),
                "actions": result.get("actions", []),
                "route_soc": result.get("route_soc", []),
                "dwpt_road_coords": result.get("dwpt_road_coords", []),
                "meta": result["meta"],
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
        dwpt_km = float(payload.get("dwpt_km", 0.0))

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
            dwpt_km=dwpt_km,
        )
        body = json.dumps(_sanitize_json(result)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if not result.get("ok"):
            print(f"Routing failed (served error to client): {result.get('reason')}")


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
    dwpt_km = float(payload.get("dwpt_km", 0.0))

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
    center_lat = (src_lat + dst_lat) / 2.0
    center_lon = (src_lon + dst_lon) / 2.0
    dist_m = max(500, int(buffer_km * 1000))
    try:
        if os.path.exists(cache_path):
            print(f"Loading cached graph: {cache_name}")
        G = load_or_build_graph(cache_path, center_lat, center_lon, dist_m)
    except Exception as e:
        print(
            "Graph build failed. The route may be too large for the current Overpass "
            f"server availability. Details: {e}",
            file=sys.stderr,
        )
        return


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
        max_soc=0.99,
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
        dwpt_km=dwpt_km,
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
        baseline_result = None
        if dwpt_km > 0.0:
            baseline_result = plan_single_charge_route(
                G=G,
                source_node=source_node,
                target_node=target_node,
                chargers_df=chargers,
                temp_c=temp_c,
                wind_kmh=wind_kmh,
                vehicle=vehicle,
                dwpt_km=0.0,
                verbose=False,
            )
        dwpt_meta = build_dwpt_summary_meta(result, baseline_result, vehicle, dwpt_km)

        print(f"Route nodes count: {len(result['route_coords'])}")
        route_out = {
            "route_coords": result["route_coords"],
            "actions": result["actions"],
            "route_soc": result.get("route_soc", []),
            "dwpt_road_coords": result.get("dwpt_road_coords", []),
            "meta": {
                "src": [src_lat, src_lon],
                "dst": [dst_lat, dst_lon],
                "temp_c": temp_c,
                "wind_kmh": wind_kmh,
                "src_name": payload.get("src_name"),
                "dst_name": payload.get("dst_name"),
                "battery_kwh": vehicle.battery_kwh,
                "charge_power_kw": vehicle.charge_power_kw,
                **dwpt_meta,
            },
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