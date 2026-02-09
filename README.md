# EV-WARCH-P
EV-Winter Aware Routing with Charger-Health Prediction (India)

**Overview**

This project is an experimental Electric Vehicle (EV) route-planning system designed for long-distance and inter-city travel in India.
Unlike conventional shortest-path routing, the system incorporates battery State of Charge (SOC), charging infrastructure, and uncertainty in charger reliability to compute risk-aware routes.

The objective is to move toward more realistic EV navigation by accounting for range constraints, charging availability, weather-dependent energy consumption, and failure risk.

**Key Features**

- Road network generation using OpenStreetMap data via OSMnx (point-radius graph build with caching)
- SOC-aware range checks with battery reserve constraints
- Charging station integration using the OpenChargeMap API
- Weather-aware energy consumption modeling (HVAC and wind effects)
- Route output includes detailed per-edge DRIVE actions and CHARGE events
- Route coordinates are exported to `route_output.json` for external map rendering

**Technology Stack**

- Python
- OSMnx and NetworkX for road graphs and routing
- OpenChargeMap API for charging station data
- Open-Meteo API for live weather information
- NumPy and Pandas for data processing

**Example Use Case**

Routing an EV trip from Vijayawada to Hyderabad while ensuring:

- Minimum SOC reserve is maintained
- Charging stops are considered when required
- Total travel cost accounts for both time and charging risk

**Sample Input**

Labeled reference (human-friendly):

```text
place: Bengaluru, Karnataka, India
source_lat: 12.9716
source_lon: 77.5946
dest_lat: 12.2958
dest_lon: 76.6394
battery_kwh: 60.0
current_soc: 0.20
min_soc_reserve: 0.10
base_kwh_per_km: 0.20
charge_power_kw: 50.0
target_cabin_temp_c: 18.0
charger_search_radius_km: 25.0
skip_live_weather: n
graph_buffer_km: 20.0
```

Files:

- `sample_input.txt` (raw input lines for the CLI)

One-line copy-paste (pipes into stdin, raw order):

```bash
python3 main.py <<'EOF'
Bengaluru, Karnataka, India
12.9716
77.5946
12.2958
76.6394
60.0
0.20
0.10
0.20
50.0
18.0
25.0
n
20.0
EOF
```

**Current Limitations**

- Long-distance graphs may still take time to build on first run
- Charger selection is single-stop (one charge) in the current routing mode
- External map visualization requires consuming `route_output.json`

**Planned Improvements**

- Multi-stop charging for very long routes
- Improved charger discovery along the entire route
- Performance optimization for long inter-state routes
- Route visualization and SOC profiling
- Integration of live charger availability data

**Motivation**

This project explores realistic EV routing under real-world constraints, particularly in regions with limited and variable charging infrastructure. It aims to bridge the gap between theoretical shortest-path routing and practical EV travel planning.
