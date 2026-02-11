# EV-WARCH-P
EV-Winter Aware Routing with Charger-Health Prediction (India)

**Overview**

This project is an experimental Electric Vehicle (EV) route-planning system designed for long-distance and inter-city travel in India. Unlike conventional shortest-path routing, the system incorporates battery State of Charge (SOC), charging infrastructure, and uncertainty in charger reliability to compute risk-aware routes.

The objective is to move toward more realistic EV navigation by accounting for range constraints, charging availability, weather-dependent energy consumption, and failure risk.

**Key Features**

- Road network generation using OpenStreetMap data via OSMnx (point-radius graph build with caching)
- SOC-aware range checks with battery reserve constraints
- Charging station integration using the OpenChargeMap API
- Weather-aware energy consumption modeling (HVAC and wind effects)
- Route output includes detailed per-edge DRIVE actions and CHARGE events
- Route coordinates are exported to `route_output.json` for external map rendering
- Predicted SOC per route node is exported as `route_soc` for hover display

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

The UI drives inputs now. If you want to run a single request via CLI, populate `inputs.json` and run:

```bash
python3 main.py
```

Example `inputs.json`:

```json
{
  "src_lat": 12.9716,
  "src_lon": 77.5946,
  "dst_lat": 12.2958,
  "dst_lon": 76.6394,
  "src_name": "Bengaluru",
  "dst_name": "Mysuru",
  "current_soc": 0.3,
  "temp_c": 28.0,
  "wind_kmh": 10.0
}
```

**Frontend Viewer**

`index.html` calls the backend and renders:

- A green route polyline
- Green charger markers for each CHARGE event
- Hover shows predicted battery percentage along the route

Run the backend server (serves the UI and handles routing):

```bash
python3 main.py --serve
```

Then open the UI:

```bash
open http://127.0.0.1:8000/index.html
```

Controls:

- Source: auto-detected by GPS or enter a source name / click the map after focusing the Source input (yellow marker).
- Destination: enter a destination name or click the map (red marker).
- Battery: type % in the battery input.
- Click **Go** to compute a route, render the polyline, and show charge markers with time labels.

The backend writes:

- `inputs.json` (latest request)
- `route_output.json` (latest route for the UI)
- cached graphs `graph_<source>_to_<dest>_rXXkm.graphml`

Stopping the server:

- Press `Ctrl+C` in the terminal.

If you prefer a simple static server, you can still serve the folder locally:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/index.html`.

**Current Limitations**

- Long-distance graphs may still take time to build on first run
- Charger selection is single-stop (one charge) in the current routing mode

**Planned Improvements**

- Multi-stop charging for very long routes
- Performance optimization for long inter-state routes
- Route visualization and SOC profiling

**Motivation**

This project explores realistic EV routing under real-world constraints, particularly in regions with limited and variable charging infrastructure. It aims to bridge the gap between theoretical shortest-path routing and practical EV travel planning.
