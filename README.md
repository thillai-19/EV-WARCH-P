# EV-WARCH-P

EV-Winter Aware Routing with Charger-Health Prediction (India)

## Overview

EV-WARCH-P is an experimental Electric Vehicle (EV) route-planning system designed for long-distance and inter-city travel in India.

Unlike conventional shortest-path routing, the system considers:

- Battery State of Charge (SOC)
- Minimum battery reserve
- Charging station availability
- Weather-dependent energy consumption
- Dynamic Wireless Power Transfer (DWPT)
- Charging time and route feasibility

The goal is to make EV routing more practical by considering real-world battery and charging constraints rather than distance alone.

## Key Features

- Road network generation using OpenStreetMap data via OSMnx
- Cached road graphs for faster repeated routing
- SOC-aware routing with configurable battery reserve
- Automatic charger discovery using the OpenChargeMap API
- Charger-to-road-node mapping for route planning
- Weather-aware energy consumption using temperature and wind
- Single-stop charging route planning
- Dynamic Wireless Power Transfer (DWPT) modeling
- Per-edge battery/SOC simulation along the route
- Detailed `DRIVE` and `CHARGE` route actions
- Predicted SOC exported as `route_soc`
- Route coordinates exported as `route_coords`
- Interactive route and charging visualization in the frontend

## How the Routing Works

The routing engine first determines whether the destination can be reached while maintaining the required battery reserve.

If charging is required, reachable charging stations are evaluated based on:

- Distance from the source
- Distance to the destination
- Energy required for each leg
- Available DWPT energy
- Required charging energy
- Estimated charging time

The charger producing the lowest total travel + charging time is selected.

The battery profile is then calculated along the route, accounting for energy consumed while driving and energy gained from DWPT segments.

## Weather-Aware Energy Model

Weather conditions influence the estimated energy consumption of the vehicle.

The model considers:

- Ambient temperature
- Wind speed
- HVAC energy requirements
- Base vehicle energy consumption

This produces a dynamic `kWh/km` estimate instead of assuming a fixed driving range.

## Dynamic Wireless Power Transfer (DWPT)

EV-WARCH-P supports experimental modeling of wireless charging roads.

DWPT segments can provide energy while the vehicle is driving. The system distributes the configured DWPT distance across the route and calculates the resulting energy gain.

The frontend visualizes DWPT road segments separately and provides an impact summary showing charging time and energy differences.

## Technology Stack

- Python
- OSMnx and NetworkX for road graphs and routing
- OpenChargeMap API for charging station data
- Open-Meteo API for weather information
- NumPy and Pandas for data processing
- Google Maps JavaScript API for frontend visualization

## Example Use Case

Routing an EV trip between Indian cities while ensuring:

- Minimum SOC reserve is maintained
- The vehicle can reach the selected charger
- Charging requirements are calculated dynamically
- Weather affects estimated energy consumption
- DWPT can contribute energy while driving
- The final route includes predicted SOC information

## Frontend Viewer

`index.html` provides an interactive map-based route viewer.

It displays:

- Route polyline
- Source and destination markers
- Charging-stop markers
- Predicted charging time
- Predicted battery percentage when hovering over the route
- DWPT road segments
- DWPT impact summary

## Running the Project

Start the backend and frontend:

```bash
python3 main.py --serve
```

Then open:

```text
http://127.0.0.1:8000/index.html
```

## Controls

- **Source:** Automatically detected using GPS, entered by name, or selected on the map
- **Destination:** Enter a location or select it on the map
- **Battery:** Set the current battery percentage
- **DWPT km:** Configure the amount of wireless-charging road available
- **Go:** Calculate and display the route

## Output

The backend generates:

- `inputs.json` — latest routing request
- `route_output.json` — route data used by the frontend
- Cached GraphML files — reusable road networks

The route output contains:

- `route_coords`
- `route_soc`
- `actions`
- `dwpt_road_coords`
- DWPT and charging metadata

## Current Limitations

- Long-distance road graphs may take time to build on the first run
- Current routing supports a single charging stop
- DWPT segment locations are modeled rather than obtained from a real infrastructure dataset
- Charger reliability/health prediction is currently experimental and can be extended with historical charger performance data

## Planned Improvements

- Multi-stop charging for very long routes
- More efficient routing for long inter-state trips
- Real charger-health/failure prediction using historical data
- Improved weather forecasting along the complete route
- More detailed SOC and energy visualizations
- Integration of real DWPT infrastructure data

## Motivation

EV-WARCH-P explores how EV navigation can move beyond shortest-path routing toward practical, constraint-aware travel planning.

By combining road networks, battery SOC, charging infrastructure, weather effects, and emerging technologies such as DWPT, the project aims to build a more realistic EV routing system for long-distance travel in India.
