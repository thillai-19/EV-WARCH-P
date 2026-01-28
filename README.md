# EV-WARCH-P
EV-Winter Aware Routing with Charger-Health Prediction (India)

**Overview**

This project is an experimental Electric Vehicle (EV) route-planning system designed for long-distance and inter-city travel in India.
Unlike conventional shortest-path routing, the system incorporates battery State of Charge (SOC), charging infrastructure, and uncertainty in charger reliability to compute risk-aware routes.

The objective is to move toward more realistic EV navigation by accounting for range constraints, charging availability, weather-dependent energy consumption, and failure risk.

**Key Features**

Road network generation using OpenStreetMap data via OSMnx

SOC-aware routing with battery and reserve constraints

Charging station integration using the OpenChargeMap API

Weather-aware energy consumption modeling (HVAC and wind effects)

Risk-adjusted routing that penalizes unreliable charging stations

Multi-state routing where each node represents both location and SOC

Support for cached offline road graphs

**Technology Stack**

Python

OSMnx and NetworkX for road graphs and routing

OpenChargeMap API for charging station data

Open-Meteo API for live weather information

NumPy and Pandas for data processing

**Example Use Case**

Routing an EV trip from Vijayawada to Hyderabad while ensuring:

Minimum SOC reserve is maintained

Charging stops are considered when required

Total travel cost accounts for both time and charging risk

**Current Limitations**

Known Issue: Charging Action Visibility

The routing engine internally considers charging actions during path computation; however, charging steps are not currently displayed explicitly in the final route action output.

This is a known issue and will be addressed in a future update.

**Planned Improvements**

Inclusion of charging actions in route output

Improved charger discovery along the entire route

Performance optimization for long inter-state routes

Route visualization and SOC profiling

Integration of live charger availability data

**Motivation**

This project explores realistic EV routing under real-world constraints, particularly in regions with limited and variable charging infrastructure. It aims to bridge the gap between theoretical shortest-path routing and practical EV travel planning.
