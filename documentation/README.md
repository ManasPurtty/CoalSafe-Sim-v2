# CoalSafe-Sim Version 1

> **Disclaimer**
>
> CoalSafe-Sim Version 1 is a physics-informed synthetic dataset developed
> for algorithm development and prototype evaluation. It is not a measured
> coal-mine dataset and does not represent a particular mine, coal grade,
> stockpile, or operational condition. Numerical parameters, risk classes,
> and anomaly thresholds are simulation parameters and must not be interpreted
> as universal ignition thresholds, gas safety limits, or operational safety
> limits. The simulator represents qualitative and parametric relationships
> between low-temperature oxidation, heat accumulation, gas evolution, heat
> migration, surface thermal signatures, and risk. Real-world deployment
> requires controlled experimental calibration, field validation, and
> site-specific engineering assessment.

---

## 📌 Project Overview

**CoalSafe-Sim Version 1** is a physics-informed synthetic multimodal dataset and interactive warning system designed to simulate early sub-surface self-heating and spontaneous combustion risk in coal stockpiles. 

Because spontaneous combustion originates deep within coal stockpiles due to low-temperature oxidation, traditional surface monitoring often detects fire too late. This project models the hidden physical progression (3-depth thermal layers, gas evolution kinetics, spatial thermal surface anomalies) to enable early-warning AI research and proactive autonomous mitigation.

---

## 🏗️ Project Structure

```text
CoalSafe-Sim-v2/
├── code/
│   ├── config.py                   # Central simulation parameters & physics coefficients
│   ├── generate_scenarios.py       # 8 scenario parameter configurations & time axis
│   ├── generate_latent_state.py    # Hidden sub-surface physics engine (3 depth layers)
│   ├── generate_observations.py    # Observable sensor data, noise, & synthetic risk scores
│   ├── generate_thermal_images.py  # 50×50 spatial PNG thermal heatmap generator
│   ├── build_dashboard.py          # Standalone web control center builder script
│   ├── validate_dataset.py         # Automated (17) & causal (8) dataset quality checkers
│   └── run_all.py                  # Master pipeline orchestrator
├── dashboard/
│   └── index.html                  # Interactive Early-Risk Score & Predictive Warning Web App
├── data/
│   ├── tabular/
│   │   └── sensor_data.csv         # Observable sensor telemetry for AI models (968 rows)
│   ├── thermal/
│   │   ├── S01/ ... S08/           # 952 false-color 50×50 spatial thermal PNG images
│   └── metadata/
│       ├── scenarios.csv           # Scenario starting conditions & random seeds
│       ├── latent_state.csv        # Hidden internal physical state (for auditing only)
│       └── thermal_metadata.csv    # Spatial thermal metadata & hotspot cell coverage
├── graphs/                         # 8 validation plots (internal vs surface, risk, gases)
├── documentation/
│   └── README.md                   # Technical documentation
└── README.md                       # Main project README
```

---

## ⚡ Quick Start & Execution

### 1. Generate Complete Dataset & Validation Plots
To execute the complete simulation pipeline, generate all 968 tabular rows, 952 thermal images, and 8 validation graph plots:
```bash
python code/run_all.py
```

### 2. Validate Dataset Integrity
Run the automated test suite (17 automated checks + 8 causal physical laws):
```bash
python code/validate_dataset.py
```

### 3. Build & Launch Interactive Web Control Center Dashboard
To compile and launch the interactive Web Control Center dashboard:
```bash
python code/build_dashboard.py
# Open dashboard/index.html in your web browser, or:
Start-Process "dashboard/index.html"
```

---

## 🏛️ System Architecture Alignment (Layers 1–6)

The project and dashboard implement the 6-layer architecture specified for spontaneous combustion forecasting:

1. **Layer 1: Hidden Sub-Surface State** — Low-temperature oxidation rate, reaction kinetics, and internal temperature distribution deep within the pile.
2. **Layer 2: Physical Progression** — 3-depth physical layer model (Deep Core $\rightarrow$ Middle Layer $\rightarrow$ Near-Surface) simulating upward heat migration with a temporal delay.
3. **Layer 3: Observable Signatures** — Surface temperature, 50×50 spatial thermal heatmaps, and gas sensors ($CO$, $CO_2$, $O_2$).
4. **Layer 4: Intelligence Layer** — Multimodal feature extraction, synchronization, and temporal modeling.
5. **Layer 5: Early-Risk Score & Predictive Warning** — Current Risk Donut Gauge (0–100%), Risk Trajectory Forecast (+1h, +2h, +4h, +6h predictive trajectory), and Warning Level banners.
6. **Layer 6: Action Layer / Automated Mitigation System** — Autonomous intervention triggers (Water/Foam Spray, Airflow Management, Stack Reconfiguration, Conveyor Control).

---

## 🧪 Scenarios Summary

All 8 scenarios cover 121 minutes of continuous simulation (timestamp 0 to 120):

| ID  | Scenario Type           | Description |
|-----|-------------------------|-------------|
| S01 | `STABLE`                | No meaningful self-heating. Baseline reference. |
| S02 | `EARLY_OXIDATION`       | Hidden sub-surface oxidation rises while surface remains normal. |
| S03 | `DEVELOPING_CORE`       | Sub-surface thermal core grows; internal & gas signals change. |
| S04 | `HEAT_MIGRATION`        | Heat reaches surface; thermal anomaly begins. |
| S05 | `HIGH_RISK`             | Thermal core and all observable signals continue increasing to critical. |
| S06 | `EARLY_MITIGATION`      | Water/foam spray intervention occurs early while event is developing. |
| S07 | `LATE_MITIGATION`       | Intervention occurs late after stronger thermal core development. |
| S08 | `ENVIRONMENTAL_VARIATION` | Environmental weather changes (wind, humidity, ambient) alter heat balance. |

---

## 📊 Variables Reference

### Observable Features (`data/tabular/sensor_data.csv`) — For AI Training

| Variable | Type | Description |
|----------|------|-------------|
| `scenario_id` | String | Scenario identifier (`S01` to `S08`) |
| `timestamp_min` | Integer | Simulation minute (0 to 120) |
| `ambient_temperature` | Float | Outside environmental temperature (°C) |
| `humidity` | Float | Relative humidity (%) |
| `wind_speed` | Float | Wind speed (m/s) |
| `coal_moisture` | Float | Coal moisture content (%) |
| `CO` | Float | Carbon monoxide concentration (ppm) |
| `CO2` | Float | Carbon dioxide concentration (ppm) |
| `O2` | Float | Atmospheric Oxygen level (%) |
| `surface_temperature_mean` | Float | Average surface grid temperature (°C) |
| `surface_temperature_max` | Float | Peak surface hotspot temperature (°C) |
| `hotspot_area` | Integer | Count of grid cells exceeding anomaly threshold |
| `hotspot_x`, `hotspot_y` | Float | Spatial centroid coordinates of the hotspot |
| `heating_rate` | Float | Surface temperature rate of change per minute |
| `risk_score` | Float | Synthetic risk score (0 to 100%) |
| `risk_class` | String | `NORMAL` / `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| `mitigation_status` | Integer | `0` (inactive) or `1` (automated spray active) |

### Hidden Latent Features (`data/metadata/latent_state.csv`) — Kept Separate for Auditing

| Variable | Description |
|----------|-------------|
| `oxidation_state` | Oxidation progression (0 to 1) |
| `oxidation_rate` | Current rate of chemical oxidation |
| `internal_temperature` | Deep core temperature (°C) |
| `heat_generation` | Heat energy produced per timestep |
| `heat_loss` | Heat energy dissipated per timestep |
| `deep_core_temperature` | Layer 0 (deepest core) temperature |
| `middle_core_temperature` | Layer 1 (middle depth) temperature |
| `near_surface_temperature` | Layer 2 (near surface) temperature |

---

## 🎯 Synthetic Risk Classification

```text
risk_score = w1 * oxidation_severity
           + w2 * internal_thermal_severity
           + w3 * heating_rate_severity
           + w4 * CO_trend_severity
           + w5 * CO2_trend_severity
           + w6 * O2_depletion_severity
           + w7 * surface_anomaly_severity
           + w8 * hotspot_growth_severity
```

| Score Range | Risk Class | Recommended Action |
|-------------|------------|--------------------|
| 0 – 19.99   | `NORMAL`   | Routine Monitoring Loop |
| 20 – 39.99  | `LOW`      | Inspect Sub-Surface Sensor Telemetry |
| 40 – 59.99  | `MEDIUM`   | Increase Airflow & Monitor Hotspot Growth |
| 60 – 79.99  | `HIGH`     | Automated Preventive Water Mitigation Triggered |
| 80 – 100.0  | `CRITICAL` | Emergency Water/Foam Spray & Stack Reconfiguration |

---

## 🔁 Reproducibility & Scientific Basis

- **Global Random Seed:** `RANDOM_SEED = 42` locked in `code/config.py`.
- **References:**
  - Onifade & Genc (2020), *International Journal of Mining Science and Technology*.
  - Fuel review on thermal-kinetic, heat and mass transport in coal spontaneous combustion (2022).
  - Wang, Dlugogorski & Kennedy (2003), *Progress in Energy and Combustion Science*.
  - Review of index gases for spontaneous-combustion forecasting (2019).
