"""
config.py — Calibrated parameters for distinct scenario curves in CoalSafe-Sim v1.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_TABULAR = ROOT / "data" / "tabular"
DATA_THERMAL = ROOT / "data" / "thermal"
DATA_META    = ROOT / "data" / "metadata"
GRAPHS_DIR   = ROOT / "graphs"
DOCS_DIR     = ROOT / "documentation"

RANDOM_SEED = 42

DURATION_MIN    = 1440
TIME_STEP_MIN   = 1
NUM_TIME_POINTS = 1441
GRID_SIZE       = 50
NUM_DEPTH_LAYERS = 3

IMAGE_SAMPLING_INTERVAL = 10  # Sample thermal images & snapshot graphs every 10 min

DEFAULT_AMBIENT_TEMP   = 28.0
DEFAULT_HUMIDITY       = 55.0
DEFAULT_WIND_SPEED     = 2.5
DEFAULT_COAL_MOISTURE  = 12.0

ENV_DRIFT_AMBIENT  = 6.0   # Diurnal day/night temperature amplitude (±6°C)
ENV_DRIFT_HUMIDITY = 15.0  # Daily humidity swing
ENV_DRIFT_WIND     = 1.2   # Daily wind variation

# ── Oxidation & Heat Transfer Calibration (24-Hour Scale) ──────
OXIDATION_BASE_RATE       = 0.00075 # Base oxidation increment per minute
OXIDATION_TEMP_SCALE      = 0.024   # Arrhenius temperature sensitivity
OXIDATION_TEMP_REF        = 30.0    # Reference temperature (°C)
OXIDATION_O2_HALF         = 10.0    # O₂ half-saturation (%)
OXIDATION_MAX             = 1.0     # Max oxidation state

HEAT_COEFFICIENT          = 70.0    # Heat output per unit oxidation_rate
HEAT_TEMP_FEEDBACK        = 0.018   # Positive feedback multiplier

BASE_HEAT_LOSS            = 0.0010  # Conductive base loss
WIND_HEAT_LOSS_FACTOR     = 0.0003  # Convective loss per m/s wind
TEMP_DIFF_LOSS_FACTOR     = 0.0002  # Gradient loss to ambient
MOISTURE_LOSS_FACTOR      = 0.0001  # Moisture latent heat loss
MITIGATION_HEAT_REMOVAL   = 0.110   # Strong heat extraction rate for early mitigation

DEEP_HEAT_ABSORPTION      = 0.85    # Layer 0 deep core heat absorption
MIDDLE_RESPONSE_RATE      = 0.006   # Layer 1 middle layer response
NEAR_SURFACE_RESPONSE     = 0.004   # Layer 2 near-surface response
SURFACE_OBS_RESPONSE      = 0.003   # Surface sensor response

CO_BASELINE               = 2.0
CO2_BASELINE              = 400.0
O2_BASELINE               = 20.9
CO_RESPONSE_COEFF         = 600.0
CO2_RESPONSE_COEFF        = 1200.0
O2_CONSUMPTION_COEFF      = 1.5
GAS_LAG_FACTOR            = 0.03
O2_FLOOR                  = 5.0
O2_CEIL                   = 21.0

NOISE_AMBIENT_TEMP  = 0.25
NOISE_HUMIDITY      = 0.40
NOISE_WIND          = 0.08
NOISE_SURFACE_TEMP  = 0.20
NOISE_CO            = 1.2
NOISE_CO2           = 4.0
NOISE_O2            = 0.06
NOISE_INTERNAL_TEMP = 0.10
NOISE_PIXEL_TEMP    = 0.4

HOTSPOT_ANOMALY_MARGIN = 2.0

THERMAL_CMAP       = "inferno"
THERMAL_BLUR_SIGMA = 2.5
THERMAL_SAVE_EVERY = 10

RISK_WEIGHTS = {
    "oxidation":        0.15,
    "internal_thermal": 0.15,
    "heating_rate":     0.10,
    "CO_trend":         0.15,
    "CO2_trend":        0.10,
    "O2_depletion":     0.10,
    "surface_anomaly":  0.15,
    "hotspot_growth":   0.10,
}

RISK_NORM = {
    "oxidation_max":        1.0,
    "internal_temp_rise_max": 60.0,
    "heating_rate_max":     0.5,
    "CO_rise_max":          150.0,
    "CO2_rise_max":         300.0,
    "O2_drop_max":          6.0,
    "surface_anomaly_max":  15.0,
    "hotspot_area_max":     600.0,
}

RISK_BANDS = [
    (0,  19.99,  "NORMAL"),
    (20, 39.99,  "LOW"),
    (40, 59.99,  "MEDIUM"),
    (60, 79.99,  "HIGH"),
    (80, 100.0,  "CRITICAL"),
]

MISSING_CO_FRACTION        = 0.02
MISSING_TEMP_FRACTION      = 0.015
MISSING_THERMAL_FRACTION   = 0.02

HOTSPOT_INITIAL_RADIUS     = 3.0
HOTSPOT_GROWTH_RATE        = 0.08
HOTSPOT_MAX_RADIUS         = 12.0
