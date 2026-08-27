
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_TABULAR = ROOT / "data" / "tabular"
DATA_THERMAL = ROOT / "data" / "thermal"
DATA_META    = ROOT / "data" / "metadata"
GRAPHS_DIR   = ROOT / "graphs"
DOCS_DIR     = ROOT / "documentation"

RANDOM_SEED = 42  

DURATION_MIN    = 120   
TIME_STEP_MIN   = 1     
NUM_TIME_POINTS = 121   
GRID_SIZE       = 50    
NUM_DEPTH_LAYERS = 3    

DEFAULT_AMBIENT_TEMP   = 30.0   
DEFAULT_HUMIDITY       = 55.0   
DEFAULT_WIND_SPEED     = 2.5    
DEFAULT_COAL_MOISTURE  = 12.0   

ENV_DRIFT_AMBIENT  = 1.5   
ENV_DRIFT_HUMIDITY = 3.0   
ENV_DRIFT_WIND     = 0.5   

OXIDATION_BASE_RATE       = 0.004   
OXIDATION_TEMP_SCALE      = 0.12    
OXIDATION_TEMP_REF        = 30.0    
OXIDATION_O2_HALF         = 10.0    
OXIDATION_MAX             = 1.0     

HEAT_COEFFICIENT          = 60.0    
HEAT_TEMP_FEEDBACK        = 0.15    

BASE_HEAT_LOSS            = 0.005   
WIND_HEAT_LOSS_FACTOR     = 0.002   
TEMP_DIFF_LOSS_FACTOR     = 0.001   
MOISTURE_LOSS_FACTOR      = 0.001   
MITIGATION_HEAT_REMOVAL   = 6.0     

DEEP_HEAT_ABSORPTION      = 0.90    
MIDDLE_RESPONSE_RATE      = 0.06    
NEAR_SURFACE_RESPONSE     = 0.035   
SURFACE_OBS_RESPONSE      = 0.025   

CO_BASELINE               = 2.0     
CO2_BASELINE              = 400.0   
O2_BASELINE               = 20.9    
CO_RESPONSE_COEFF         = 4000.0  
CO2_RESPONSE_COEFF        = 8000.0  
O2_CONSUMPTION_COEFF      = 60.0    
GAS_LAG_FACTOR            = 0.25    
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
THERMAL_SAVE_EVERY = 1   

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
    "internal_temp_rise_max": 30.0,   
    "heating_rate_max":     0.5,      
    "CO_rise_max":          200.0,    
    "CO2_rise_max":         400.0,    
    "O2_drop_max":          10.0,     
    "surface_anomaly_max":  15.0,     
    "hotspot_area_max":     800.0,    
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
