"""
generate_scenarios.py — Defines the eight required scenarios.
"""

import numpy as np
import config as cfg

def get_time_axis():
    return np.arange(0, cfg.NUM_TIME_POINTS, cfg.TIME_STEP_MIN)


SCENARIOS = [
    {
        "scenario_id":        "S01",
        "scenario_type":      "STABLE",
        "random_seed":        cfg.RANDOM_SEED + 0,
        "duration_min":       cfg.DURATION_MIN,
        "time_step_min":      cfg.TIME_STEP_MIN,
        "core_x":             25,
        "core_y":             25,
        "core_depth":         0,
        "heating_pattern":    "none",
        "scenario_factor":    0.05,       # Baseline normal
        "mitigation_start_min": None,
        "ambient_setting":    28.0,
        "humidity_setting":   55.0,
        "wind_setting":       2.5,
        "coal_moisture_setting": 12.0,
        "env_variation":      False,
        "notes": "Stable baseline. Normal 24h diurnal cycle without self-heating.",
    },
    {
        "scenario_id":        "S02",
        "scenario_type":      "EARLY_OXIDATION",
        "random_seed":        cfg.RANDOM_SEED + 1,
        "duration_min":       cfg.DURATION_MIN,
        "time_step_min":      cfg.TIME_STEP_MIN,
        "core_x":             24,
        "core_y":             26,
        "core_depth":         0,
        "heating_pattern":    "slow",
        "scenario_factor":    0.35,      # Incubation phase: hidden sub-surface oxidation
        "mitigation_start_min": None,
        "ambient_setting":    28.0,
        "humidity_setting":   52.0,
        "wind_setting":       2.0,
        "coal_moisture_setting": 10.0,
        "env_variation":      False,
        "notes": "Hidden oxidation rises over 24h while surface remains almost normal.",
    },
    {
        "scenario_id":        "S03",
        "scenario_type":      "DEVELOPING_CORE",
        "random_seed":        cfg.RANDOM_SEED + 2,
        "duration_min":       cfg.DURATION_MIN,
        "time_step_min":      cfg.TIME_STEP_MIN,
        "core_x":             26,
        "core_y":             24,
        "core_depth":         0,
        "heating_pattern":    "moderate",
        "scenario_factor":    0.55,      # Sub-surface thermal core grows
        "mitigation_start_min": None,
        "ambient_setting":    29.0,
        "humidity_setting":   48.0,
        "wind_setting":       1.8,
        "coal_moisture_setting": 9.0,
        "env_variation":      False,
        "notes": "Sub-surface thermal core grows over 24h; gas and internal signals change.",
    },
    {
        "scenario_id":        "S04",
        "scenario_type":      "HEAT_MIGRATION",
        "random_seed":        cfg.RANDOM_SEED + 3,
        "duration_min":       cfg.DURATION_MIN,
        "time_step_min":      cfg.TIME_STEP_MIN,
        "core_x":             25,
        "core_y":             25,
        "core_depth":         0,
        "heating_pattern":    "strong",
        "scenario_factor":    0.80,      # Heat reaches surface
        "mitigation_start_min": None,
        "ambient_setting":    30.0,
        "humidity_setting":   45.0,
        "wind_setting":       1.5,
        "coal_moisture_setting": 8.0,
        "env_variation":      False,
        "notes": "Heat reaches surface in afternoon/evening; thermal anomaly expands.",
    },
    {
        "scenario_id":        "S05",
        "scenario_type":      "HIGH_RISK",
        "random_seed":        cfg.RANDOM_SEED + 4,
        "duration_min":       cfg.DURATION_MIN,
        "time_step_min":      cfg.TIME_STEP_MIN,
        "core_x":             23,
        "core_y":             27,
        "core_depth":         0,
        "heating_pattern":    "rapid",
        "scenario_factor":    1.15,      # Rapid runaway oxidation
        "mitigation_start_min": None,
        "ambient_setting":    32.0,
        "humidity_setting":   38.0,
        "wind_setting":       1.0,
        "coal_moisture_setting": 6.0,
        "env_variation":      False,
        "notes": "Thermal core and all observable signals rapidly accelerate to runaway ignition.",
    },
    {
        "scenario_id":        "S06",
        "scenario_type":      "EARLY_MITIGATION",
        "random_seed":        cfg.RANDOM_SEED + 5,
        "duration_min":       cfg.DURATION_MIN,
        "time_step_min":      cfg.TIME_STEP_MIN,
        "core_x":             25,
        "core_y":             25,
        "core_depth":         0,
        "heating_pattern":    "moderate",
        "scenario_factor":    0.85,      
        "mitigation_start_min": 360,     # Early intervention at Hour 6 (t=360 min)
        "ambient_setting":    28.0,
        "humidity_setting":   50.0,
        "wind_setting":       2.0,
        "coal_moisture_setting": 10.0,
        "env_variation":      False,
        "notes": "Intervention occurs early at Hour 6 while core is still moderate (~45°C).",
    },
    {
        "scenario_id":        "S07",
        "scenario_type":      "LATE_MITIGATION",
        "random_seed":        cfg.RANDOM_SEED + 6,
        "duration_min":       cfg.DURATION_MIN,
        "time_step_min":      cfg.TIME_STEP_MIN,
        "core_x":             25,
        "core_y":             25,
        "core_depth":         0,
        "heating_pattern":    "moderate",
        "scenario_factor":    1.04,      
        "mitigation_start_min": 840,     # Late intervention at Hour 14 (t=840 min)
        "ambient_setting":    28.0,
        "humidity_setting":   50.0,
        "wind_setting":       2.0,
        "coal_moisture_setting": 10.0,
        "env_variation":      False,
        "notes": "Intervention occurs late at Hour 14 after severe core heating (>115°C).",
    },
    {
        "scenario_id":        "S08",
        "scenario_type":      "ENVIRONMENTAL_VARIATION",
        "random_seed":        cfg.RANDOM_SEED + 7,
        "duration_min":       cfg.DURATION_MIN,
        "time_step_min":      cfg.TIME_STEP_MIN,
        "core_x":             25,
        "core_y":             24,
        "core_depth":         0,
        "heating_pattern":    "moderate",
        "scenario_factor":    0.20,      
        "mitigation_start_min": None,
        "ambient_setting":    28.0,      
        "humidity_setting":   60.0,
        "wind_setting":       3.5,       # Windier
        "coal_moisture_setting": 14.0,   # Wet coal
        "env_variation":      True,      # Full 24h diurnal weather swings
        "notes": "24-hour diurnal weather variations alter the heat balance.",
    },
]

def get_scenarios():
    return SCENARIOS
