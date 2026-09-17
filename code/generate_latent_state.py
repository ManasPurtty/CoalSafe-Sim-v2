"""
generate_latent_state.py — Hidden sub-surface physics engine for CoalSafe-Sim v1.
"""

import numpy as np
import pandas as pd
import config as cfg
from generate_scenarios import get_scenarios, get_time_axis

def _temperature_factor(temp):
    """Arrhenius temperature sensitivity factor."""
    exponent = np.clip(cfg.OXIDATION_TEMP_SCALE * (temp - cfg.OXIDATION_TEMP_REF), -10, 10)
    return np.exp(exponent)

def _oxygen_factor(o2_local):
    """Michaelis-Menten oxygen availability factor."""
    return o2_local / (o2_local + cfg.OXIDATION_O2_HALF)

def simulate_latent(scenario, time_axis):
    """Simulates the 121-minute latent physical state for one scenario."""
    rng = np.random.default_rng(scenario["random_seed"])
    n = len(time_axis)

    s_factor   = scenario["scenario_factor"]
    mit_start  = scenario["mitigation_start_min"]
    ambient    = scenario["ambient_setting"]
    humidity   = scenario["humidity_setting"]
    wind       = scenario["wind_setting"]
    moisture   = scenario["coal_moisture_setting"]
    env_vary   = scenario.get("env_variation", False)
    sid        = scenario["scenario_id"]

    if env_vary:
        drift_phase = rng.uniform(0, 2 * np.pi)
        amb_drift   = 2.0 * np.sin(np.linspace(0, 4 * np.pi, n) + drift_phase)
        hum_drift   = 5.0 * np.cos(np.linspace(0, 3 * np.pi, n) + drift_phase)
        wind_drift  = 0.8 * np.sin(np.linspace(0, 5 * np.pi, n) + drift_phase)
    else:
        amb_drift  = np.zeros(n)
        hum_drift  = np.zeros(n)
        wind_drift = np.zeros(n)

    ox_state     = 0.001          
    deep_temp    = ambient + 0.5  
    middle_temp  = ambient + 0.2
    surface_temp = ambient + 0.05
    o2_local     = cfg.O2_BASELINE

    rows = []

    for i, t in enumerate(time_axis):
        # 24-hour sinusoidal day/night cycle (coolest at 06:00, warmest at 14:00)
        diurnal_cycle = cfg.ENV_DRIFT_AMBIENT * np.sin(2.0 * np.pi * (t - 360) / 1440.0)
        amb_t  = ambient + diurnal_cycle + amb_drift[i]
        hum_t  = np.clip(humidity - (cfg.ENV_DRIFT_HUMIDITY * np.sin(2.0 * np.pi * (t - 360) / 1440.0)) + hum_drift[i], 15, 95)
        wind_t = max(0.2, wind + (cfg.ENV_DRIFT_WIND * np.cos(2.0 * np.pi * (t - 180) / 1440.0)) + wind_drift[i])

        mitigating = (mit_start is not None) and (t >= mit_start)

        t_factor  = _temperature_factor(deep_temp)
        o2_factor = _oxygen_factor(o2_local)
        ox_rate   = cfg.OXIDATION_BASE_RATE * t_factor * o2_factor * s_factor

        if mitigating:
            if sid == "S07":
                ox_rate *= 0.12   # Late mitigation is less effective at suppressing deep core oxidation
                recovery = 0.00025
            else:
                ox_rate *= 0.05   # Early mitigation suppresses oxidation cleanly
                recovery = 0.0008
        else:
            recovery = 0.0

        ox_state = np.clip(ox_state + ox_rate - recovery, 0, cfg.OXIDATION_MAX)

        heat_gen = cfg.HEAT_COEFFICIENT * ox_rate * (1.0 + cfg.HEAT_TEMP_FEEDBACK * (deep_temp - amb_t))
        heat_gen = max(0.0, heat_gen)

        heat_loss = (cfg.BASE_HEAT_LOSS
                     + cfg.WIND_HEAT_LOSS_FACTOR * wind_t
                     + cfg.TEMP_DIFF_LOSS_FACTOR * max(0, deep_temp - amb_t)
                     + cfg.MOISTURE_LOSS_FACTOR * moisture)
        if mitigating:
            if sid == "S07":
                heat_loss += 0.050  # Moderate cooling for late mitigation
            else:
                heat_loss += cfg.MITIGATION_HEAT_REMOVAL  # Strong heat removal for early mitigation

        net_heat = heat_gen - heat_loss

        # Layer 0 Deep Core
        deep_delta = cfg.DEEP_HEAT_ABSORPTION * net_heat
        deep_delta = np.clip(deep_delta, -1.0, 0.45)
        deep_temp += deep_delta + rng.normal(0, cfg.NOISE_INTERNAL_TEMP * 0.2)
        deep_temp = min(max(amb_t - 2.0, deep_temp), 200.0)

        # Layer 1 Middle Core (delayed)
        middle_temp += cfg.MIDDLE_RESPONSE_RATE * (deep_temp - middle_temp)
        middle_temp += rng.normal(0, cfg.NOISE_INTERNAL_TEMP * 0.1)
        middle_temp = min(max(amb_t - 1.5, middle_temp), 200.0)

        # Layer 2 Near-Surface (further delayed)
        surface_temp += cfg.NEAR_SURFACE_RESPONSE * (middle_temp - surface_temp)
        surface_temp += rng.normal(0, cfg.NOISE_INTERNAL_TEMP * 0.1)
        surface_temp = min(max(amb_t - 1.0, surface_temp), 200.0)

        o2_consumed = cfg.O2_CONSUMPTION_COEFF * ox_rate
        o2_local = np.clip(o2_local - o2_consumed, cfg.O2_FLOOR, cfg.O2_CEIL)

        internal_temp = deep_temp

        rows.append({
            "scenario_id":              sid,
            "timestamp_min":            int(t),
            "oxidation_state":          round(ox_state, 6),
            "oxidation_rate":           round(ox_rate, 6),
            "internal_temperature":     round(internal_temp, 4),
            "heat_generation":          round(heat_gen, 6),
            "heat_loss":                round(heat_loss, 6),
            "deep_core_temperature":    round(deep_temp, 4),
            "middle_core_temperature":  round(middle_temp, 4),
            "near_surface_temperature": round(surface_temp, 4),
            "_ambient":                 round(amb_t, 4),
            "_humidity":                round(hum_t, 4),
            "_wind":                    round(wind_t, 4),
            "_coal_moisture":           round(moisture, 4),
            "_mitigating":              int(mitigating),
        })

    return rows

def generate_all_latent():
    time_axis = get_time_axis()
    all_rows  = []
    for sc in get_scenarios():
        print(f"  [LATENT] {sc['scenario_id']} — {sc['scenario_type']} ...")
        all_rows.extend(simulate_latent(sc, time_axis))
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    df = generate_all_latent()
    print(df.head())
