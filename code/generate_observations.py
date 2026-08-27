
import math
import numpy as np
import pandas as pd
import config as cfg
from generate_scenarios import get_scenarios

def _risk_class(score):

    for lo, hi, label in cfg.RISK_BANDS:
        if lo <= score <= hi:
            return label
    return "CRITICAL"

def _sev(value, max_val):
    return min(max(value, 0.0) / max(max_val, 1e-9), 1.0)

def generate_observations(latent_df):

    scenarios = get_scenarios()
    sc_lookup = {s["scenario_id"]: s for s in scenarios}

    obs_rows = []

    for sid in latent_df["scenario_id"].unique():
        sc   = sc_lookup[sid]
        rows = latent_df[latent_df["scenario_id"] == sid].reset_index(drop=True)
        rng  = np.random.default_rng(sc["random_seed"] + 100)

        n = len(rows)

        surf_obs = rows["_ambient"].iloc[0]
        surf_obs_series = []

        co_accum  = cfg.CO_BASELINE
        co2_accum = cfg.CO2_BASELINE
        o2_obs    = cfg.O2_BASELINE

        prev_surf = surf_obs

        for idx in range(n):
            r = rows.iloc[idx]

            near_surf = r["near_surface_temperature"]
            surf_obs += cfg.SURFACE_OBS_RESPONSE * (near_surf - surf_obs)
            surf_obs += rng.normal(0, cfg.NOISE_SURFACE_TEMP)
            surf_obs_series.append(surf_obs)

            ox_rate = r["oxidation_rate"]
            int_temp = r["internal_temperature"]
            temp_above = max(0, int_temp - cfg.OXIDATION_TEMP_REF)

            co_target  = cfg.CO_BASELINE + cfg.CO_RESPONSE_COEFF * ox_rate * (1 + 0.02 * temp_above)
            co2_target = cfg.CO2_BASELINE + cfg.CO2_RESPONSE_COEFF * ox_rate * (1 + 0.015 * temp_above)
            o2_target  = cfg.O2_BASELINE - cfg.O2_CONSUMPTION_COEFF * ox_rate

            lag = cfg.GAS_LAG_FACTOR
            co_accum  = co_accum  + (1 - lag) * (co_target - co_accum)
            co2_accum = co2_accum + (1 - lag) * (co2_target - co2_accum)
            o2_obs    = o2_obs    + (1 - lag) * (o2_target - o2_obs)

            co_obs  = max(0.0, co_accum + rng.normal(0, cfg.NOISE_CO))
            co2_obs = max(0.0, co2_accum + rng.normal(0, cfg.NOISE_CO2))
            o2_obs_noisy = np.clip(o2_obs + rng.normal(0, cfg.NOISE_O2),
                                   cfg.O2_FLOOR, cfg.O2_CEIL)

            amb_obs  = r["_ambient"]  + rng.normal(0, cfg.NOISE_AMBIENT_TEMP)
            hum_obs  = np.clip(r["_humidity"] + rng.normal(0, cfg.NOISE_HUMIDITY), 0, 100)
            wind_obs = max(0.0, r["_wind"] + rng.normal(0, cfg.NOISE_WIND))

            heating_rate = surf_obs - prev_surf if idx > 0 else 0.0
            prev_surf = surf_obs

            anomaly = near_surf - r["_ambient"]
            if anomaly > cfg.HOTSPOT_ANOMALY_MARGIN:
                
                sigma = min(cfg.HOTSPOT_INITIAL_RADIUS
                            + cfg.HOTSPOT_GROWTH_RATE * r["oxidation_state"] * 100,
                            cfg.HOTSPOT_MAX_RADIUS)
                area = int(np.clip(np.pi * sigma**2 * (anomaly / 5.0), 1, 2500))
                hx = sc["core_x"] + rng.normal(0, 0.5)
                hy = sc["core_y"] + rng.normal(0, 0.5)
            else:
                area = 0
                hx = float("nan")
                hy = float("nan")

            surf_mean = surf_obs
            surf_max  = surf_obs + max(0, anomaly * 0.6)

            ox_sev     = _sev(r["oxidation_state"],
                              cfg.RISK_NORM["oxidation_max"])
            it_sev     = _sev(r["internal_temperature"] - r["_ambient"],
                              cfg.RISK_NORM["internal_temp_rise_max"])
            hr_sev     = _sev(abs(heating_rate),
                              cfg.RISK_NORM["heating_rate_max"])
            co_sev     = _sev(co_obs - cfg.CO_BASELINE,
                              cfg.RISK_NORM["CO_rise_max"])
            co2_sev    = _sev(co2_obs - cfg.CO2_BASELINE,
                              cfg.RISK_NORM["CO2_rise_max"])
            o2_sev     = _sev(cfg.O2_BASELINE - o2_obs_noisy,
                              cfg.RISK_NORM["O2_drop_max"])
            surf_sev   = _sev(surf_max - amb_obs,
                              cfg.RISK_NORM["surface_anomaly_max"])
            hs_sev     = _sev(area,
                              cfg.RISK_NORM["hotspot_area_max"])

            w = cfg.RISK_WEIGHTS
            risk_raw = (w["oxidation"]        * ox_sev
                      + w["internal_thermal"] * it_sev
                      + w["heating_rate"]     * hr_sev
                      + w["CO_trend"]         * co_sev
                      + w["CO2_trend"]        * co2_sev
                      + w["O2_depletion"]     * o2_sev
                      + w["surface_anomaly"]  * surf_sev
                      + w["hotspot_growth"]   * hs_sev)

            risk_score = np.clip(risk_raw * 100, 0, 100)
            risk_label = _risk_class(risk_score)

            obs_rows.append({
                "scenario_id":              sid,
                "timestamp_min":            int(r["timestamp_min"]),
                "ambient_temperature":      round(amb_obs, 2),
                "humidity":                 round(hum_obs, 2),
                "wind_speed":               round(wind_obs, 2),
                "coal_moisture":            round(r["_coal_moisture"], 2),
                "CO":                       round(co_obs, 2),
                "CO2":                      round(co2_obs, 2),
                "O2":                       round(o2_obs_noisy, 2),
                "surface_temperature_mean": round(surf_mean, 2),
                "surface_temperature_max":  round(surf_max, 2),
                "hotspot_area":             area,
                "hotspot_x":                round(hx, 2) if not math.isnan(hx) else float("nan"),
                "hotspot_y":                round(hy, 2) if not math.isnan(hy) else float("nan"),
                "heating_rate":             round(heating_rate, 4),
                "risk_score":               round(risk_score, 2),
                "risk_class":               risk_label,
                "mitigation_status":        int(r["_mitigating"]),
            })

        print(f"  [OBS]    {sid} — {n} observation rows generated")

    obs_df = pd.DataFrame(obs_rows)

    rng_miss = np.random.default_rng(cfg.RANDOM_SEED + 999)
    n_total = len(obs_df)

    n_miss_co = max(1, int(n_total * cfg.MISSING_CO_FRACTION))
    miss_idx_co = rng_miss.choice(n_total, size=n_miss_co, replace=False)
    obs_df.loc[miss_idx_co, "CO"] = float("nan")

    n_miss_t = max(1, int(n_total * cfg.MISSING_TEMP_FRACTION))
    miss_idx_t = rng_miss.choice(n_total, size=n_miss_t, replace=False)
    obs_df.loc[miss_idx_t, "surface_temperature_mean"] = float("nan")
    obs_df.loc[miss_idx_t, "surface_temperature_max"]  = float("nan")

    print(f"  [MISS]   Injected {n_miss_co} missing CO + {n_miss_t} missing temp readings")

    drop_cols = [c for c in latent_df.columns if c.startswith("_")]
    latent_clean = latent_df.drop(columns=drop_cols)

    return obs_df, latent_clean

if __name__ == "__main__":
    from generate_latent_state import generate_all_latent
    lat = generate_all_latent()
    obs, lat_clean = generate_observations(lat)
    print(obs.head())
    print(f"\nObs rows: {len(obs)}, Latent rows: {len(lat_clean)}")
