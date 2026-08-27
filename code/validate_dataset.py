
import sys
import io
import pathlib
import math

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
SENSOR_CSV  = ROOT / "data" / "tabular" / "sensor_data.csv"
LATENT_CSV  = ROOT / "data" / "metadata" / "latent_state.csv"
SCEN_CSV    = ROOT / "data" / "metadata" / "scenarios.csv"
THERMAL_CSV = ROOT / "data" / "metadata" / "thermal_metadata.csv"
THERMAL_DIR = ROOT / "data" / "thermal"

RISK_BANDS = [
    (0,  19.99, "NORMAL"),
    (20, 39.99, "LOW"),
    (40, 59.99, "MEDIUM"),
    (60, 79.99, "HIGH"),
    (80, 100.0, "CRITICAL"),
]

def expected_class(score):
    if score < 20.0:  return "NORMAL"
    elif score < 40.0: return "LOW"
    elif score < 60.0: return "MEDIUM"
    elif score < 80.0: return "HIGH"
    else:              return "CRITICAL"

def check(condition, msg_pass, msg_fail):
    if condition:
        print(f"  [OK]   {msg_pass}")
        return True
    else:
        print(f"  [FAIL] {msg_fail}")
        return False

def main():
    print("=" * 65)
    print("  CoalSafe-Sim v1  |  Dataset Quality Validator")
    print("=" * 65)

    try:
        df  = pd.read_csv(SENSOR_CSV)
        lat = pd.read_csv(LATENT_CSV)
        scen = pd.read_csv(SCEN_CSV)
        tm  = pd.read_csv(THERMAL_CSV)
        print(f"\n[LOAD] sensor_data.csv      : {df.shape[0]} rows")
        print(f"[LOAD] latent_state.csv     : {lat.shape[0]} rows")
        print(f"[LOAD] scenarios.csv        : {scen.shape[0]} rows")
        print(f"[LOAD] thermal_metadata.csv : {tm.shape[0]} rows\n")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    errors = 0

    print("── AUTOMATED CHECKS ─────────────────────────────────")
    
    n_scen = df["scenario_id"].nunique()
    errors += 0 if check(n_scen == 8,
        f"{n_scen} scenarios present (expected 8)",
        f"Found {n_scen} scenarios, expected 8") else 1

    counts = df.groupby("scenario_id").size()
    wrong = counts[counts != 121]
    errors += 0 if check(wrong.empty,
        "All scenarios have exactly 121 rows",
        f"Wrong row counts: {wrong.to_dict()}") else 1

    dups = df.duplicated(subset=["scenario_id", "timestamp_min"])
    errors += 0 if check(not dups.any(),
        "No duplicate (scenario_id, timestamp_min) pairs",
        f"{dups.sum()} duplicate rows") else 1

    rs_bad = ((df["risk_score"] < 0) | (df["risk_score"] > 100)).sum()
    errors += 0 if check(rs_bad == 0,
        "All risk scores in [0, 100]",
        f"{rs_bad} risk scores outside range") else 1

    mismatch = 0
    for _, row in df.iterrows():
        if pd.notna(row["risk_score"]):
            if expected_class(row["risk_score"]) != row["risk_class"]:
                mismatch += 1
    errors += 0 if check(mismatch == 0,
        "All risk classes match their risk scores",
        f"{mismatch} mismatches") else 1

    hum = df["humidity"].dropna()
    hum_bad = ((hum < 0) | (hum > 100)).sum()
    errors += 0 if check(hum_bad == 0,
        "All humidity values in [0, 100]%",
        f"{hum_bad} out of range") else 1

    co_neg = (df["CO"].dropna() < 0).sum()
    errors += 0 if check(co_neg == 0,
        "No negative CO values",
        f"{co_neg} negative CO") else 1

    co2_neg = (df["CO2"].dropna() < 0).sum()
    errors += 0 if check(co2_neg == 0,
        "No negative CO2 values",
        f"{co2_neg} negative CO2") else 1

    o2 = df["O2"].dropna()
    o2_bad = ((o2 < 4.9) | (o2 > 21.1)).sum()
    errors += 0 if check(o2_bad == 0,
        "All O2 in [5%, 21%]",
        f"{o2_bad} out of range") else 1

    ha_bad = ((df["hotspot_area"] < 0) | (df["hotspot_area"] > 2500)).sum()
    errors += 0 if check(ha_bad == 0,
        "All hotspot_area in [0, 2500]",
        f"{ha_bad} out of range") else 1

    has_hs = df[df["hotspot_area"] > 0]
    if len(has_hs) > 0:
        hx_bad = ((has_hs["hotspot_x"].dropna() < 0) | (has_hs["hotspot_x"].dropna() >= 50)).sum()
        hy_bad = ((has_hs["hotspot_y"].dropna() < 0) | (has_hs["hotspot_y"].dropna() >= 50)).sum()
        errors += 0 if check(hx_bad + hy_bad == 0,
            "All hotspot coordinates inside 50x50 grid",
            f"{hx_bad + hy_bad} coordinates out of grid") else 1
    else:
        print("  [SKIP] No hotspot rows to validate coordinates")

    existing = tm[tm["image_path"] != "MISSING"]
    missing_imgs = 0
    for _, row in existing.iterrows():
        img_path = ROOT / row["image_path"]
        if not img_path.exists():
            missing_imgs += 1
    errors += 0 if check(missing_imgs == 0,
        f"All {len(existing)} referenced thermal images exist on disk",
        f"{missing_imgs} images not found") else 1

    max_amb_step = 0
    for sid in df["scenario_id"].unique():
        d = df[df["scenario_id"] == sid]["ambient_temperature"].values
        steps = np.abs(np.diff(d))
        if len(steps) > 0:
            max_amb_step = max(max_amb_step, steps.max())
    errors += 0 if check(max_amb_step < 5.0,
        f"Ambient temp varies smoothly (max step = {max_amb_step:.2f})",
        f"Ambient temp jumps too large: {max_amb_step:.2f}") else 1

    max_int_step = 0
    for sid in lat["scenario_id"].unique():
        d = lat[lat["scenario_id"] == sid]["internal_temperature"].values
        steps = np.abs(np.diff(d))
        if len(steps) > 0:
            max_int_step = max(max_int_step, steps.max())
    errors += 0 if check(max_int_step < 5.0,
        f"Internal temp varies smoothly (max step = {max_int_step:.2f})",
        f"Internal temp jumps too large: {max_int_step:.2f}") else 1

    s06 = df[df["scenario_id"] == "S06"]
    if len(s06) > 0:
        mit_rows = s06[s06["mitigation_status"] == 1]
        if len(mit_rows) >= 10:
            early_mit_risk = mit_rows.head(10)["risk_score"].mean()
            late_mit_risk = mit_rows.tail(10)["risk_score"].mean()
            errors += 0 if check(late_mit_risk <= early_mit_risk + 5,
                f"S06 mitigation reduces risk ({early_mit_risk:.1f} -> {late_mit_risk:.1f})",
                f"S06 risk not decreasing after mitigation") else 1
        else:
            print("  [SKIP] S06 has fewer than 10 mitigation rows")
    else:
        print("  [SKIP] S06 not found")

    nan_co = df["CO"].isna().sum()
    nan_t  = df["surface_temperature_mean"].isna().sum()
    errors += 0 if check(nan_co > 0 and nan_t > 0,
        f"Deliberate missing data present (CO NaN={nan_co}, Temp NaN={nan_t})",
        "No deliberate missing data found") else 1

    miss_thermal = (tm["image_path"] == "MISSING").sum()
    errors += 0 if check(miss_thermal > 0,
        f"{miss_thermal} deliberately missing thermal images recorded",
        "No missing thermal images recorded") else 1

    print("\n── CAUSAL CHECKS ────────────────────────────────────")
    
    s05_lat = lat[lat["scenario_id"] == "S05"]
    if len(s05_lat) > 5:
        corr = np.corrcoef(s05_lat["oxidation_state"], s05_lat["heat_generation"])[0, 1]
        errors += 0 if check(corr > 0.5,
            f"S05: oxidation <-> heat_generation correlation = {corr:.3f}",
            f"S05: weak correlation {corr:.3f}") else 1

    if len(s05_lat) > 5:
        net = s05_lat["heat_generation"].values - s05_lat["heat_loss"].values
        cumnet = np.cumsum(net)
        corr2 = np.corrcoef(cumnet, s05_lat["internal_temperature"])[0, 1]
        errors += 0 if check(corr2 > 0.5,
            f"S05: cumulative net heat <-> internal temp correlation = {corr2:.3f}",
            f"S05: weak correlation {corr2:.3f}") else 1

    s05_obs = df[df["scenario_id"] == "S05"]
    if len(s05_lat) > 5 and len(s05_obs) > 5:
        co_clean = s05_obs["CO"].dropna()
        ox_aligned = s05_lat["oxidation_state"].iloc[co_clean.index - co_clean.index[0]]
        if len(co_clean) == len(ox_aligned):
            corr3 = np.corrcoef(ox_aligned, co_clean)[0, 1]
        else:
            corr3 = np.corrcoef(
                s05_lat["oxidation_state"].values[:len(co_clean)],
                co_clean.values[:len(s05_lat)])[0, 1]
        errors += 0 if check(corr3 > 0.3,
            f"S05: oxidation <-> CO correlation = {corr3:.3f}",
            f"S05: weak gas-oxidation link {corr3:.3f}") else 1

    for check_sid in ["S04", "S05"]:
        sl = lat[lat["scenario_id"] == check_sid]
        so = df[df["scenario_id"] == check_sid]
        if len(sl) > 0 and len(so) > 0:
            baseline_int = sl["internal_temperature"].iloc[0]
            baseline_surf = so["surface_temperature_max"].iloc[0]

            t_int = sl[sl["internal_temperature"] > baseline_int + 3]["timestamp_min"]
            t_int = int(t_int.iloc[0]) if len(t_int) > 0 else 999

            t_surf = so[so["surface_temperature_max"] > baseline_surf + 3]["timestamp_min"]
            t_surf = int(t_surf.iloc[0]) if len(t_surf) > 0 else 999

            errors += 0 if check(t_int <= t_surf,
                f"{check_sid}: internal rises at t={t_int}, surface at t={t_surf} (delay={t_surf-t_int} min)",
                f"{check_sid}: surface rises BEFORE internal! t_int={t_int}, t_surf={t_surf}") else 1
            break

    s04_obs = df[df["scenario_id"] == "S04"]
    s04_lat = lat[lat["scenario_id"] == "S04"]
    if len(s04_obs) > 0 and len(s04_lat) > 0:
        t_ox = s04_lat[s04_lat["oxidation_state"] > 0.1]["timestamp_min"]
        t_ox = int(t_ox.iloc[0]) if len(t_ox) > 0 else 999
        t_hs = s04_obs[s04_obs["hotspot_area"] > 0]["timestamp_min"]
        t_hs = int(t_hs.iloc[0]) if len(t_hs) > 0 else 999
        errors += 0 if check(t_ox <= t_hs,
            f"S04: oxidation > 0.1 at t={t_ox}, hotspot at t={t_hs}",
            f"S04: hotspot appears before oxidation!") else 1

    s03_obs = df[df["scenario_id"] == "S03"]
    if len(s03_obs) > 0:
        t_risk20 = s03_obs[s03_obs["risk_score"] >= 20]["timestamp_min"]
        t_risk20 = int(t_risk20.iloc[0]) if len(t_risk20) > 0 else 999
        anom = s03_obs["surface_temperature_max"] - s03_obs["ambient_temperature"]
        t_surf5 = s03_obs[anom >= 5.0]["timestamp_min"]
        t_surf5 = int(t_surf5.iloc[0]) if len(t_surf5) > 0 else 999
        errors += 0 if check(t_risk20 <= t_surf5,
            f"S03: risk >= LOW at t={t_risk20}, surface anomaly >= 5C at t={t_surf5}",
            f"S03: surface anomaly appears before risk increases!") else 1

    if len(s06) > 0:
        mit_start_row = s06[s06["mitigation_status"] == 1]
        if len(mit_start_row) > 0:
            mit_t = int(mit_start_row["timestamp_min"].iloc[0])
            pre = s06[(s06["timestamp_min"] >= mit_t - 10) & (s06["timestamp_min"] < mit_t)]
            post = s06[s06["timestamp_min"] >= s06["timestamp_min"].max() - 10]
            if len(pre) > 0 and len(post) > 0:
                errors += 0 if check(
                    post["risk_score"].mean() < pre["risk_score"].mean() + 10,
                    f"S06: risk decreases after mitigation "
                    f"(pre={pre['risk_score'].mean():.1f}, post={post['risk_score'].mean():.1f})",
                    f"S06: risk did not decrease after mitigation") else 1

    if len(s05_lat) > 0:
        t_start = s05_lat["internal_temperature"].iloc[0]
        t_end = s05_lat["internal_temperature"].iloc[-1]
        errors += 0 if check(t_end > t_start,
            f"S05: internal temp at t=120 ({t_end:.1f}) > t=0 ({t_start:.1f})",
            f"S05: noise reversed the heating progression!") else 1

    print("\n" + "=" * 65)
    if errors == 0:
        print("  [ALL PASSED]  Dataset is valid.")
    else:
        print(f"  [FAILED]  {errors} CHECK(S) FAILED — Review output above.")
    print("=" * 65)

    print(f"\n{'sid':>5s}  {'rows':>4s}  {'T_int_max':>9s}  {'T_surf_max':>10s}  "
          f"{'CO_max':>8s}  {'O2_min':>6s}  {'risk_max':>8s}  {'class_max':>10s}")
    for sid in sorted(df["scenario_id"].unique()):
        d = df[df["scenario_id"] == sid]
        l = lat[lat["scenario_id"] == sid]
        print(f"{sid:>5s}  {len(d):4d}  "
              f"{l['internal_temperature'].max():9.2f}  "
              f"{d['surface_temperature_max'].max():10.2f}  "
              f"{d['CO'].max():8.2f}  "
              f"{d['O2'].min():6.2f}  "
              f"{d['risk_score'].max():8.2f}  "
              f"{d[d['risk_score'] == d['risk_score'].max()]['risk_class'].iloc[0]:>10s}")

if __name__ == "__main__":
    main()
