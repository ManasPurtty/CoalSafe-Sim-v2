
import sys
import io
import pathlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CODE_DIR = pathlib.Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from generate_scenarios import get_scenarios, get_time_axis
from generate_latent_state import generate_all_latent
from generate_observations import generate_observations
from generate_thermal_images import generate_thermal_images
from generate_per_minute_graphs import generate_per_minute_graphs

SCENARIO_DISPLAY_NAMES = {
    "S01": "Stable Baseline",
    "S02": "Early Oxidation",
    "S03": "Developing Core",
    "S04": "Heat Migration",
    "S05": "High Risk",
    "S06": "Early Mitigation (t=45m)",
    "S07": "Late Mitigation (t=85m)",
    "S08": "Environmental Variation",
}

SCENARIO_COLORS = {
    "S01": "#004578",  # Navy Blue
    "S02": "#d97706",  # Amber Yellow
    "S03": "#0891b2",  # Teal Cyan
    "S04": "#ea580c",  # Vibrant Orange
    "S05": "#dc2626",  # Crimson Red
    "S06": "#16a34a",  # Forest Green
    "S07": "#9333ea",  # Deep Purple
    "S08": "#64748b",  # Slate Gray
}

SCENARIO_STYLES = {
    "S01": {"ls": "-",  "lw": 2.0, "marker": None},
    "S02": {"ls": "--", "lw": 2.0, "marker": None},
    "S03": {"ls": "-",  "lw": 2.2, "marker": None},
    "S04": {"ls": "-.", "lw": 2.2, "marker": None},
    "S05": {"ls": "-",  "lw": 2.5, "marker": None},
    "S06": {"ls": "-",  "lw": 2.5, "marker": "o"},
    "S07": {"ls": "--", "lw": 2.5, "marker": "^"},
    "S08": {"ls": ":",  "lw": 2.0, "marker": None},
}

def generate_plots(obs_df, latent_df, thermal_meta):

    gdir = cfg.GRAPHS_DIR
    gdir.mkdir(parents=True, exist_ok=True)
    scenarios = get_scenarios()
    sids = [s["scenario_id"] for s in scenarios]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for sid in sids:
        d = latent_df[latent_df["scenario_id"] == sid]
        st = SCENARIO_STYLES.get(sid, {"ls": "-", "lw": 1.5, "marker": None})
        ax.plot(
            d["timestamp_min"],
            d["internal_temperature"],
            label=SCENARIO_DISPLAY_NAMES.get(sid, sid),
            color=SCENARIO_COLORS.get(sid, None),
            linestyle=st["ls"],
            linewidth=st["lw"],
            marker=st["marker"],
            markevery=12 if st["marker"] else None,
            markersize=5
        )
    ax.axvline(45, color="#16a34a", linestyle=":", alpha=0.4, label="_nolegend_")
    ax.axvline(85, color="#9333ea", linestyle=":", alpha=0.4, label="_nolegend_")
    ax.set_xlabel("Time (min)", fontweight="bold")
    ax.set_ylabel("Internal Temperature (°C)", fontweight="bold")
    ax.set_title("Internal Temperature vs Time", fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(gdir / "01_internal_temp_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for sid in sids:
        d = obs_df[obs_df["scenario_id"] == sid]
        st = SCENARIO_STYLES.get(sid, {"ls": "-", "lw": 1.5, "marker": None})
        ax.plot(
            d["timestamp_min"],
            d["surface_temperature_max"],
            label=SCENARIO_DISPLAY_NAMES.get(sid, sid),
            color=SCENARIO_COLORS.get(sid, None),
            linestyle=st["ls"],
            linewidth=st["lw"],
            marker=st["marker"],
            markevery=12 if st["marker"] else None,
            markersize=5
        )
    ax.set_xlabel("Time (min)", fontweight="bold")
    ax.set_ylabel("Surface Temp Max (°C)", fontweight="bold")
    ax.set_title("Surface Temperature (Max) vs Time", fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(gdir / "02_surface_temp_vs_time.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for sid in sids:
        d = obs_df[obs_df["scenario_id"] == sid]
        st = SCENARIO_STYLES.get(sid, {"ls": "-", "lw": 1.5, "marker": None})
        axes[0].plot(d["timestamp_min"], d["CO"], label=SCENARIO_DISPLAY_NAMES.get(sid, sid), color=SCENARIO_COLORS.get(sid), linestyle=st["ls"], lw=st["lw"])
        axes[1].plot(d["timestamp_min"], d["CO2"], label=SCENARIO_DISPLAY_NAMES.get(sid, sid), color=SCENARIO_COLORS.get(sid), linestyle=st["ls"], lw=st["lw"])
        axes[2].plot(d["timestamp_min"], d["O2"], label=SCENARIO_DISPLAY_NAMES.get(sid, sid), color=SCENARIO_COLORS.get(sid), linestyle=st["ls"], lw=st["lw"])
    axes[0].set_ylabel("CO (ppm)", fontweight="bold")
    axes[0].set_title("Gas Concentrations vs Time", fontweight="bold")
    axes[1].set_ylabel("CO2 (ppm)", fontweight="bold")
    axes[2].set_ylabel("O2 (%)", fontweight="bold")
    axes[2].set_xlabel("Time (min)", fontweight="bold")
    for ax in axes:
        ax.legend(fontsize=7, ncol=4, loc="upper left", framealpha=0.95)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(gdir / "03_gases_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for sid in sids:
        d = obs_df[obs_df["scenario_id"] == sid]
        st = SCENARIO_STYLES.get(sid, {"ls": "-", "lw": 1.5, "marker": None})
        ax.plot(
            d["timestamp_min"],
            d["risk_score"],
            label=SCENARIO_DISPLAY_NAMES.get(sid, sid),
            color=SCENARIO_COLORS.get(sid, None),
            linestyle=st["ls"],
            linewidth=st["lw"]
        )
    ax.set_xlabel("Time (min)", fontweight="bold")
    ax.set_ylabel("Risk Score (0-100)", fontweight="bold")
    ax.set_title("Synthetic Risk Score vs Time", fontweight="bold")
    ax.axhline(20, color="green", ls="--", alpha=0.4, label="LOW threshold")
    ax.axhline(40, color="orange", ls="--", alpha=0.4, label="MEDIUM threshold")
    ax.axhline(60, color="red", ls="--", alpha=0.4, label="HIGH threshold")
    ax.axhline(80, color="darkred", ls="--", alpha=0.4, label="CRITICAL threshold")
    ax.legend(fontsize=7, ncol=5, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(gdir / "04_risk_score_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for sid in sids:
        d = obs_df[obs_df["scenario_id"] == sid]
        st = SCENARIO_STYLES.get(sid, {"ls": "-", "lw": 1.5, "marker": None})
        ax.plot(
            d["timestamp_min"],
            d["hotspot_area"],
            label=SCENARIO_DISPLAY_NAMES.get(sid, sid),
            color=SCENARIO_COLORS.get(sid, None),
            linestyle=st["ls"],
            linewidth=st["lw"]
        )
    ax.set_xlabel("Time (min)", fontweight="bold")
    ax.set_ylabel("Hotspot Area (cells)", fontweight="bold")
    ax.set_title("Hotspot Area vs Time", fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(gdir / "05_hotspot_area_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for sid in sids:
        dl = latent_df[latent_df["scenario_id"] == sid]
        do = obs_df[obs_df["scenario_id"] == sid]
        ax.scatter(dl["internal_temperature"].values,
                   do["surface_temperature_max"].values,
                   s=5, alpha=0.5, label=SCENARIO_DISPLAY_NAMES.get(sid, sid))
    ax.set_xlabel("Internal Temperature (°C)")
    ax.set_ylabel("Surface Temperature Max (°C)")
    ax.set_title("Internal vs Surface Temperature")
    ax.plot([25, 200], [25, 200], "k--", alpha=0.3, label="1:1 line")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(gdir / "06_internal_vs_surface_temp.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for sid in sids:
        do = obs_df[obs_df["scenario_id"] == sid]
        anomaly = do["surface_temperature_max"] - do["ambient_temperature"]
        ax.scatter(anomaly.values, do["risk_score"].values,
                   s=5, alpha=0.5, label=SCENARIO_DISPLAY_NAMES.get(sid, sid))
    ax.set_xlabel("Surface Anomaly (°C above ambient)")
    ax.set_ylabel("Risk Score")
    ax.set_title("Risk Score vs Surface Temperature Anomaly")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(gdir / "07_risk_vs_surface_anomaly.png", dpi=150)
    plt.close(fig)

    ew_data = []
    for sid in ["S02", "S03", "S04", "S05"]:
        do = obs_df[obs_df["scenario_id"] == sid]
        t_risk = do[do["risk_score"] >= 60]["timestamp_min"]
        t_risk = int(t_risk.iloc[0]) if len(t_risk) > 0 else None

        anomaly = do["surface_temperature_max"] - do["ambient_temperature"]
        t_surf = do[anomaly >= 5.0]["timestamp_min"]
        t_surf = int(t_surf.iloc[0]) if len(t_surf) > 0 else None

        if t_risk is not None and t_surf is not None:
            window = t_surf - t_risk
        else:
            window = None
        ew_data.append({"scenario": SCENARIO_DISPLAY_NAMES.get(sid, sid), "T_risk_HIGH": t_risk,
                        "T_surface_5C": t_surf, "window_min": window})

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [d["scenario"] for d in ew_data]
    windows = [d["window_min"] if d["window_min"] is not None else 0 for d in ew_data]
    colors = ["green" if w > 0 else "gray" for w in windows]
    ax.bar(labels, windows, color=colors)
    ax.set_ylabel("Early-Warning Window (min)")
    ax.set_title("Early Warning: Risk HIGH Before Surface Anomaly > 5°C")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(gdir / "08_early_warning_window.png", dpi=150)
    plt.close(fig)

    print(f"  [PLOTS]  8 validation graphs saved to {gdir}")
    for ew in ew_data:
        print(f"           {ew['scenario']}: T_risk_HIGH={ew['T_risk_HIGH']}, "
              f"T_surface_5C={ew['T_surface_5C']}, "
              f"window={ew['window_min']} min")

def main():
    print("=" * 65)
    print("  CoalSafe-Sim Version 1")
    print("  Physics-Informed Synthetic Multimodal Dataset Generator")
    print("=" * 65)

    print("\n[PHASE 1] Generating hidden sub-surface state ...")
    latent_raw = generate_all_latent()

    print("\n[PHASE 2] Generating observable sensor data ...")
    obs_df, latent_clean = generate_observations(latent_raw)

    print("\n[PHASE 3] Generating thermal images ...")
    thermal_meta = generate_thermal_images(latent_raw, obs_df)

    print("\n[PHASE 4] Saving CSV files ...")
    cfg.DATA_TABULAR.mkdir(parents=True, exist_ok=True)
    cfg.DATA_META.mkdir(parents=True, exist_ok=True)

    obs_df.to_csv(cfg.DATA_TABULAR / "sensor_data.csv", index=False)
    latent_clean.to_csv(cfg.DATA_META / "latent_state.csv", index=False)
    thermal_meta.to_csv(cfg.DATA_META / "thermal_metadata.csv", index=False)

    sc_df = pd.DataFrame(get_scenarios())
    sc_df = sc_df.drop(columns=["env_variation"], errors="ignore")
    sc_df.to_csv(cfg.DATA_META / "scenarios.csv", index=False)

    print(f"  [SAVE] sensor_data.csv      : {obs_df.shape[0]} rows x {obs_df.shape[1]} cols")
    print(f"  [SAVE] latent_state.csv     : {latent_clean.shape[0]} rows x {latent_clean.shape[1]} cols")
    print(f"  [SAVE] thermal_metadata.csv : {thermal_meta.shape[0]} rows")
    print(f"  [SAVE] scenarios.csv        : {sc_df.shape[0]} rows")

    print("\n[PHASE 5] Generating validation plots ...")
    generate_plots(obs_df, latent_clean, thermal_meta)

    print("\n[PHASE 6] Generating per-minute graph snapshots ...")
    generate_per_minute_graphs(latent_clean)

    print("\n" + "=" * 65)
    print("  [DONE] CoalSafe-Sim Version 1 dataset generation complete.")
    print(f"         Scenarios           : {obs_df['scenario_id'].nunique()}")
    print(f"         Sensor rows         : {len(obs_df)}")
    print(f"         Latent-state rows   : {len(latent_clean)}")
    print(f"         Thermal images      : {(thermal_meta['image_path'] != 'MISSING').sum()}")
    print(f"         Missing images      : {(thermal_meta['image_path'] == 'MISSING').sum()}")

    risk_dist = obs_df["risk_class"].value_counts()
    print(f"\n  Risk distribution:")
    for cls in ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        cnt = risk_dist.get(cls, 0)
        print(f"    {cls:10s} : {cnt:4d} ({cnt/len(obs_df)*100:.1f}%)")
    print("=" * 65)

if __name__ == "__main__":
    main()
