"""
generate_per_minute_graphs.py — Generates per-minute cumulative graph snapshots.

For each simulation minute (0 to 120), creates an "Internal Temperature vs Time"
graph showing data from minute 0 up to that minute for all 8 scenarios.
Graphs are saved to graphs/per_minute/ directory.
"""

import sys
import io
import pathlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CODE_DIR = pathlib.Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from generate_scenarios import get_scenarios

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

SCENARIO_FOLDERS = {
    "S01": "S01_Stable_Baseline",
    "S02": "S02_Early_Oxidation",
    "S03": "S03_Developing_Core",
    "S04": "S04_Heat_Migration",
    "S05": "S05_High_Risk",
    "S06": "S06_Early_Mitigation",
    "S07": "S07_Late_Mitigation",
    "S08": "S08_Environmental_Variation",
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


def generate_per_minute_graphs(latent_df=None):
    """Generate 121 per-minute cumulative Internal Temperature vs Time graphs for each scenario folder and combined."""

    if latent_df is None:
        latent_path = cfg.DATA_META / "latent_state.csv"
        if not latent_path.exists():
            print("  [ERROR] latent_state.csv not found. Run run_all.py first.")
            return
        latent_df = pd.read_csv(latent_path)

    base_out_dir = cfg.GRAPHS_DIR / "per_minute"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = get_scenarios()
    sids = [s["scenario_id"] for s in scenarios]

    y_min = latent_df["internal_temperature"].min() - 2
    y_max = latent_df["internal_temperature"].max() + 5
    total_minutes = cfg.NUM_TIME_POINTS  # 121 (0 to 120)

    # 1. Create folders for each scenario
    for sid in sids:
        folder_name = SCENARIO_FOLDERS[sid]
        s_dir = base_out_dir / folder_name
        s_dir.mkdir(parents=True, exist_ok=True)

    combined_dir = base_out_dir / "combined_all_scenarios"
    combined_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [PER-MIN] Generating per-minute graphs for each scene under {base_out_dir} ...")

    # Get baseline S01 data for reference comparison in individual scenario graphs
    s01_data = latent_df[latent_df["scenario_id"] == "S01"]

    for sid in sids:
        folder_name = SCENARIO_FOLDERS[sid]
        s_dir = base_out_dir / folder_name
        s_name = SCENARIO_DISPLAY_NAMES[sid]
        s_color = SCENARIO_COLORS[sid]
        st = SCENARIO_STYLES[sid]
        d_scenario = latent_df[latent_df["scenario_id"] == sid]

        for t in range(total_minutes):
            fig, ax = plt.subplots(figsize=(10, 5))

            # Reference baseline line S01 (if current scenario is not S01)
            if sid != "S01":
                s01_slice = s01_data[s01_data["timestamp_min"] <= t]
                ax.plot(
                    s01_slice["timestamp_min"],
                    s01_slice["internal_temperature"],
                    label="Stable Baseline Reference",
                    color="#004578",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.6,
                )

            # Scenario curve
            d_slice = d_scenario[d_scenario["timestamp_min"] <= t]
            ax.plot(
                d_slice["timestamp_min"],
                d_slice["internal_temperature"],
                label=s_name,
                color=s_color,
                linestyle=st["ls"],
                linewidth=st["lw"] + 0.5,
                marker=st["marker"],
                markevery=12 if st["marker"] else None,
                markersize=6,
            )

            # Annotations for mitigation triggers
            if sid == "S06" and t >= 45:
                ax.axvline(45, color="#16a34a", linestyle="--", alpha=0.7, label="Early Mitigation Trigger (t=45m)")
            elif sid == "S07" and t >= 85:
                ax.axvline(85, color="#9333ea", linestyle="--", alpha=0.7, label="Late Mitigation Trigger (t=85m)")

            ax.set_xlabel("Time (min)", fontsize=10, fontweight="bold")
            ax.set_ylabel("Internal Temperature (°C)", fontsize=10, fontweight="bold")
            ax.set_title(f"{s_name} — Internal Temp vs Time (Minute {t})", fontsize=11, fontweight="bold")
            ax.set_xlim(0, cfg.DURATION_MIN)
            ax.set_ylim(y_min, y_max)
            ax.legend(fontsize=8, loc="upper left", framealpha=0.95, handlelength=3.5)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            filename = f"internal_temp_minute_{t:03d}.png"
            fig.savefig(s_dir / filename, dpi=100)
            plt.close(fig)

        print(f"           Scene {sid} ({s_name}): 121 per-minute images saved to {folder_name}/")

    # 2. Combined per-minute graphs
    print("  [PER-MIN] Generating combined 8-scenario per-minute graphs ...")
    for t in range(total_minutes):
        fig, ax = plt.subplots(figsize=(12, 5.5))

        for sid in sids:
            d = latent_df[latent_df["scenario_id"] == sid]
            d_slice = d[d["timestamp_min"] <= t]
            st = SCENARIO_STYLES.get(sid, {"ls": "-", "lw": 1.5, "marker": None})

            ax.plot(
                d_slice["timestamp_min"],
                d_slice["internal_temperature"],
                label=SCENARIO_DISPLAY_NAMES.get(sid, sid),
                color=SCENARIO_COLORS.get(sid, None),
                linestyle=st["ls"],
                linewidth=st["lw"],
                marker=st["marker"],
                markevery=12 if st["marker"] else None,
                markersize=5,
            )

        if t >= 45:
            ax.axvline(45, color="#16a34a", linestyle=":", alpha=0.5, label="_nolegend_")
        if t >= 85:
            ax.axvline(85, color="#9333ea", linestyle=":", alpha=0.5, label="_nolegend_")

        ax.set_xlabel("Time (min)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Internal Temperature (°C)", fontsize=10, fontweight="bold")
        ax.set_title(f"Internal Temperature vs Time — Minute {t}", fontsize=12, fontweight="bold")
        ax.set_xlim(0, cfg.DURATION_MIN)
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.95, handlelength=3.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        filename = f"internal_temp_minute_{t:03d}.png"
        fig.savefig(base_out_dir / filename, dpi=100)
        fig.savefig(combined_dir / filename, dpi=100)
        plt.close(fig)

    print(f"  [PER-MIN] Done! All scenario subfolders and combined graphs generated under {base_out_dir}")


if __name__ == "__main__":
    generate_per_minute_graphs()
