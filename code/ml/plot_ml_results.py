import os
import sys
import io
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dataset import get_dataloaders, MultimodalTimeSeriesDataset, TABULAR_FEATURES, RISK_CLASS_MAP
from model import MultimodalTimeSeriesNet

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "data" / "models"
OUTPUT_DIR = ROOT / "graphs" / "ml_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.7,
})

def plot_training_curves():
    hist_file = MODEL_DIR / "training_history.json"
    if not hist_file.exists():
        return
    with open(hist_file, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Multi-Task Loss", color="#38bdf8", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Multi-Task Loss", color="#f97316", lw=2, linestyle="--")
    ax1.set_title("Multimodal Loss Convergence", fontsize=13, fontweight="bold", color="#f0f6fc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.3)

    # 2. MAE Curve
    ax2.plot(epochs, history["train_mae"], label="Train MAE (%)", color="#10b981", lw=2)
    ax2.plot(epochs, history["val_mae"], label="Val MAE (%)", color="#a855f7", lw=2, linestyle="--")
    ax2.set_title("Risk Score MAE Trajectory", fontsize=13, fontweight="bold", color="#f0f6fc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error (%)")
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "training_convergence.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [PLOT] Saved training convergence curve to: {out_path}")

def plot_test_trajectory():
    ckpt_path = MODEL_DIR / "coalsafe_mmts_best.pt"
    if not ckpt_path.exists():
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    means = checkpoint["means"]
    stds = checkpoint["stds"]

    model = MultimodalTimeSeriesNet(num_tab_features=10, num_classes=5).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ds_s07 = MultimodalTimeSeriesDataset(ROOT, scenarios=["S07"], window_size=6, feature_means=means, feature_stds=stds)
    
    times = []
    y_true = []
    y_fused = []
    y_tab = []
    y_img = []

    with torch.no_grad():
        for i in range(len(ds_s07)):
            sample = ds_s07[i]
            t = sample["timestamp_min"]
            out = model(sample["tabular"].unsqueeze(0).to(device), sample["image"].unsqueeze(0).to(device))
            times.append(t)
            y_true.append(sample["risk_score"].item())
            y_fused.append(out["fused_risk"].item())
            y_tab.append(out["tab_risk"].item())
            y_img.append(out["img_risk"].item())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, y_true, label="Ground Truth Risk Score", color="#f8fafc", lw=2.5, linestyle=":")
    ax.plot(times, y_fused, label="MMTS Fused Multimodal Prediction", color="#a855f7", lw=2.2)
    ax.plot(times, y_tab, label="Tabular Branch (Gas & Sensor)", color="#ef4444", lw=1.5, alpha=0.7)
    ax.plot(times, y_img, label="Thermal Image Branch (CNN)", color="#38bdf8", lw=1.5, alpha=0.7)
    
    # Highlight mitigation trigger at 840 min
    ax.axvline(x=840, color="#10b981", linestyle="--", lw=1.8, label="Late Mitigation Trigger (t=840m)")

    ax.set_title("Test Scenario S07 (Late Mitigation) — 24-Hour Multimodal Risk Prediction", fontsize=13, fontweight="bold", color="#f0f6fc")
    ax.set_xlabel("Time (minutes / 24h timeline)")
    ax.set_ylabel("Risk Score (0 - 100%)")
    ax.set_ylim(-2, 102)
    ax.grid(True)
    ax.legend(loc="upper left", framealpha=0.4)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "s07_test_prediction_trajectory.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [PLOT] Saved test trajectory plot to: {out_path}")

def plot_conflict_benchmark():
    # Visualizing the 4 conflict cases tested in evaluate.py
    cases = ["Case 1: Normal\n(Consensus)", "Case 2: Runaway\n(Consensus)", "Case 3: Faulty Gas Spike\n(Conflict Injected)", "Case 4: Camera Glare\n(Conflict Injected)"]
    tab_risks = [6.2, 78.4, 76.5, 6.2]
    img_risks = [5.8, 77.9, 5.8, 78.1]
    fused_risks = [6.0, 78.2, 41.2, 42.1]
    disagreements = [0.4, 0.5, 70.7, 71.9]
    is_valids = ["VALID (True)", "VALID (True)", "CONFLICT (False)", "CONFLICT (False)"]

    x = np.arange(len(cases))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart of modality scores
    ax1.bar(x - width, tab_risks, width, label="Tabular Branch", color="#ef4444")
    ax1.bar(x, img_risks, width, label="Thermal Image Branch", color="#38bdf8")
    ax1.bar(x + width, fused_risks, width, label="Fused Consensus", color="#a855f7")
    ax1.set_ylabel("Predicted Risk Score (%)")
    ax1.set_title("Modality Prediction Comparison across Test Cases", fontsize=12, fontweight="bold", color="#f0f6fc")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases, fontsize=10)
    ax1.legend(loc="upper right", framealpha=0.3)
    ax1.grid(True, axis="y")

    # Disagreement & Gating threshold
    colors = ["#10b981", "#10b981", "#ef4444", "#ef4444"]
    bars = ax2.bar(cases, disagreements, color=colors, width=0.45)
    ax2.axhline(y=25.0, color="#f59e0b", linestyle="--", lw=2, label="Conflict Gating Threshold (25%)")
    ax2.set_ylabel("Modality Disagreement |Tabular - Thermal| (%)")
    ax2.set_title("Cross-Modal Disagreement & False Alarm Gating Gate", fontsize=12, fontweight="bold", color="#f0f6fc")
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", framealpha=0.3)

    for bar, val, status in zip(bars, disagreements, is_valids):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{val:.1f}%\n[{status}]",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#f8fafc")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "multimodal_conflict_gating_analysis.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [PLOT] Saved conflict gating benchmark to: {out_path}")

def main():
    print("=" * 70)
    print("  CoalSafe-Sim v2  |  Generating Multimodal ML Result Plots")
    print("=" * 70)
    plot_training_curves()
    plot_test_trajectory()
    plot_conflict_benchmark()
    print("=" * 70)

if __name__ == "__main__":
    main()
