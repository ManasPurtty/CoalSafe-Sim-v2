"""
=============================================================================
  CoalSafe-Sim v2  |  Forecast Validation & Backtesting Framework
=============================================================================
  Purpose:
    Validates the trustworthiness of our +1h Future Forecasting model using
    standard time-series forecasting evaluation criteria from the field:

    1. Walk-Forward Backtesting         - step through 24h data, predict +1h
                                          at each step, compare with actual
    2. Regression Metrics:
       - MAE   (Mean Absolute Error)
       - RMSE  (Root Mean Square Error)
       - MAPE  (Mean Absolute Percentage Error)
       - R²    (Coefficient of Determination)
    3. Safety-Critical Metrics:
       - Hit Rate         - correctly warned of real danger
       - False Alarm Rate - falsely triggered when safe
       - Miss Rate        - failed to warn of real danger (most critical!)
       - Precision, Recall, F1-Score
    4. Directional Accuracy             - did model predict UP/DOWN correctly?
    5. Horizon Degradation Analysis     - accuracy at +30min vs +60min
    6. Confidence Band Coverage         - is actual value in predicted range?

  Approach:
    Since we don't have real-world data, we use our 24-hour simulation as
    "ground truth" and validate using walk-forward backtesting:
      - Train window: Scenarios S01-S06
      - Held-out validation: Scenarios S07, S08 (never seen during training)
      - At each time step t, predict risk at t+60m, compare with actual t+60m

  Output:
    - graphs/ml_results/forecast_validation_metrics.png
    - graphs/ml_results/walkforward_backtest_S07.png
    - graphs/ml_results/walkforward_backtest_S08.png
    - data/models/forecast_validation_report.json
=============================================================================
"""

import os
import sys
import io
import json
import pathlib
# Force UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ── Path Setup ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
from ml.dataset import MultimodalTimeSeriesDataset, TABULAR_FEATURES
from ml.model import MultimodalTimeSeriesNet
from torch.utils.data import DataLoader

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "data" / "models" / "coalsafe_mmts_best.pt"
GRAPH_DIR  = ROOT / "graphs" / "ml_results"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ROOT / "data" / "models" / "forecast_validation_report.json"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#e2e8f0",
    "grid.color":       "#334155",
    "grid.alpha":       0.4,
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#475569",
    "font.size":        9,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
})

COLORS = {
    "actual":     "#22d3ee",   # cyan
    "predicted":  "#f97316",   # orange
    "error":      "#ef4444",   # red
    "good":       "#22c55e",   # green
    "warn":       "#eab308",   # yellow
    "danger":     "#dc2626",   # dark red
    "safe":       "#3b82f6",   # blue
    "band":       "#f97316",   # orange (for confidence band)
}

DANGER_THRESHOLD = 60.0   # % risk → action threshold

# =============================================================================
#  STEP 1: Load Model
# =============================================================================
def load_model():
    model = MultimodalTimeSeriesNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  [MODEL]  Loaded checkpoint from: {MODEL_PATH}")
    return model

# =============================================================================
#  STEP 2: Walk-Forward Backtesting on a Scenario
# =============================================================================
def walk_forward_backtest(model, scenario_ids, train_means, train_stds, label):
    """
    For each time step t in the scenario:
      - Feed the 60-min history window [t-60m ... t]
      - Get model prediction: risk at t+60m
      - Compare with ACTUAL risk at t+60m from simulation data
    """
    dataset = MultimodalTimeSeriesDataset(
        root_dir=ROOT,
        scenarios=scenario_ids,
        feature_means=train_means,
        feature_stds=train_stds,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    y_true_now    = []  # actual risk at t (present)
    y_pred_now    = []  # model predicted risk at t
    y_true_60m    = []  # actual risk at t+60m (ground truth)
    y_pred_60m    = []  # model predicted risk at t+60m
    y_true_30m_list = []  # actual risk at t+30m
    y_pred_30m_list = []  # predicted risk at t+30m
    y_true_cls    = []  # actual class at t+60m
    y_pred_cls    = []  # predicted class at t+60m

    with torch.no_grad():
        for batch in loader:
            x_tab = batch["tabular"].to(DEVICE)
            x_img = batch["image"].to(DEVICE)

            out = model(x_tab, x_img)

            risk_now   = out["fused_risk_now"].item()
            risk_30m   = out["risk_forecast_30m"].item()
            risk_60m   = out["risk_forecast_60m"].item()
            class_60m  = out["class_logits_60m"].argmax(dim=-1).item()

            true_now   = batch["risk_score_now"].item()
            true_30m   = batch["risk_score_30m"].item()
            true_60m   = batch["risk_score_60m"].item()
            true_cls   = batch["risk_class_60m"].item()

            y_true_now.append(true_now)
            y_pred_now.append(risk_now)
            y_true_60m.append(true_60m)
            y_pred_60m.append(risk_60m)
            y_true_cls.append(true_cls)
            y_pred_cls.append(class_60m)
            y_true_30m_list.append(true_30m)
            y_pred_30m_list.append(risk_30m)

    results = {
        "label":          label,
        "y_true_now":     np.array(y_true_now),
        "y_pred_now":     np.array(y_pred_now),
        "y_true_60m":     np.array(y_true_60m),
        "y_pred_60m":     np.array(y_pred_60m),
        "y_true_cls":     np.array(y_true_cls),
        "y_pred_cls":     np.array(y_pred_cls),
        "y_true_30m":     np.array(y_true_30m_list),
        "y_pred_30m":     np.array(y_pred_30m_list),
    }
    return results

# =============================================================================
#  STEP 3: Compute All Standard Evaluation Metrics
# =============================================================================
def compute_metrics(results):
    y_true = results["y_true_60m"]
    y_pred = results["y_pred_60m"]
    y_tc   = results["y_true_cls"]
    y_pc   = results["y_pred_cls"]
    label  = results["label"]

    n = len(y_true)

    # ── Regression Metrics ──────────────────────────────────────────────────
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — Mean Absolute Percentage Error
    # Avoid divide by zero: skip samples where true=0
    nonzero = y_true > 1.0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if nonzero.sum() > 0 else 0.0

    # ── Directional Accuracy (Trend Hit Rate) ────────────────────────────────
    # Did the model correctly predict if risk goes UP or DOWN vs present?
    y_true_now = results["y_true_now"]
    delta_true = np.sign(y_true - y_true_now)    # actual direction
    delta_pred = np.sign(y_pred - results["y_pred_now"])  # predicted direction
    dir_acc = np.mean(delta_true == delta_pred) * 100

    # ── Safety-Critical Binary Metrics (Danger = risk >= 60%) ────────────────
    binary_true = (y_true >= DANGER_THRESHOLD).astype(int)
    binary_pred = (y_pred >= DANGER_THRESHOLD).astype(int)

    # Avoid division errors for edge cases
    tp = np.sum((binary_pred == 1) & (binary_true == 1))
    fp = np.sum((binary_pred == 1) & (binary_true == 0))
    tn = np.sum((binary_pred == 0) & (binary_true == 0))
    fn = np.sum((binary_pred == 0) & (binary_true == 1))

    hit_rate        = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 100.0  # Recall for danger
    false_alarm_rate= (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0   # FPR
    miss_rate       = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0   # FNR (most critical!)
    precision_val   = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 100.0
    f1              = 2 * (precision_val * hit_rate) / (precision_val + hit_rate) if (precision_val + hit_rate) > 0 else 0.0

    # ── Confidence Band Coverage (95% band = pred ± 1.96*RMSE) ──────────────
    band_half = 1.96 * rmse
    in_band   = np.sum((y_true >= y_pred - band_half) & (y_true <= y_pred + band_half))
    coverage  = (in_band / n) * 100

    # ── Horizon Degradation (compare 30m vs 60m error) ──────────────────────
    y_true_30m = results.get("y_true_30m", None)
    y_pred_30m = results.get("y_pred_30m", None)
    mae_30m    = mean_absolute_error(y_true_30m, y_pred_30m) if y_true_30m is not None else None

    metrics = {
        "scenario":           label,
        "n_samples":          n,
        # Regression
        "mae_60m":            round(mae, 4),
        "rmse_60m":           round(rmse, 4),
        "mape_60m":           round(mape, 2),
        "r2_60m":             round(r2, 4),
        # Directional
        "directional_accuracy": round(dir_acc, 2),
        # Safety binary
        "hit_rate_pct":       round(hit_rate, 2),
        "false_alarm_rate_pct": round(false_alarm_rate, 2),
        "miss_rate_pct":      round(miss_rate, 2),
        "precision_pct":      round(precision_val, 2),
        "f1_score":           round(f1, 2),
        # Coverage
        "confidence_band_coverage_pct": round(coverage, 2),
        # Counts
        "true_positives":     int(tp),
        "false_positives":    int(fp),
        "true_negatives":     int(tn),
        "false_negatives":    int(fn),
    }

    if mae_30m is not None:
        metrics["mae_30m"] = round(mae_30m, 4)
        metrics["horizon_degradation_pct"] = round(((mae - mae_30m) / mae_30m) * 100, 1)

    return metrics

# =============================================================================
#  STEP 4: Walk-Forward Backtest Plot for One Scenario
# =============================================================================
def plot_walkforward(results, metrics, save_path):
    y_true    = results["y_true_60m"]
    y_pred    = results["y_pred_60m"]
    y_true_now= results["y_true_now"]
    label     = results["label"]
    n         = len(y_true)
    t         = np.arange(n) * 10  # minutes (10-min steps)

    error       = np.abs(y_true - y_pred)
    band_half   = 1.96 * metrics["rmse_60m"]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Walk-Forward Backtest: {label}  |  +1-Hour Future Forecasting Validation",
        fontsize=13, fontweight="bold", color="#f1f5f9", y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── Panel 1: Prediction vs Actual (top span) ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(t, y_pred - band_half, y_pred + band_half,
                     alpha=0.18, color=COLORS["band"], label="95% Confidence Band")
    ax1.plot(t, y_true, color=COLORS["actual"], linewidth=1.8,
             label=f"Actual +1h Risk (Ground Truth)", zorder=3)
    ax1.plot(t, y_pred, color=COLORS["predicted"], linewidth=1.6,
             linestyle="--", label=f"AI Predicted +1h Risk", zorder=3)
    ax1.axhline(DANGER_THRESHOLD, color="#ef4444", linewidth=1.2,
                linestyle=":", label=f"Danger Threshold ({DANGER_THRESHOLD}%)")
    ax1.set_ylabel("Risk Score (%)")
    ax1.set_title(f"Predicted vs Actual Future Risk  |  R²={metrics['r2_60m']:.4f}  |  MAE={metrics['mae_60m']:.2f}%  |  MAPE={metrics['mape_60m']:.1f}%")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(0, 105)
    ax1.grid(True)

    # ── Panel 2: Prediction Error Over Time ──────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(t, error, color=COLORS["error"], alpha=0.7, width=8,
            label="Absolute Error per Step")
    ax2.axhline(metrics["mae_60m"], color=COLORS["warn"], linewidth=1.5,
                linestyle="--", label=f"Mean Error (MAE={metrics['mae_60m']:.2f}%)")
    ax2.set_ylabel("Absolute Error (%)")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_title("Prediction Error Over Time (Walk-Forward)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, t[-1])
    ax2.grid(True)

    # ── Panel 3: Directional Accuracy ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    delta_true = np.sign(y_true - y_true_now)
    delta_pred = np.sign(y_pred - results["y_pred_now"])
    correct_dir = (delta_true == delta_pred).astype(int)
    window = 20
    rolling_dir = pd.Series(correct_dir).rolling(window, min_periods=1).mean() * 100

    ax3.plot(t, rolling_dir, color=COLORS["good"], linewidth=1.8,
             label=f"Rolling Directional Accuracy ({window}-step window)")
    ax3.axhline(50, color="#64748b", linewidth=1, linestyle=":")
    ax3.axhline(metrics["directional_accuracy"], color=COLORS["warn"],
                linewidth=1.4, linestyle="--",
                label=f"Overall Dir. Acc: {metrics['directional_accuracy']:.1f}%")
    ax3.set_ylabel("Directional Accuracy (%)")
    ax3.set_xlabel("Time (minutes)")
    ax3.set_title("Trend Direction Prediction Accuracy (UP/DOWN)")
    ax3.set_ylim(0, 110)
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, t[-1])
    ax3.grid(True)

    # ── Panel 4: Safety Confusion Matrix ─────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    conf = np.array([
        [metrics["true_negatives"],  metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]]
    ])
    im = ax4.imshow(conf, cmap="RdYlGn", vmin=0)
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Predicted SAFE", "Predicted DANGER"])
    ax4.set_yticklabels(["Actual SAFE", "Actual DANGER"])
    ax4.set_title("Safety Alarm Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(conf[i, j]),
                     ha="center", va="center", fontsize=13, fontweight="bold",
                     color="#0f172a")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # ── Panel 5: Metrics Summary ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    miss_color   = "#22c55e" if metrics["miss_rate_pct"] == 0 else "#ef4444"
    alarm_color  = "#22c55e" if metrics["false_alarm_rate_pct"] < 10 else "#eab308"
    hit_color    = "#22c55e" if metrics["hit_rate_pct"] >= 80 else "#ef4444"

    summary_text = (
        f"{'EVALUATION METRICS SUMMARY':^38}\n"
        f"{'='*38}\n"
        f"  REGRESSION ACCURACY\n"
        f"  MAE  (Mean Abs Error)    : {metrics['mae_60m']:.2f}%\n"
        f"  RMSE (Root Mean Sq Err)  : {metrics['rmse_60m']:.2f}%\n"
        f"  MAPE (Mean Abs % Error)  : {metrics['mape_60m']:.1f}%\n"
        f"  R²   (Explained Variance): {metrics['r2_60m']:.4f}\n"
        f"\n"
        f"  DIRECTIONAL ACCURACY\n"
        f"  Trend Hit Rate           : {metrics['directional_accuracy']:.1f}%\n"
        f"\n"
        f"  SAFETY-CRITICAL METRICS\n"
        f"  Hit Rate (Danger Recall) : {metrics['hit_rate_pct']:.1f}%\n"
        f"  Miss Rate (Danger Missed): {metrics['miss_rate_pct']:.1f}%  ← critical\n"
        f"  False Alarm Rate         : {metrics['false_alarm_rate_pct']:.1f}%\n"
        f"  Precision                : {metrics['precision_pct']:.1f}%\n"
        f"  F1-Score                 : {metrics['f1_score']:.1f}\n"
        f"\n"
        f"  95% CONFIDENCE COVERAGE  : {metrics['confidence_band_coverage_pct']:.1f}%\n"
        f"{'='*38}"
    )

    ax5.text(0.05, 0.97, summary_text,
             transform=ax5.transAxes,
             fontsize=8.2,
             verticalalignment="top",
             fontfamily="monospace",
             color="#e2e8f0",
             bbox=dict(boxstyle="round", facecolor="#1e293b", edgecolor="#475569", alpha=0.95))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  [PLOT]   Saved walk-forward backtest plot: {save_path}")

# =============================================================================
#  STEP 5: Multi-Scenario Metrics Comparison Bar Chart
# =============================================================================
def plot_metrics_comparison(all_metrics, save_path):
    scenarios = [m["scenario"] for m in all_metrics]
    n_s = len(scenarios)
    x   = np.arange(n_s)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Forecast Validation: Multi-Metric Comparison Across All Held-Out Scenarios",
        fontsize=13, fontweight="bold", color="#f1f5f9", y=0.99
    )
    fig.patch.set_facecolor("#0f172a")

    def bar(ax, vals, title, ylabel, color, threshold=None, lower_is_better=True):
        bars = ax.bar(x, vals, color=[
            COLORS["good"] if (v <= threshold if lower_is_better else v >= threshold) else COLORS["error"]
            for v in vals
        ] if threshold is not None else [color]*n_s, width=0.5, zorder=3)
        for bar_, val in zip(bars, vals):
            ax.text(bar_.get_x() + bar_.get_width()/2, bar_.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, color="#e2e8f0")
        if threshold is not None:
            ax.axhline(threshold, color=COLORS["warn"], linewidth=1.3,
                       linestyle="--", label=f"Target: {threshold}")
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_facecolor("#1e293b")
        ax.grid(True, axis="y", alpha=0.4)

    bar(axes[0][0], [m["mae_60m"] for m in all_metrics],
        "MAE — Mean Absolute Error (%)", "MAE (%)", COLORS["predicted"],
        threshold=10.0, lower_is_better=True)

    bar(axes[0][1], [m["rmse_60m"] for m in all_metrics],
        "RMSE — Root Mean Square Error (%)", "RMSE (%)", COLORS["error"],
        threshold=15.0, lower_is_better=True)

    bar(axes[0][2], [m["r2_60m"] for m in all_metrics],
        "R² — Explained Variance (higher = better)", "R²", COLORS["actual"],
        threshold=0.60, lower_is_better=False)

    bar(axes[1][0], [m["hit_rate_pct"] for m in all_metrics],
        "Hit Rate — Danger Correctly Detected (%)", "Hit Rate (%)", COLORS["good"],
        threshold=70.0, lower_is_better=False)

    bar(axes[1][1], [m["miss_rate_pct"] for m in all_metrics],
        "Miss Rate — Danger MISSED (%) ← Must Be 0!", "Miss Rate (%)", COLORS["error"],
        threshold=10.0, lower_is_better=True)

    bar(axes[1][2], [m["directional_accuracy"] for m in all_metrics],
        "Directional Accuracy — Trend UP/DOWN (%)", "Dir. Acc (%)", COLORS["safe"],
        threshold=60.0, lower_is_better=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  [PLOT]   Saved metrics comparison chart: {save_path}")

# =============================================================================
#  MAIN
# =============================================================================
def main():
    print("=" * 75)
    print("  CoalSafe-Sim v2  |  Forecast Validation & Walk-Forward Backtesting")
    print("=" * 75)

    # ── Get training normalization stats ──────────────────────────────────
    print("\n  [STEP 1] Computing normalization stats from training scenarios...")
    train_ds = MultimodalTimeSeriesDataset(
        root_dir=ROOT,
        scenarios=["S01","S02","S03","S04","S05","S06"],
    )
    train_means = train_ds.feature_means
    train_stds  = train_ds.feature_stds
    print(f"  [DATA]   Training scenarios: S01-S06 | {len(train_ds)} samples")

    # ── Load best model ───────────────────────────────────────────────────
    print("\n  [STEP 2] Loading trained model checkpoint...")
    model = load_model()

    # ── Walk-Forward Backtesting on HELD-OUT Scenarios ────────────────────
    print("\n  [STEP 3] Running Walk-Forward Backtesting on held-out scenarios...")
    all_metrics = []
    backtest_scenarios = [
        (["S07"], "Scenario S07 (Reactive Late Mitigation)"),
        (["S08"], "Scenario S08 (Partial Sensor Fault)"),
    ]

    for sc_ids, sc_label in backtest_scenarios:
        print(f"\n  -- Backtesting: {sc_label} --")
        results = walk_forward_backtest(model, sc_ids, train_means, train_stds, sc_label)

        metrics = compute_metrics(results)
        all_metrics.append(metrics)

        # Print results to console
        W = 55
        print(f"  +{'-'*W}+")
        print(f"  |  Validation Results: {sc_label}")
        print(f"  +{'-'*W}+")
        print(f"  |  REGRESSION METRICS (+1h Forecast)")
        print(f"  |  MAE  (Mean Absolute Error)     : {metrics['mae_60m']:>6.2f}%")
        print(f"  |  RMSE (Root Mean Square Error)  : {metrics['rmse_60m']:>6.2f}%")
        print(f"  |  MAPE (Mean Abs Percentage Err) : {metrics['mape_60m']:>6.1f}%")
        print(f"  |  R2   (Explained Variance)      : {metrics['r2_60m']:>6.4f}")
        print(f"  +{'-'*W}+")
        print(f"  |  DIRECTIONAL ACCURACY")
        print(f"  |  Trend Prediction Accuracy      : {metrics['directional_accuracy']:>6.1f}%")
        print(f"  +{'-'*W}+")
        print(f"  |  SAFETY-CRITICAL METRICS (Danger threshold: 60%)")
        print(f"  |  Hit Rate  (Danger Detected)    : {metrics['hit_rate_pct']:>6.1f}%")
        print(f"  |  Miss Rate (Danger MISSED) ***  : {metrics['miss_rate_pct']:>6.1f}%  <- critical!")
        print(f"  |  False Alarm Rate               : {metrics['false_alarm_rate_pct']:>6.1f}%")
        print(f"  |  Precision                      : {metrics['precision_pct']:>6.1f}%")
        print(f"  |  F1-Score                       : {metrics['f1_score']:>6.1f}")
        print(f"  +{'-'*W}+")
        print(f"  |  95% Confidence Band Coverage   : {metrics['confidence_band_coverage_pct']:>6.1f}%")
        print(f"  +{'-'*W}+")

        if "horizon_degradation_pct" in metrics:
            print(f"  [HORIZON] +30min MAE: {metrics['mae_30m']:.2f}%  |  +60min MAE: {metrics['mae_60m']:.2f}%  "
                  f"|  Degradation: +{metrics['horizon_degradation_pct']:.1f}%")

        # Plot walk-forward backtest for this scenario
        plot_walkforward(
            results, metrics,
            GRAPH_DIR / f"walkforward_backtest_{sc_ids[0]}.png"
        )

    # ── Multi-Metric Comparison Chart ─────────────────────────────────────
    print("\n  [STEP 4] Generating multi-scenario metrics comparison chart...")
    plot_metrics_comparison(all_metrics, GRAPH_DIR / "forecast_validation_metrics.png")

    # ── Save JSON Report ──────────────────────────────────────────────────
    report = {
        "validation_type":   "walk_forward_backtest",
        "approach":          "24h simulation used as ground truth; held-out scenarios S07/S08 for validation",
        "training_scenarios": ["S01","S02","S03","S04","S05","S06"],
        "held_out_scenarios": ["S07","S08"],
        "danger_threshold_pct": DANGER_THRESHOLD,
        "results":           all_metrics,
        "interpretation": {
            "MAE":             "Average prediction error in % risk — lower is better. <10% is good.",
            "RMSE":            "Penalizes large errors more than MAE — lower is better.",
            "MAPE":            "Error relative to actual value (%). <20% is acceptable.",
            "R2":              "0=no explanation, 1=perfect. Above 0.6 is good for forecasting.",
            "hit_rate":        "Of all real dangerous moments, what % did AI detect? Must be high.",
            "miss_rate":       "Of all real dangerous moments, what % did AI MISS? Must be 0%.",
            "false_alarm_rate":"Of all safe moments, what % did AI falsely alarm? Should be low.",
            "directional_acc": "Did AI predict if risk goes UP or DOWN? >60% is better than random.",
            "band_coverage":   "Is actual value inside 95% confidence range? Should be ~95%.",
        }
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  [SAVED]  Forecast validation report: {REPORT_PATH}")
    print("\n  [OUTPUTS]")
    print(f"  • graphs/ml_results/walkforward_backtest_S07.png")
    print(f"  • graphs/ml_results/walkforward_backtest_S08.png")
    print(f"  • graphs/ml_results/forecast_validation_metrics.png")
    print(f"  • data/models/forecast_validation_report.json")
    print("\n" + "=" * 75)
    print("  [SUCCESS] Forecast Validation Complete!")
    print("=" * 75)

if __name__ == "__main__":
    main()
