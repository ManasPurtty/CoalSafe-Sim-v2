import os
import sys
import io
import json
import pathlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dataset import get_dataloaders, MultimodalTimeSeriesDataset, TABULAR_FEATURES
from model import MultimodalTimeSeriesNet

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "data" / "models"

def run_evaluation():
    print("=" * 75)
    print("  CoalSafe-Sim v2  |  Multimodal +1h Future Forecaster & Mitigation Suite")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = MODEL_DIR / "coalsafe_mmts_best.pt"

    if not ckpt_path.exists():
        print(f"  [ERROR] Checkpoint not found at {ckpt_path}. Run train.py first.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    means = checkpoint["means"]
    stds = checkpoint["stds"]

    model = MultimodalTimeSeriesNet(num_tab_features=10, num_classes=5, conflict_threshold=25.0).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test loader (S07)
    _, _, test_loader, _ = get_dataloaders(
        root_dir=ROOT,
        batch_size=16,
        window_size=6,
        sampling_step=10,
        forecast_horizon_min=60,
        train_scenarios=("S01", "S02", "S03", "S05"),
        val_scenarios=("S04", "S06", "S08"),
        test_scenarios=("S07",),
    )

    y_true_now, y_pred_now = [], []
    y_true_60m, y_pred_60m = [], []
    temp_true_60m, temp_pred_60m = [], []
    cls_true_60m, cls_pred_60m = [], []

    with torch.no_grad():
        for batch in test_loader:
            x_tab = batch["tabular"].to(device)
            x_img = batch["image"].to(device)
            
            y_risk_now = batch["risk_score_now"].to(device)
            y_risk_60m = batch["risk_score_60m"].to(device)
            y_temp_60m = batch["temp_core_60m"].to(device)
            y_cls_60m  = batch["risk_class_60m"].to(device)

            out = model(x_tab, x_img)

            y_true_now.extend(y_risk_now.cpu().numpy())
            y_pred_now.extend(out["fused_risk_now"].cpu().numpy())

            y_true_60m.extend(y_risk_60m.cpu().numpy())
            y_pred_60m.extend(out["risk_forecast_60m"].cpu().numpy())

            temp_true_60m.extend(y_temp_60m.cpu().numpy())
            temp_pred_60m.extend(out["temp_core_forecast_60m"].cpu().numpy())

            cls_true_60m.extend(y_cls_60m.cpu().numpy())
            cls_pred_60m.extend(torch.argmax(out["class_logits_60m"], dim=1).cpu().numpy())

    y_true_60m = np.array(y_true_60m)
    y_pred_60m = np.array(y_pred_60m)
    cls_true_60m = np.array(cls_true_60m)
    cls_pred_60m = np.array(cls_pred_60m)

    # Metrics on +1h Future Forecast
    mae_60m = mean_absolute_error(y_true_60m, y_pred_60m)
    rmse_60m = np.sqrt(mean_squared_error(y_true_60m, y_pred_60m))
    r2_60m = r2_score(y_true_60m, y_pred_60m)
    acc_60m = accuracy_score(cls_true_60m, cls_pred_60m) * 100.0
    f1_60m = f1_score(cls_true_60m, cls_pred_60m, average="weighted") * 100.0

    print("\n  [TEST METRICS — Held-out Unseen Scenario S07 (+1 Hour Future Forecast)]")
    print(f"  • +1h Forecast MAE           : {mae_60m:.2f}%")
    print(f"  • +1h Forecast RMSE          : {rmse_60m:.2f}%")
    print(f"  • +1h Forecast R² Score      : {r2_60m:.4f}")
    print(f"  • +1h Classification Acc     : {acc_60m:.1f}%")
    print(f"  • +1h Weighted F1-Score      : {f1_60m:.1f}%")

    # =========================================================================
    # INJECTED CONFLICT & PREDICTIVE MITIGATION BENCHMARKS
    # =========================================================================
    print("\n" + "-" * 75)
    print("  [CROSS-MODAL CONFLICT & PREDICTIVE EARLY MITIGATION BENCHMARKS]")
    print("-" * 75)

    # 1. Normal Consensus Test (S01)
    ds_s01 = MultimodalTimeSeriesDataset(ROOT, scenarios=["S01"], window_size=6, feature_means=means, feature_stds=stds)
    sample_s01 = ds_s01[10]
    with torch.no_grad():
        out1 = model(sample_s01["tabular"].unsqueeze(0).to(device), sample_s01["image"].unsqueeze(0).to(device))
        dec1 = model.evaluate_predictive_mitigation(out1)
    
    diff1 = torch.abs(out1["tab_risk_now"] - out1["img_risk_now"]).item()
    print("\n  [CASE 1: Normal Consensus — S01 Baseline]")
    print(f"  • Present Risk (t=now)      : {dec1['risk_now'].item():.1f}%")
    print(f"  • Forecasted Risk (+1h)     : {dec1['risk_forecast_60m'].item():.1f}%")
    print(f"  • Forecasted Core Temp (+1h): {dec1['temp_core_forecast_60m'].item():.1f}°C")
    print(f"  • Disagreement (|Δ|)        : {diff1:.1f}%")
    print(f"  • Decision Status           : {dec1['status']}")
    print(f"  • Controller Action         : {dec1['action']}")

    # 2. Confirmed Runaway Consensus Test (S05 at t=780m, before peak)
    ds_s05 = MultimodalTimeSeriesDataset(ROOT, scenarios=["S05"], window_size=6, feature_means=means, feature_stds=stds)
    sample_s05 = ds_s05[-10] # developing runaway
    with torch.no_grad():
        out2 = model(sample_s05["tabular"].unsqueeze(0).to(device), sample_s05["image"].unsqueeze(0).to(device))
        dec2 = model.evaluate_predictive_mitigation(out2)
    
    diff2 = torch.abs(out2["tab_risk_now"] - out2["img_risk_now"]).item()
    print("\n  [CASE 2: Predictive Early Mitigation — S05 Developing Runaway]")
    print(f"  • Present Risk (t=now)      : {dec2['risk_now'].item():.1f}%")
    print(f"  • Forecasted Risk (+1h)     : {dec2['risk_forecast_60m'].item():.1f}% (Projected HIGH RISK!)")
    print(f"  • Forecasted Core Temp (+1h): {dec2['temp_core_forecast_60m'].item():.1f}°C")
    print(f"  • Disagreement (|Δ|)        : {diff2:.1f}%")
    print(f"  • Decision Status           : ★ {dec2['status']} ★")
    print(f"  • Controller Action         : {dec2['action']}")

    # 3. Injected Sensor Glitch (Faulty Gas Sensor Spike on Normal S01)
    corrupted_tab = sample_s01["tabular"].clone()
    co_idx = TABULAR_FEATURES.index("CO")
    corrupted_tab[:, co_idx] = (250.0 - means[co_idx]) / stds[co_idx]
    
    with torch.no_grad():
        out3 = model(corrupted_tab.unsqueeze(0).to(device), sample_s01["image"].unsqueeze(0).to(device))
        dec3 = model.evaluate_predictive_mitigation(out3)

    diff3 = torch.abs(out3["tab_risk_now"] - out3["img_risk_now"]).item()
    print("\n  [CASE 3: Injected Sensor Glitch — Faulty CO Spike (250 ppm)]")
    print(f"  • Present Risk (Sensor Spike): {out3['tab_risk_now'].item():.1f}% | Thermal Image: {out3['img_risk_now'].item():.1f}%")
    print(f"  • Disagreement (|Δ|)        : {diff3:.1f}% (Threshold: 25.0%)")
    print(f"  • Decision Status           : {dec3['status']}")
    print(f"  • Controller Action         : {dec3['action']}")

    # 4. Injected Optical Glare (Sunlight Reflection on Normal Coal Pile)
    corrupted_img = sample_s05["image"].clone()
    with torch.no_grad():
        out4 = model(sample_s01["tabular"].unsqueeze(0).to(device), corrupted_img.unsqueeze(0).to(device))
        dec4 = model.evaluate_predictive_mitigation(out4)

    diff4 = torch.abs(out4["tab_risk_now"] - out4["img_risk_now"]).item()
    print("\n  [CASE 4: Injected Camera Glare — False Surface Thermal Hotspot]")
    print(f"  • Present Risk (Gas Sensor) : {out4['tab_risk_now'].item():.1f}% | Thermal Glare: {out4['img_risk_now'].item():.1f}%")
    print(f"  • Disagreement (|Δ|)        : {diff4:.1f}% (Threshold: 25.0%)")
    print(f"  • Decision Status           : {dec4['status']}")
    print(f"  • Controller Action         : {dec4['action']}")

    # Save evaluation metrics to JSON
    eval_results = {
        "test_scenario": "S07 (Late Mitigation - Held-out Unseen)",
        "forecast_horizon": "+1 Hour (60 minutes)",
        "future_forecast_metrics": {
            "future_mae_percent": round(float(mae_60m), 2),
            "future_rmse_percent": round(float(rmse_60m), 2),
            "future_r2_score": round(float(r2_60m), 4),
            "future_accuracy_percent": round(float(acc_60m), 1),
            "future_f1_percent": round(float(f1_60m), 1)
        },
        "benchmarks": {
            "case_1_normal_consensus": {
                "present_risk_percent": round(float(dec1['risk_now'].item()), 1),
                "forecast_risk_60m_percent": round(float(dec1['risk_forecast_60m'].item()), 1),
                "forecast_temp_core_60m": round(float(dec1['temp_core_forecast_60m'].item()), 1),
                "disagreement_percent": round(float(diff1), 1),
                "status": dec1['status'],
                "action": dec1['action']
            },
            "case_2_predictive_early_mitigation": {
                "present_risk_percent": round(float(dec2['risk_now'].item()), 1),
                "forecast_risk_60m_percent": round(float(dec2['risk_forecast_60m'].item()), 1),
                "forecast_temp_core_60m": round(float(dec2['temp_core_forecast_60m'].item()), 1),
                "disagreement_percent": round(float(diff2), 1),
                "status": dec2['status'],
                "action": dec2['action']
            },
            "case_3_injected_sensor_glitch": {
                "disagreement_percent": round(float(diff3), 1),
                "status": dec3['status'],
                "action": dec3['action']
            },
            "case_4_injected_camera_glare": {
                "disagreement_percent": round(float(diff4), 1),
                "status": dec4['status'],
                "action": dec4['action']
            }
        }
    }

    results_path = MODEL_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n  [SAVED] Future forecasting metrics saved to: {results_path}")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()
