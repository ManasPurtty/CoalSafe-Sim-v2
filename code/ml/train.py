import os
import sys
import io
import json
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dataset import get_dataloaders
from model import MultimodalTimeSeriesNet

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_epoch(model, dataloader, optimizer, criterion_reg, criterion_cls, device):
    model.train()
    total_loss = 0.0
    total_mae_now = 0.0
    total_mae_60m = 0.0
    correct_cls_60m = 0
    total_samples = 0

    for batch in dataloader:
        x_tab = batch["tabular"].to(device)
        x_img = batch["image"].to(device)
        
        y_risk_now = batch["risk_score_now"].to(device)
        y_risk_30m = batch["risk_score_30m"].to(device)
        y_risk_60m = batch["risk_score_60m"].to(device)
        y_temp_60m = batch["temp_core_60m"].to(device)
        y_cls_now  = batch["risk_class_now"].to(device)
        y_cls_60m  = batch["risk_class_60m"].to(device)

        optimizer.zero_grad()

        out = model(x_tab, x_img)

        # Multi-task & Multi-horizon future forecasting loss
        loss_now_fused   = criterion_reg(out["fused_risk_now"], y_risk_now)
        loss_now_tab     = criterion_reg(out["tab_risk_now"], y_risk_now)
        loss_now_img     = criterion_reg(out["img_risk_now"], y_risk_now)
        loss_cls_now     = criterion_cls(out["class_logits_now"], y_cls_now)
        
        # Future Forecast Losses (+30m and +60m / +1 Hour Ahead)
        loss_fc_30m      = criterion_reg(out["risk_forecast_30m"], y_risk_30m)
        loss_fc_60m      = criterion_reg(out["risk_forecast_60m"], y_risk_60m)
        loss_temp_60m    = criterion_reg(out["temp_core_forecast_60m"], y_temp_60m)
        loss_cls_60m     = criterion_cls(out["class_logits_60m"], y_cls_60m)

        # Modality consistency regularizer
        loss_cons        = torch.mean(torch.abs(out["tab_risk_now"] - out["img_risk_now"]))

        loss = (
            loss_now_fused
            + 0.2 * loss_now_tab
            + 0.2 * loss_now_img
            + 0.3 * loss_cls_now
            + 0.3 * loss_fc_30m
            + 0.6 * loss_fc_60m       # High weight on +1h future forecasting accuracy!
            + 0.1 * loss_temp_60m
            + 0.3 * loss_cls_60m
            + 0.1 * loss_cons
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        batch_size = x_tab.size(0)
        total_loss += loss.item() * batch_size
        total_mae_now += torch.sum(torch.abs(out["fused_risk_now"] - y_risk_now)).item()
        total_mae_60m += torch.sum(torch.abs(out["risk_forecast_60m"] - y_risk_60m)).item()
        
        preds_cls_60m = torch.argmax(out["class_logits_60m"], dim=1)
        correct_cls_60m += torch.sum(preds_cls_60m == y_cls_60m).item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae_now = total_mae_now / total_samples
    avg_mae_60m = total_mae_60m / total_samples
    avg_acc_60m = (correct_cls_60m / total_samples) * 100.0
    return avg_loss, avg_mae_now, avg_mae_60m, avg_acc_60m

def evaluate(model, dataloader, criterion_reg, criterion_cls, device):
    model.eval()
    total_loss = 0.0
    total_mae_now = 0.0
    total_mae_60m = 0.0
    total_sq_err_60m = 0.0
    correct_cls_60m = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x_tab = batch["tabular"].to(device)
            x_img = batch["image"].to(device)
            
            y_risk_now = batch["risk_score_now"].to(device)
            y_risk_30m = batch["risk_score_30m"].to(device)
            y_risk_60m = batch["risk_score_60m"].to(device)
            y_temp_60m = batch["temp_core_60m"].to(device)
            y_cls_now  = batch["risk_class_now"].to(device)
            y_cls_60m  = batch["risk_class_60m"].to(device)

            out = model(x_tab, x_img)

            loss_now_fused = criterion_reg(out["fused_risk_now"], y_risk_now)
            loss_fc_60m    = criterion_reg(out["risk_forecast_60m"], y_risk_60m)
            loss_cls_60m   = criterion_cls(out["class_logits_60m"], y_cls_60m)

            loss = loss_now_fused + 0.6 * loss_fc_60m + 0.3 * loss_cls_60m

            batch_size = x_tab.size(0)
            total_loss += loss.item() * batch_size
            total_mae_now += torch.sum(torch.abs(out["fused_risk_now"] - y_risk_now)).item()
            total_mae_60m += torch.sum(torch.abs(out["risk_forecast_60m"] - y_risk_60m)).item()
            total_sq_err_60m += torch.sum((out["risk_forecast_60m"] - y_risk_60m) ** 2).item()

            preds_cls_60m = torch.argmax(out["class_logits_60m"], dim=1)
            correct_cls_60m += torch.sum(preds_cls_60m == y_cls_60m).item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae_now = total_mae_now / total_samples
    avg_mae_60m = total_mae_60m / total_samples
    avg_rmse_60m = np.sqrt(total_sq_err_60m / total_samples)
    avg_acc_60m = (correct_cls_60m / total_samples) * 100.0
    return avg_loss, avg_mae_now, avg_mae_60m, avg_rmse_60m, avg_acc_60m

def main():
    print("=" * 75)
    print("  CoalSafe-Sim v2  |  Training Multimodal Future Forecaster (+1h Horizon)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [DEVICE] Using compute backend: {device}")
    if torch.cuda.is_available():
        print(f"  [GPU]    {torch.cuda.get_device_name(0)}")

    # Load Data with +1h future forecasting targets
    train_loader, val_loader, test_loader, (means, stds) = get_dataloaders(
        root_dir=ROOT,
        batch_size=16,
        window_size=6,
        sampling_step=10,
        forecast_horizon_min=60,
        train_scenarios=("S01", "S02", "S03", "S05"),
        val_scenarios=("S04", "S06", "S08"),
        test_scenarios=("S07",),
    )

    print(f"  [DATA]   Train Samples: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Initialize Model
    model = MultimodalTimeSeriesNet(num_tab_features=10, num_classes=5, conflict_threshold=25.0).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [MODEL]  Trainable Parameters: {num_params:,}")

    criterion_reg = nn.SmoothL1Loss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 40
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mae = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_mae_now": [], "val_mae_now": [],
        "train_mae_60m": [], "val_mae_60m": [],
        "val_rmse_60m": [], "val_acc_60m": []
    }

    print("\n  Starting Training Loop with +1h Future Forecasting...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mae_now, tr_mae_60m, tr_acc_60m = train_epoch(model, train_loader, optimizer, criterion_reg, criterion_cls, device)
        val_loss, val_mae_now, val_mae_60m, val_rmse_60m, val_acc_60m = evaluate(model, val_loader, criterion_reg, criterion_cls, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_mae_now"].append(tr_mae_now)
        history["val_mae_now"].append(val_mae_now)
        history["train_mae_60m"].append(tr_mae_60m)
        history["val_mae_60m"].append(val_mae_60m)
        history["val_rmse_60m"].append(val_rmse_60m)
        history["val_acc_60m"].append(val_acc_60m)

        # Checkpoint based on future +1h forecasting accuracy
        if val_mae_60m < best_val_mae:
            best_val_mae = val_mae_60m
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae_60m": val_mae_60m,
                "val_mae_now": val_mae_now,
                "means": means,
                "stds": stds,
            }, MODEL_DIR / "coalsafe_mmts_best.pt")
            saved_str = " ★ [BEST +1h FORECASTER CHECKPOINT SAVED]"
        else:
            saved_str = ""

        if epoch % 5 == 0 or epoch == 1 or saved_str:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {tr_loss:.4f} | "
                  f"Now MAE: {val_mae_now:.2f}% | "
                  f"+1h Forecast MAE: {val_mae_60m:.2f}% | "
                  f"+1h RMSE: {val_rmse_60m:.2f}% | "
                  f"+1h Acc: {val_acc_60m:.1f}%{saved_str}")

    # Save training history
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 75)
    print(f"  [SUCCESS] Training Completed! Best +1h Future Forecast MAE: {best_val_mae:.2f}%")
    print(f"  [MODEL]   Saved checkpoint to: {MODEL_DIR / 'coalsafe_mmts_best.pt'}")
    print("=" * 75)

if __name__ == "__main__":
    main()
