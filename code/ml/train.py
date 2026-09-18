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
    total_mae = 0.0
    correct_cls = 0
    total_samples = 0

    for batch in dataloader:
        x_tab = batch["tabular"].to(device)
        x_img = batch["image"].to(device)
        y_risk = batch["risk_score"].to(device)
        y_cls = batch["risk_class"].to(device)

        optimizer.zero_grad()

        out = model(x_tab, x_img)

        # Multi-task loss
        loss_fused = criterion_reg(out["fused_risk"], y_risk)
        loss_tab   = criterion_reg(out["tab_risk"], y_risk)
        loss_img   = criterion_reg(out["img_risk"], y_risk)
        loss_cls   = criterion_cls(out["class_logits"], y_cls)
        loss_cons  = torch.mean(torch.abs(out["tab_risk"] - out["img_risk"]))

        loss = loss_fused + 0.3 * loss_tab + 0.3 * loss_img + 0.4 * loss_cls + 0.1 * loss_cons

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        batch_size = x_tab.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.sum(torch.abs(out["fused_risk"] - y_risk)).item()
        
        preds_cls = torch.argmax(out["class_logits"], dim=1)
        correct_cls += torch.sum(preds_cls == y_cls).item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_acc = (correct_cls / total_samples) * 100.0
    return avg_loss, avg_mae, avg_acc

def evaluate(model, dataloader, criterion_reg, criterion_cls, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_sq_err = 0.0
    correct_cls = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x_tab = batch["tabular"].to(device)
            x_img = batch["image"].to(device)
            y_risk = batch["risk_score"].to(device)
            y_cls = batch["risk_class"].to(device)

            out = model(x_tab, x_img)

            loss_fused = criterion_reg(out["fused_risk"], y_risk)
            loss_tab   = criterion_reg(out["tab_risk"], y_risk)
            loss_img   = criterion_reg(out["img_risk"], y_risk)
            loss_cls   = criterion_cls(out["class_logits"], y_cls)

            loss = loss_fused + 0.3 * loss_tab + 0.3 * loss_img + 0.4 * loss_cls

            batch_size = x_tab.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.sum(torch.abs(out["fused_risk"] - y_risk)).item()
            total_sq_err += torch.sum((out["fused_risk"] - y_risk) ** 2).item()

            preds_cls = torch.argmax(out["class_logits"], dim=1)
            correct_cls += torch.sum(preds_cls == y_cls).item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = np.sqrt(total_sq_err / total_samples)
    avg_acc = (correct_cls / total_samples) * 100.0
    return avg_loss, avg_mae, avg_rmse, avg_acc

def main():
    print("=" * 70)
    print("  CoalSafe-Sim v2  |  Training Multimodal Time-Series Model (MMTS)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [DEVICE] Using compute backend: {device}")
    if torch.cuda.is_available():
        print(f"  [GPU]    {torch.cuda.get_device_name(0)}")

    # Load Data
    train_loader, val_loader, test_loader, (means, stds) = get_dataloaders(
        root_dir=ROOT,
        batch_size=16,
        window_size=6,
        sampling_step=10,
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
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "val_rmse": [], "val_acc": []}

    print("\n  Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mae, tr_acc = train_epoch(model, train_loader, optimizer, criterion_reg, criterion_cls, device)
        val_loss, val_mae, val_rmse, val_acc = evaluate(model, val_loader, criterion_reg, criterion_cls, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_acc"].append(val_acc)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "means": means,
                "stds": stds,
            }, MODEL_DIR / "coalsafe_mmts_best.pt")
            saved_str = " ★ [BEST CHECKPOINT SAVED]"
        else:
            saved_str = ""

        if epoch % 5 == 0 or epoch == 1 or saved_str:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {tr_loss:.4f} | Train MAE: {tr_mae:.2f}% | "
                  f"Val MAE: {val_mae:.2f}% | Val RMSE: {val_rmse:.2f}% | Val Acc: {val_acc:.1f}%{saved_str}")

    # Save training history
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  [SUCCESS] Training Completed! Best Validation MAE: {best_val_mae:.2f}%")
    print(f"  [MODEL]   Saved checkpoint to: {MODEL_DIR / 'coalsafe_mmts_best.pt'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
