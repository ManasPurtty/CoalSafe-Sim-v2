import os
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

TABULAR_FEATURES = [
    "ambient_temperature",
    "humidity",
    "wind_speed",
    "coal_moisture",
    "CO",
    "CO2",
    "O2",
    "surface_temperature_mean",
    "surface_temperature_max",
    "heating_rate",
]

RISK_CLASS_MAP = {
    "NORMAL": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

class MultimodalTimeSeriesDataset(Dataset):
    """
    Multimodal Time-Series Dataset with Multi-Horizon Future Forecasting (+1h Horizon).
    
    Extracts:
      - X_tab: Sliding window history of tabular sensor channels (W, D) from (t-60m to t)
      - X_img: Spatial thermal heatmap frame at current time t (3, 50, 50)
      - y_risk_now: Current risk score at time t
      - y_risk_30m: Future forecasted risk score at t + 30 min
      - y_risk_60m: Future forecasted risk score at t + 60 min (+1 Hour ahead!)
      - y_temp_core_60m: Future internal core temperature at t + 60 min
      - y_class_now: Current risk classification (0-4)
      - y_class_60m: Future risk classification at t + 60 min (0-4)
    """
    def __init__(
        self,
        root_dir,
        scenarios=None,
        window_size=6,
        sampling_step=10,
        forecast_horizon_min=60,
        feature_means=None,
        feature_stds=None,
        transform=None,
    ):
        self.root_dir = pathlib.Path(root_dir)
        self.window_size = window_size
        self.sampling_step = sampling_step
        self.forecast_horizon_min = forecast_horizon_min
        self.transform = transform or transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load CSV tables
        sensor_path = self.root_dir / "data" / "tabular" / "sensor_data.csv"
        thermal_path = self.root_dir / "data" / "metadata" / "thermal_metadata.csv"
        latent_path = self.root_dir / "data" / "metadata" / "latent_state.csv"

        df_sensor = pd.read_csv(sensor_path)
        df_thermal = pd.read_csv(thermal_path)
        df_latent = pd.read_csv(latent_path) if latent_path.exists() else pd.DataFrame()

        # Merge tabular sensors with thermal metadata
        df = pd.merge(
            df_sensor,
            df_thermal[["scenario_id", "timestamp_min", "image_path"]],
            on=["scenario_id", "timestamp_min"],
            how="left"
        )

        # Merge latent internal core temperature for physics-informed future temperature forecasting
        if not df_latent.empty:
            df = pd.merge(
                df,
                df_latent[["scenario_id", "timestamp_min", "internal_temperature", "deep_core_temperature"]],
                on=["scenario_id", "timestamp_min"],
                how="left"
            )
        else:
            df["internal_temperature"] = df["surface_temperature_mean"]
            df["deep_core_temperature"] = df["surface_temperature_mean"]

        if scenarios is not None:
            df = df[df["scenario_id"].isin(scenarios)].copy()

        # Handle missing values across time-series per scenario
        cleaned_dfs = []
        for sid, group in df.groupby("scenario_id"):
            group = group.sort_values("timestamp_min").copy()
            # Forward-fill then backward-fill missing sensor values
            for col in TABULAR_FEATURES + ["internal_temperature", "deep_core_temperature"]:
                if col in group.columns:
                    group[col] = group[col].ffill().bfill().fillna(0.0)
            cleaned_dfs.append(group)

        self.df = pd.concat(cleaned_dfs, ignore_index=True)

        # Compute or set normalization parameters
        if feature_means is None or feature_stds is None:
            self.feature_means = self.df[TABULAR_FEATURES].mean().values.astype(np.float32)
            self.feature_stds = self.df[TABULAR_FEATURES].std().values.astype(np.float32)
            # Avoid division by zero
            self.feature_stds[self.feature_stds < 1e-6] = 1.0
        else:
            self.feature_means = np.array(feature_means, dtype=np.float32)
            self.feature_stds = np.array(feature_stds, dtype=np.float32)

        # Create sliding window sample indices mapped to future timestamps
        self.samples = []
        for sid, group in self.df.groupby("scenario_id"):
            group = group.sort_values("timestamp_min").reset_index(drop=True)
            # Map timestamps to row indices for O(1) future lookups
            t_to_idx = {int(row["timestamp_min"]): idx for idx, row in group.iterrows()}
            sampled_indices = group[group["timestamp_min"] % self.sampling_step == 0].index.tolist()
            
            for idx in sampled_indices:
                pos_in_sampled = sampled_indices.index(idx)
                if pos_in_sampled >= self.window_size - 1:
                    window_df_indices = sampled_indices[pos_in_sampled - self.window_size + 1 : pos_in_sampled + 1]
                    
                    t_curr = int(group.iloc[idx]["timestamp_min"])
                    t_30m = min(1440, t_curr + 30)
                    t_60m = min(1440, t_curr + 60)
                    
                    idx_30m = t_to_idx.get(t_30m, len(group) - 1)
                    idx_60m = t_to_idx.get(t_60m, len(group) - 1)

                    self.samples.append((group, window_df_indices, idx, idx_30m, idx_60m))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        group, window_indices, current_idx, idx_30m, idx_60m = self.samples[idx]
        
        # 1. Tabular sequence history window (W, D)
        window_rows = group.iloc[window_indices]
        raw_tab = window_rows[TABULAR_FEATURES].values.astype(np.float32)
        norm_tab = (raw_tab - self.feature_means) / self.feature_stds
        tensor_tab = torch.tensor(norm_tab, dtype=torch.float32)

        # 2. Thermal image at current time t
        curr_row = group.iloc[current_idx]
        img_rel_path = curr_row["image_path"]
        
        img_full_path = self.root_dir / img_rel_path if pd.notna(img_rel_path) and img_rel_path != "MISSING" else None
        
        if img_full_path and img_full_path.exists():
            try:
                img = Image.open(img_full_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (50, 50), color=(30, 30, 30))
        else:
            img = Image.new("RGB", (50, 50), color=(30, 30, 30))

        tensor_img = self.transform(img)

        # 3. Present and Multi-Horizon Future Ground Truth Targets
        row_30m = group.iloc[idx_30m]
        row_60m = group.iloc[idx_60m]

        risk_now = float(curr_row["risk_score"])
        risk_30m = float(row_30m["risk_score"])
        risk_60m = float(row_60m["risk_score"]) # +1 Hour Forecast Target

        temp_core_now = float(curr_row.get("deep_core_temperature", curr_row.get("internal_temperature", 30.0)))
        temp_core_60m = float(row_60m.get("deep_core_temperature", row_60m.get("internal_temperature", 30.0)))

        class_now = RISK_CLASS_MAP.get(curr_row["risk_class"], 0)
        class_60m = RISK_CLASS_MAP.get(row_60m["risk_class"], 0)
        
        return {
            "tabular": tensor_tab,                                             # (W, D)
            "image": tensor_img,                                               # (3, 50, 50)
            "risk_score_now": torch.tensor(risk_now, dtype=torch.float32),     # scalar (0-100)
            "risk_score_30m": torch.tensor(risk_30m, dtype=torch.float32),     # scalar (0-100)
            "risk_score_60m": torch.tensor(risk_60m, dtype=torch.float32),     # scalar (0-100) -> +1h Horizon Target!
            "temp_core_60m": torch.tensor(temp_core_60m, dtype=torch.float32), # scalar (°C)
            "risk_class_now": torch.tensor(class_now, dtype=torch.long),
            "risk_class_60m": torch.tensor(class_60m, dtype=torch.long),
            "scenario_id": curr_row["scenario_id"],
            "timestamp_min": int(curr_row["timestamp_min"]),
        }

def get_dataloaders(
    root_dir,
    batch_size=16,
    window_size=6,
    sampling_step=10,
    forecast_horizon_min=60,
    train_scenarios=("S01", "S02", "S03", "S05"),
    val_scenarios=("S04", "S06", "S08"),
    test_scenarios=("S07",),
):
    """
    Creates train, validation, and test PyTorch DataLoaders with +1h future forecasting targets.
    """
    train_dataset = MultimodalTimeSeriesDataset(
        root_dir=root_dir,
        scenarios=train_scenarios,
        window_size=window_size,
        sampling_step=sampling_step,
        forecast_horizon_min=forecast_horizon_min,
    )

    means = train_dataset.feature_means
    stds = train_dataset.feature_stds

    val_dataset = MultimodalTimeSeriesDataset(
        root_dir=root_dir,
        scenarios=val_scenarios,
        window_size=window_size,
        sampling_step=sampling_step,
        forecast_horizon_min=forecast_horizon_min,
        feature_means=means,
        feature_stds=stds,
    )

    test_dataset = MultimodalTimeSeriesDataset(
        root_dir=root_dir,
        scenarios=test_scenarios,
        window_size=window_size,
        sampling_step=sampling_step,
        forecast_horizon_min=forecast_horizon_min,
        feature_means=means,
        feature_stds=stds,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, (means, stds)
