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
    Multimodal Time-Series Dataset for Coal Spontaneous Combustion Early Warning.
    
    Extracts:
      - X_tab: Sliding window history of tabular sensor channels (B, W, D)
      - X_img: Spatial thermal heatmap frame at current time (B, C, H, W)
      - y_risk: Continuous risk score in [0, 100]
      - y_class: Discrete risk category in {0, 1, 2, 3, 4}
    """
    def __init__(
        self,
        root_dir,
        scenarios=None,
        window_size=6,
        sampling_step=10,
        feature_means=None,
        feature_stds=None,
        transform=None,
    ):
        self.root_dir = pathlib.Path(root_dir)
        self.window_size = window_size
        self.sampling_step = sampling_step
        self.transform = transform or transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load CSV tables
        sensor_path = self.root_dir / "data" / "tabular" / "sensor_data.csv"
        thermal_path = self.root_dir / "data" / "metadata" / "thermal_metadata.csv"

        df_sensor = pd.read_csv(sensor_path)
        df_thermal = pd.read_csv(thermal_path)

        # Merge on scenario_id and timestamp_min
        df = pd.merge(
            df_sensor,
            df_thermal[["scenario_id", "timestamp_min", "image_path"]],
            on=["scenario_id", "timestamp_min"],
            how="left"
        )

        if scenarios is not None:
            df = df[df["scenario_id"].isin(scenarios)].copy()

        # Handle missing values across time-series per scenario
        cleaned_dfs = []
        for sid, group in df.groupby("scenario_id"):
            group = group.sort_values("timestamp_min").copy()
            # Forward-fill then backward-fill missing sensor values
            for col in TABULAR_FEATURES:
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

        # Create sliding window sample indices
        self.samples = []
        for sid, group in self.df.groupby("scenario_id"):
            group = group.sort_values("timestamp_min").reset_index(drop=True)
            # Sample only at sampling_step intervals (where thermal images exist)
            sampled_indices = group[group["timestamp_min"] % self.sampling_step == 0].index.tolist()
            
            for idx in sampled_indices:
                # Need at least window_size past sampled points
                pos_in_sampled = sampled_indices.index(idx)
                if pos_in_sampled >= self.window_size - 1:
                    window_df_indices = sampled_indices[pos_in_sampled - self.window_size + 1 : pos_in_sampled + 1]
                    self.samples.append((group, window_df_indices, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        group, window_indices, current_idx = self.samples[idx]
        
        # 1. Tabular sequence window (W, D)
        window_rows = group.iloc[window_indices]
        raw_tab = window_rows[TABULAR_FEATURES].values.astype(np.float32)
        norm_tab = (raw_tab - self.feature_means) / self.feature_stds
        tensor_tab = torch.tensor(norm_tab, dtype=torch.float32)

        # 2. Thermal image at current time
        curr_row = group.iloc[current_idx]
        img_rel_path = curr_row["image_path"]
        
        img_full_path = self.root_dir / img_rel_path if pd.notna(img_rel_path) and img_rel_path != "MISSING" else None
        
        if img_full_path and img_full_path.exists():
            try:
                img = Image.open(img_full_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (50, 50), color=(30, 30, 30))
        else:
            # Fallback for missing images
            img = Image.new("RGB", (50, 50), color=(30, 30, 30))

        tensor_img = self.transform(img)

        # 3. Targets
        risk_score = float(curr_row["risk_score"])
        risk_class = RISK_CLASS_MAP.get(curr_row["risk_class"], 0)
        
        return {
            "tabular": tensor_tab,                          # (W, D)
            "image": tensor_img,                            # (3, 50, 50)
            "risk_score": torch.tensor(risk_score, dtype=torch.float32), # scalar
            "risk_class": torch.tensor(risk_class, dtype=torch.long),    # scalar
            "scenario_id": curr_row["scenario_id"],
            "timestamp_min": int(curr_row["timestamp_min"]),
        }

def get_dataloaders(
    root_dir,
    batch_size=16,
    window_size=6,
    sampling_step=10,
    train_scenarios=("S01", "S02", "S03", "S05"),
    val_scenarios=("S04", "S06", "S08"),
    test_scenarios=("S07",),
):
    """
    Creates train, validation, and test PyTorch DataLoaders.
    """
    train_dataset = MultimodalTimeSeriesDataset(
        root_dir=root_dir,
        scenarios=train_scenarios,
        window_size=window_size,
        sampling_step=sampling_step,
    )

    means = train_dataset.feature_means
    stds = train_dataset.feature_stds

    val_dataset = MultimodalTimeSeriesDataset(
        root_dir=root_dir,
        scenarios=val_scenarios,
        window_size=window_size,
        sampling_step=sampling_step,
        feature_means=means,
        feature_stds=stds,
    )

    test_dataset = MultimodalTimeSeriesDataset(
        root_dir=root_dir,
        scenarios=test_scenarios,
        window_size=window_size,
        sampling_step=sampling_step,
        feature_means=means,
        feature_stds=stds,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, (means, stds)
