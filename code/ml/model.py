import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Self-attention layer across temporal history steps.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, rnn_out):
        # rnn_out: (B, W, hidden_dim)
        scores = self.attn(rnn_out)             # (B, W, 1)
        weights = F.softmax(scores, dim=1)      # (B, W, 1)
        context = torch.sum(weights * rnn_out, dim=1) # (B, hidden_dim)
        return context, weights

class TabularTemporalEncoder(nn.Module):
    """
    Bi-directional GRU Temporal Sequence Encoder with Attention for Multi-Sensor Telemetry.
    """
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = TemporalAttention(hidden_dim * 2)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        # x: (B, W, input_dim)
        out, _ = self.gru(x)                  # (B, W, hidden_dim * 2)
        context, weights = self.attn(out)     # (B, hidden_dim * 2)
        h_tab = self.proj(context)            # (B, 128)
        return h_tab, weights

class ThermalSpatialEncoder(nn.Module):
    """
    2D Convolutional Spatial Feature Extractor for 50x50 Thermal Heatmaps.
    """
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            # Block 1: 50x50 -> 25x25
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            # Block 2: 25x25 -> 12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            # Block 3: 12x12 -> 6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        features = self.conv_net(x)           # (B, 128, 1, 1)
        h_img = self.proj(features)           # (B, 128)
        return h_img

class CrossModalGatedFusion(nn.Module):
    """
    Gated Bilinear Attention Fusion uniting Tabular Temporal dynamics and Thermal Spatial features.
    """
    def __init__(self, dim_tab=128, dim_img=128, fused_dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_tab + dim_img, dim_tab),
            nn.Sigmoid()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim_tab * 3, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, h_tab, h_img):
        # Gating coefficient between modalities
        g = self.gate(torch.cat([h_tab, h_img], dim=-1)) # (B, 128)
        
        # Interaction representation
        interaction = h_tab * h_img
        gated_tab = g * h_tab
        gated_img = (1.0 - g) * h_img

        combined = torch.cat([gated_tab, gated_img, interaction], dim=-1) # (B, 384)
        h_fused = self.fusion_mlp(combined) # (B, 256)
        return h_fused, g

class MultimodalTimeSeriesNet(nn.Module):
    """
    CoalSafe-MMTS: Multimodal Time-Series Neural Network.
    
    Features:
      - Tabular Temporal Encoder (GRU + Attention)
      - Thermal Spatial Encoder (2D-CNN)
      - Gated Cross-Modal Fusion
      - Auxiliary Individual Modality Heads (Tabular Head, Thermal Head)
      - Joint Consensus Risk Regression Head & 5-Class Classification Head
      - Cross-Modal Conflict & Anomaly Detection Gate
    """
    def __init__(self, num_tab_features=10, num_classes=5, conflict_threshold=25.0):
        super().__init__()
        self.conflict_threshold = conflict_threshold

        # Encoders
        self.tab_encoder = TabularTemporalEncoder(input_dim=num_tab_features, hidden_dim=64)
        self.img_encoder = ThermalSpatialEncoder(in_channels=3, out_dim=128)
        self.fusion = CrossModalGatedFusion(dim_tab=128, dim_img=128, fused_dim=256)

        # Auxiliary Individual Modality Heads (for modality consistency & conflict checks)
        self.tab_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.img_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Joint Multimodal Heads
        self.fused_reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fused_class_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_tab, x_img):
        """
        Forward pass.
        x_tab: (B, W, D)
        x_img: (B, 3, H, W)
        """
        h_tab, tab_attn = self.tab_encoder(x_tab)
        h_img = self.img_encoder(x_img)

        # Auxiliary predictions (scaled to 0-100%)
        pred_tab_risk = self.tab_head(h_tab) * 100.0
        pred_img_risk = self.img_head(h_img) * 100.0

        # Joint Cross-Modal Fusion
        h_fused, gate_scores = self.fusion(h_tab, h_img)

        # Joint Predictions
        pred_fused_risk = self.fused_reg_head(h_fused) * 100.0
        class_logits = self.fused_class_head(h_fused)

        return {
            "fused_risk": pred_fused_risk.squeeze(-1),
            "tab_risk": pred_tab_risk.squeeze(-1),
            "img_risk": pred_img_risk.squeeze(-1),
            "class_logits": class_logits,
            "gate_scores": gate_scores,
            "tab_attn": tab_attn,
        }

    def evaluate_consensus(self, out_dict):
        """
        Evaluates cross-modal agreement and flags conflicts.
        Returns:
          - is_valid: Boolean mask (True = Consensus Valid, False = Conflict Detected)
          - disagreement: Absolute difference between tabular and thermal predictions
          - conflict_status: Explanatory string
        """
        tab_r = out_dict["tab_risk"]
        img_r = out_dict["img_risk"]
        fused_r = out_dict["fused_risk"]

        disagreement = torch.abs(tab_r - img_r)
        is_valid = disagreement <= self.conflict_threshold

        return {
            "is_valid": is_valid,
            "disagreement": disagreement,
            "fused_risk": fused_r,
            "tab_risk": tab_r,
            "img_risk": img_r,
        }
