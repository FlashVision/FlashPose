"""SimCC — Simple Coordinate Classification head for keypoint estimation.

Reference: "SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation"
           (Li et al., ECCV 2022)

Instead of predicting heatmaps, SimCC classifies the x and y coordinates
independently into discrete bins, which is simpler and often more accurate.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashpose.cfg.config import PoseConfig


class SimCCHead(nn.Module):
    """SimCC coordinate classification head.

    Predicts x and y coordinate distributions independently for each keypoint
    as 1D classification problems (instead of 2D heatmap regression).

    The input feature map is pooled and projected to produce per-keypoint
    logits over discretized x and y bins.
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 17,
        config: PoseConfig = None,
        simcc_split_ratio: float = 2.0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints

        if config is not None:
            input_h, input_w = config.input_size
        else:
            input_h, input_w = 256, 192

        self.x_size = int(input_w * simcc_split_ratio)
        self.y_size = int(input_h * simcc_split_ratio)

        hidden_dim = max(in_channels // 2, 256)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.fc_x = nn.Linear(hidden_dim, num_keypoints * self.x_size)
        self.fc_y = nn.Linear(hidden_dim, num_keypoints * self.y_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass producing x and y coordinate distributions.

        Args:
            x: Feature map (B, C, H, W) from backbone.

        Returns:
            Dict with 'simcc_x' (B, K, W*ratio) and 'simcc_y' (B, K, H*ratio).
        """
        B = x.shape[0]
        feat = self.mlp(x)

        pred_x = self.fc_x(feat).reshape(B, self.num_keypoints, self.x_size)
        pred_y = self.fc_y(feat).reshape(B, self.num_keypoints, self.y_size)

        return {
            "simcc_x": pred_x,
            "simcc_y": pred_y,
        }

    def decode(self, simcc_x: torch.Tensor, simcc_y: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        """Decode SimCC predictions to keypoint coordinates.

        Args:
            simcc_x: (B, K, x_size) x-coordinate logits.
            simcc_y: (B, K, y_size) y-coordinate logits.
            input_size: (H, W) of the input image.

        Returns:
            (B, K, 2) keypoint coordinates in pixel space.
        """
        x_locs = simcc_x.argmax(dim=-1).float()
        y_locs = simcc_y.argmax(dim=-1).float()

        input_h, input_w = input_size
        x_locs = x_locs / self.x_size * input_w
        y_locs = y_locs / self.y_size * input_h

        return torch.stack([x_locs, y_locs], dim=-1)
