"""Direct regression head for keypoint coordinate prediction."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from flashpose.cfg.config import PoseConfig
from flashpose.registry import HEADS


@HEADS.register("RegressionHead")
class RegressionHead(nn.Module):
    """Direct coordinate regression head.

    Directly regresses keypoint (x, y) coordinates from feature maps
    without intermediate heatmap representation. Faster but typically
    less accurate than heatmap-based approaches for 2D pose.
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 17,
        config: PoseConfig = None,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        input_h = config.input_size[0] if config else 256
        input_w = config.input_size[1] if config else 192

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_keypoints * 2),
        )

        self.input_size = (input_h, input_w)
        self._init_weights()

    def _init_weights(self):
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict keypoint coordinates directly.

        Args:
            x: Feature map (B, C, H, W) from backbone.

        Returns:
            Dict with 'keypoints' of shape (B, K, 2) in normalized [0, 1] coords.
        """
        B = x.shape[0]
        feat = self.pool(x)
        coords = self.regressor(feat)
        coords = coords.reshape(B, self.num_keypoints, 2)
        coords = torch.sigmoid(coords)
        return {"keypoints": coords}

    def decode(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Convert normalized coords to pixel coordinates.

        Args:
            keypoints: (B, K, 2) normalized coordinates in [0, 1].

        Returns:
            (B, K, 2) pixel coordinates.
        """
        h, w = self.input_size
        pixel_coords = keypoints.clone()
        pixel_coords[..., 0] *= w
        pixel_coords[..., 1] *= h
        return pixel_coords
