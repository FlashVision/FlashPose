"""Heatmap-based prediction head for keypoint detection."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashpose.cfg.config import PoseConfig
from flashpose.registry import HEADS


@HEADS.register("HeatmapHead")
class HeatmapHead(nn.Module):
    """Deconvolution-based heatmap head for top-down pose estimation.

    Upsamples the backbone feature map and produces one heatmap per keypoint.
    Uses transposed convolutions for learnable upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 17,
        config: PoseConfig = None,
        num_deconv_layers: int = 2,
        num_deconv_filters: int = 256,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints

        target_h = config.heatmap_size[0] if config else 64
        target_w = config.heatmap_size[1] if config else 48
        self.target_size = (target_h, target_w)

        model_name = config.model_name if config else "ViTPose"
        if model_name == "HRNet":
            num_deconv_layers = 0

        layers = []
        current_ch = in_channels
        for i in range(num_deconv_layers):
            out_ch = num_deconv_filters
            layers.extend([
                nn.ConvTranspose2d(current_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            current_ch = out_ch

        self.deconv = nn.Sequential(*layers) if layers else nn.Identity()
        self.final_conv = nn.Conv2d(current_ch, num_keypoints, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.final_conv.weight, std=0.001)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Produce heatmaps from backbone features.

        Args:
            x: Feature map (B, C, H, W) from backbone.

        Returns:
            Dict with 'heatmaps' of shape (B, K, H_target, W_target).
        """
        x = self.deconv(x)
        heatmaps = self.final_conv(x)
        if heatmaps.shape[2:] != self.target_size:
            heatmaps = F.interpolate(heatmaps, size=self.target_size, mode="bilinear", align_corners=False)
        return {"heatmaps": heatmaps}

    @staticmethod
    def decode_heatmaps(heatmaps: torch.Tensor, input_size: tuple) -> torch.Tensor:
        """Decode heatmaps to keypoint coordinates using argmax.

        Args:
            heatmaps: (B, K, H, W) predicted heatmaps.
            input_size: (H, W) of the original input image.

        Returns:
            (B, K, 3) tensor with (x, y, confidence) per keypoint.
        """
        B, K, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.reshape(B, K, -1)
        max_vals, max_idx = heatmaps_flat.max(dim=-1)

        coords_y = (max_idx // W).float()
        coords_x = (max_idx % W).float()

        stride_x = input_size[1] / W
        stride_y = input_size[0] / H

        coords_x = coords_x * stride_x
        coords_y = coords_y * stride_y

        confidence = max_vals.sigmoid()

        return torch.stack([coords_x, coords_y, confidence], dim=-1)
