"""RTMPose — Real-Time Multi-Person Pose Estimation backbone.

Reference: "RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose"
           (Jiang et al., arXiv 2023)

Uses a CSPDarknet-style backbone with depthwise separable convolutions
for efficient real-time inference.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashpose.cfg.config import PoseConfig
from flashpose.registry import MODELS


class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU activation block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 1):
        super().__init__()
        self.depthwise = ConvBNSiLU(in_ch, in_ch, kernel, stride, groups=in_ch)
        self.pointwise = ConvBNSiLU(in_ch, out_ch, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class CSPBlock(nn.Module):
    """Cross Stage Partial block with depthwise separable convolutions."""

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 3, expand_ratio: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * expand_ratio)

        self.main_conv = ConvBNSiLU(in_ch, hidden_ch, kernel=1)
        self.short_conv = ConvBNSiLU(in_ch, hidden_ch, kernel=1)

        self.blocks = nn.Sequential(*[
            DepthwiseSeparable(hidden_ch, hidden_ch, kernel=5) for _ in range(num_blocks)
        ])

        self.final_conv = ConvBNSiLU(hidden_ch * 2, out_ch, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.short_conv(x)
        main = self.blocks(self.main_conv(x))
        return self.final_conv(torch.cat([main, short], dim=1))


class RTMPoseBackbone(nn.Module):
    """CSPDarknet-style backbone for RTMPose."""

    def __init__(self, channels: List[int], depths: List[int]):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNSiLU(3, channels[0], kernel=3, stride=2),
            ConvBNSiLU(channels[0], channels[0], kernel=3, stride=1),
        )

        stages = []
        in_ch = channels[0]
        for ch, depth in zip(channels[1:], depths):
            stages.append(nn.Sequential(
                ConvBNSiLU(in_ch, ch, kernel=3, stride=2),
                CSPBlock(ch, ch, num_blocks=depth),
            ))
            in_ch = ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class GatedAttentionUnit(nn.Module):
    """Gated Attention Unit for feature refinement in RTMPose."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.act = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gap = x.mean(dim=[2, 3])
        gate = self.sigmoid(self.fc2(self.act(self.fc1(gap))))
        return x * gate.view(B, C, 1, 1)


@MODELS.register("RTMPose")
class RTMPose(nn.Module):
    """RTMPose backbone for real-time pose estimation.

    Lightweight architecture based on CSPDarknet with depthwise separable
    convolutions and gated attention for efficient inference.

    Variants:
        - rtmpose_t: channels=[32, 64, 128, 256], depths=[1, 2, 2, 1]
        - rtmpose_s: channels=[32, 64, 128, 256], depths=[2, 4, 4, 2]
        - rtmpose_m: channels=[48, 96, 192, 384], depths=[2, 4, 4, 2]
        - rtmpose_l: channels=[64, 128, 256, 512], depths=[3, 6, 6, 3]
    """

    CONFIGS = {
        "rtmpose_t": {"channels": [32, 64, 128, 256], "depths": [1, 2, 2, 1]},
        "rtmpose_s": {"channels": [32, 64, 128, 256], "depths": [2, 4, 4, 2]},
        "rtmpose_m": {"channels": [48, 96, 192, 384], "depths": [2, 4, 4, 2]},
        "rtmpose_l": {"channels": [64, 128, 256, 512], "depths": [3, 6, 6, 3]},
    }

    def __init__(self, config: PoseConfig):
        super().__init__()
        backbone_name = config.backbone if config.backbone in self.CONFIGS else "rtmpose_s"
        cfg = self.CONFIGS[backbone_name]

        channels = cfg["channels"]
        depths = cfg["depths"]

        self.backbone = RTMPoseBackbone(channels, depths)
        self.gau = GatedAttentionUnit(channels[-1])

        self.upsample = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-2], 1, bias=False),
            nn.BatchNorm2d(channels[-2]),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels[-2], channels[-2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-2]),
            nn.SiLU(inplace=True),
        )

        self._out_channels = channels[-2]

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning feature map at 1/16 resolution.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Feature map (B, C, H/16, W/16).
        """
        feat = self.backbone(x)
        feat = self.gau(feat)
        feat = self.upsample(feat)
        return feat
