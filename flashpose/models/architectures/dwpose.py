"""DWPose — Distillation-based Whole-body Pose Estimation.

DWPose = RTMPose + knowledge distillation, achieving state-of-the-art
whole-body pose estimation with a lightweight architecture.

Implements:
- Dual-stage distillation (feature-level + logit-level)
- Head-aware attention for multi-granularity body parts
- Compatible with RTMPose backbone variants

Reference: "Effective Whole-body Pose Estimation with Two-stages
Distillation" (Yang et al., ICCV 2023 Workshop)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashpose.cfg.config import PoseConfig
from flashpose.registry import MODELS


class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5):
        super().__init__()
        self.dw = ConvBNSiLU(in_ch, in_ch, kernel, groups=in_ch)
        self.pw = ConvBNSiLU(in_ch, out_ch, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class CSPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 3):
        super().__init__()
        hidden = out_ch // 2
        self.main = ConvBNSiLU(in_ch, hidden, 1)
        self.short = ConvBNSiLU(in_ch, hidden, 1)
        self.blocks = nn.Sequential(*[DepthwiseSeparable(hidden, hidden) for _ in range(num_blocks)])
        self.final = ConvBNSiLU(hidden * 2, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(torch.cat([self.blocks(self.main(x)), self.short(x)], dim=1))


class HeadAwareAttention(nn.Module):
    """Attention module that learns body-part-specific feature weighting."""

    def __init__(self, channels: int, num_parts: int = 5):
        super().__init__()
        self.num_parts = num_parts
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(channels // 4, channels * num_parts),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, num_parts, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        ch_weights = self.channel_attn(x).reshape(B, self.num_parts, C, 1, 1)
        sp_weights = self.spatial_attn(x)

        part_features = []
        for i in range(self.num_parts):
            feat = x * ch_weights[:, i] * sp_weights[:, i:i+1]
            part_features.append(feat)
        return part_features


class DWPoseBackbone(nn.Module):
    """Lightweight CSPDarknet backbone for DWPose."""

    def __init__(self, channels: List[int], depths: List[int]):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNSiLU(3, channels[0], 3, 2),
            ConvBNSiLU(channels[0], channels[0], 3, 1),
        )
        stages = []
        in_ch = channels[0]
        for ch, depth in zip(channels[1:], depths):
            stages.append(nn.Sequential(
                ConvBNSiLU(in_ch, ch, 3, 2),
                CSPBlock(ch, ch, depth),
            ))
            in_ch = ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class DistillationHead(nn.Module):
    """Prediction head with distillation adaptor layers."""

    def __init__(self, in_channels: int, num_keypoints: int, input_size: Tuple[int, int]):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.input_size = input_size

        self.feature_adaptor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        simcc_x_size = input_size[1] * 2
        simcc_y_size = input_size[0] * 2

        self.fc_x = nn.Linear(in_channels, num_keypoints * simcc_x_size)
        self.fc_y = nn.Linear(in_channels, num_keypoints * simcc_y_size)
        self.simcc_x_size = simcc_x_size
        self.simcc_y_size = simcc_y_size

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.feature_adaptor(x)
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)

        simcc_x = self.fc_x(pooled).reshape(-1, self.num_keypoints, self.simcc_x_size)
        simcc_y = self.fc_y(pooled).reshape(-1, self.num_keypoints, self.simcc_y_size)

        return {"simcc_x": simcc_x, "simcc_y": simcc_y, "features": pooled}


@MODELS.register("DWPose")
class DWPose(nn.Module):
    """DWPose: Distillation-based Whole-body Pose Estimation.

    Combines RTMPose-style backbone with head-aware attention and
    a distillation-ready prediction head for whole-body pose.

    Supports distillation from a larger teacher model during training.

    Variants:
        - dwpose_t: channels=[32, 64, 128, 256], depths=[1, 2, 2, 1]
        - dwpose_s: channels=[32, 64, 128, 256], depths=[2, 4, 4, 2]
        - dwpose_m: channels=[48, 96, 192, 384], depths=[2, 4, 4, 2]
        - dwpose_l: channels=[64, 128, 256, 512], depths=[3, 6, 6, 3]
    """

    CONFIGS = {
        "dwpose_t": {"channels": [32, 64, 128, 256], "depths": [1, 2, 2, 1]},
        "dwpose_s": {"channels": [32, 64, 128, 256], "depths": [2, 4, 4, 2]},
        "dwpose_m": {"channels": [48, 96, 192, 384], "depths": [2, 4, 4, 2]},
        "dwpose_l": {"channels": [64, 128, 256, 512], "depths": [3, 6, 6, 3]},
    }

    def __init__(self, config: PoseConfig):
        super().__init__()
        backbone_name = config.backbone if config.backbone in self.CONFIGS else "dwpose_s"
        cfg = self.CONFIGS[backbone_name]

        channels = cfg["channels"]
        depths = cfg["depths"]

        self.backbone = DWPoseBackbone(channels, depths)
        self.head_attn = HeadAwareAttention(channels[-1], num_parts=5)

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
        self.pred_head = DistillationHead(channels[-2], config.num_keypoints, config.input_size)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        part_feats = self.head_attn(feat)
        feat_combined = sum(part_feats)
        feat_up = self.upsample(feat_combined)
        return self.pred_head(feat_up)

    def compute_distillation_loss(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Dict[str, torch.Tensor],
        temperature: float = 4.0,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Compute KD loss between student and teacher predictions.

        Args:
            student_output: Student model's forward output.
            teacher_output: Teacher model's forward output.
            temperature: Softmax temperature for logit distillation.
            alpha: Balance between feature-level and logit-level loss.

        Returns:
            Combined distillation loss.
        """
        s_x = student_output["simcc_x"] / temperature
        t_x = teacher_output["simcc_x"] / temperature
        s_y = student_output["simcc_y"] / temperature
        t_y = teacher_output["simcc_y"] / temperature

        kd_loss_x = F.kl_div(F.log_softmax(s_x, dim=-1), F.softmax(t_x, dim=-1), reduction="batchmean")
        kd_loss_y = F.kl_div(F.log_softmax(s_y, dim=-1), F.softmax(t_y, dim=-1), reduction="batchmean")
        logit_loss = (kd_loss_x + kd_loss_y) * (temperature ** 2)

        feat_loss = F.mse_loss(student_output["features"], teacher_output["features"])

        return alpha * logit_loss + (1 - alpha) * feat_loss
