"""DensePose — Continuous surface embedding for dense body pose estimation.

Maps every pixel of a detected person to a point on a 3D body surface
(UV coordinates on body parts), enabling dense correspondence.

Reference: "DensePose: Dense Human Pose Estimation In The Wild"
           (Güler et al., CVPR 2018)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashpose.registry import MODELS


BODY_PARTS = [
    "background", "torso", "right_hand", "left_hand",
    "left_foot", "right_foot", "upper_right_leg", "upper_left_leg",
    "lower_right_leg", "lower_left_leg", "upper_left_arm", "upper_right_arm",
    "lower_left_arm", "lower_right_arm", "head_left", "head_right",
    "torso_back", "right_hand_back", "left_hand_back",
    "left_foot_back", "right_foot_back", "upper_right_leg_back",
    "upper_left_leg_back", "lower_right_leg_back", "lower_left_leg_back",
]

NUM_BODY_PARTS = 25


class FPNBlock(nn.Module):
    """Feature Pyramid Network building block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, top_down: Optional[torch.Tensor] = None) -> torch.Tensor:
        lateral = self.lateral(x)
        if top_down is not None:
            top_down = F.interpolate(top_down, size=lateral.shape[2:], mode="bilinear", align_corners=False)
            lateral = lateral + top_down
        return self.smooth(lateral)


class DensePoseBackbone(nn.Module):
    """ResNet-FPN backbone for DensePose."""

    def __init__(self, base_channels: int = 64, fpn_channels: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, 3)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 4, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 6, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 3, stride=2)

        self.fpn4 = FPNBlock(base_channels * 8, fpn_channels)
        self.fpn3 = FPNBlock(base_channels * 4, fpn_channels)
        self.fpn2 = FPNBlock(base_channels * 2, fpn_channels)
        self.fpn1 = FPNBlock(base_channels, fpn_channels)

        self.out_channels = fpn_channels

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        for i in range(blocks):
            s = stride if i == 0 else 1
            ic = in_ch if i == 0 else out_ch
            downsample = None
            if s != 1 or ic != out_ch:
                downsample = nn.Sequential(nn.Conv2d(ic, out_ch, 1, s, bias=False), nn.BatchNorm2d(out_ch))
            layers.append(_ResBlock(ic, out_ch, s, downsample))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.fpn4(c4)
        p3 = self.fpn3(c3, p4)
        p2 = self.fpn2(c2, p3)
        p1 = self.fpn1(c1, p2)

        return p1


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class DensePoseHead(nn.Module):
    """DensePose prediction head: body part segmentation + UV coordinates."""

    def __init__(self, in_channels: int = 256, num_parts: int = NUM_BODY_PARTS, uv_channels: int = 2):
        super().__init__()
        self.num_parts = num_parts

        self.shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.part_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_parts, 1),
        )

        self.u_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_parts, 1),
            nn.Sigmoid(),
        )

        self.v_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_parts, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared_layers(x)
        part_logits = self.part_head(shared)
        u_coords = self.u_head(shared)
        v_coords = self.v_head(shared)
        return {"part_logits": part_logits, "u": u_coords, "v": v_coords}


@MODELS.register("DensePose")
class DensePose(nn.Module):
    """DensePose: Dense human pose estimation mapping pixels to body surface UV.

    Predicts body part segmentation and UV coordinates for each pixel,
    enabling dense correspondence between image pixels and 3D body surface.

    Args:
        base_channels: Backbone base channel width.
        fpn_channels: FPN feature dimension.
        num_parts: Number of body surface parts.
        output_stride: Output stride relative to input.
    """

    def __init__(
        self,
        base_channels: int = 64,
        fpn_channels: int = 256,
        num_parts: int = NUM_BODY_PARTS,
        output_stride: int = 4,
    ):
        super().__init__()
        self.output_stride = output_stride
        self.backbone = DensePoseBackbone(base_channels, fpn_channels)
        self.head = DensePoseHead(fpn_channels, num_parts)

    @property
    def out_channels(self) -> int:
        return self.backbone.out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Dict with 'part_logits' (B, P, H', W'), 'u' (B, P, H', W'),
            'v' (B, P, H', W') where H', W' = H/stride, W/stride.
        """
        features = self.backbone(x)
        return self.head(features)

    def get_dense_uv(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get full-resolution UV map.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Dict with dense UV coordinates and part assignments.
        """
        output = self.forward(x)
        H, W = x.shape[2], x.shape[3]

        part_logits = F.interpolate(output["part_logits"], size=(H, W), mode="bilinear", align_corners=False)
        u = F.interpolate(output["u"], size=(H, W), mode="bilinear", align_corners=False)
        v = F.interpolate(output["v"], size=(H, W), mode="bilinear", align_corners=False)

        part_labels = part_logits.argmax(dim=1)

        B, P = u.shape[:2]
        u_final = torch.zeros(B, 1, H, W, device=x.device)
        v_final = torch.zeros(B, 1, H, W, device=x.device)
        for b in range(B):
            for p in range(1, P):
                mask = (part_labels[b] == p)
                u_final[b, 0, mask] = u[b, p, mask]
                v_final[b, 0, mask] = v[b, p, mask]

        return {
            "part_labels": part_labels,
            "u": u_final,
            "v": v_final,
            "part_logits": part_logits,
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_parts: torch.Tensor,
        gt_u: torch.Tensor,
        gt_v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute DensePose training loss.

        Args:
            predictions: Model output dict.
            gt_parts: Ground-truth part labels (B, H, W).
            gt_u: Ground-truth U coordinates (B, H, W).
            gt_v: Ground-truth V coordinates (B, H, W).
            mask: Valid pixel mask (B, H, W).

        Returns:
            Dict with loss components.
        """
        pred_parts = predictions["part_logits"]
        if pred_parts.shape[2:] != gt_parts.shape[1:]:
            pred_parts = F.interpolate(pred_parts, size=gt_parts.shape[1:], mode="bilinear", align_corners=False)

        part_loss = F.cross_entropy(pred_parts, gt_parts.long(), ignore_index=-1)

        if mask is not None:
            valid = mask.bool()
        else:
            valid = gt_parts > 0

        pred_u = predictions["u"]
        pred_v = predictions["v"]
        if pred_u.shape[2:] != gt_u.shape[1:]:
            pred_u = F.interpolate(pred_u, size=gt_u.shape[1:], mode="bilinear", align_corners=False)
            pred_v = F.interpolate(pred_v, size=gt_v.shape[1:], mode="bilinear", align_corners=False)

        B = gt_u.shape[0]
        u_loss = torch.tensor(0.0, device=gt_u.device)
        v_loss = torch.tensor(0.0, device=gt_v.device)
        count = 0
        for b in range(B):
            for p in range(1, pred_u.shape[1]):
                m = valid[b] & (gt_parts[b] == p)
                if m.any():
                    u_loss = u_loss + F.smooth_l1_loss(pred_u[b, p][m], gt_u[b][m])
                    v_loss = v_loss + F.smooth_l1_loss(pred_v[b, p][m], gt_v[b][m])
                    count += 1

        if count > 0:
            u_loss = u_loss / count
            v_loss = v_loss / count

        total = part_loss + u_loss + v_loss
        return {"total": total, "part_loss": part_loss, "u_loss": u_loss, "v_loss": v_loss}
