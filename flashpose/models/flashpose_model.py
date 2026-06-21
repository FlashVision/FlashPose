"""FlashPose — unified model builder that composes backbone + head for any task."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from flashpose.cfg.config import PoseConfig
from flashpose.registry import MODELS


@MODELS.register("FlashPose")
class FlashPose(nn.Module):
    """Unified pose estimation model combining a backbone with a prediction head.

    Supports ViTPose, HRNet, and RTMPose backbones with heatmap, regression,
    or SimCC heads.
    """

    def __init__(self, config: PoseConfig):
        super().__init__()
        self.config = config
        self.backbone = self._build_backbone(config)
        self.head = self._build_head(config)

    def _build_backbone(self, config: PoseConfig) -> nn.Module:
        from flashpose.models.architectures.vitpose import ViTPose
        from flashpose.models.architectures.hrnet import HRNet
        from flashpose.models.architectures.rtmpose import RTMPose

        backbone_map = {
            "ViTPose": ViTPose,
            "HRNet": HRNet,
            "RTMPose": RTMPose,
        }

        backbone_cls = backbone_map.get(config.model_name)
        if backbone_cls is None:
            raise ValueError(f"Unknown backbone: {config.model_name}. Choose from {list(backbone_map.keys())}")

        return backbone_cls(config)

    def _build_head(self, config: PoseConfig) -> nn.Module:
        from flashpose.heads.heatmap_head import HeatmapHead
        from flashpose.heads.regression_head import RegressionHead
        from flashpose.heads.simcc_head import SimCCHead

        head_map = {
            "heatmap": HeatmapHead,
            "regression": RegressionHead,
            "simcc": SimCCHead,
        }

        head_cls = head_map.get(config.head)
        if head_cls is None:
            raise ValueError(f"Unknown head: {config.head}. Choose from {list(head_map.keys())}")

        in_channels = self.backbone.out_channels
        return head_cls(in_channels=in_channels, num_keypoints=config.num_keypoints, config=config)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone and head.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Dictionary with 'heatmaps' or 'keypoints' depending on head type.
        """
        features = self.backbone(x)
        output = self.head(features)
        return output

    def load_pretrained(self, path: str, strict: bool = False) -> None:
        """Load pretrained weights from a checkpoint file.

        Args:
            path: Path to .pth checkpoint.
            strict: Whether to require exact key matching.
        """
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            cleaned[k] = v

        self.load_state_dict(cleaned, strict=strict)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
