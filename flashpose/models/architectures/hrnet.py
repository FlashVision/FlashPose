"""HRNet — High-Resolution Network backbone for pose estimation.

Reference: "Deep High-Resolution Representation Learning for Visual Recognition"
           (Sun et al., CVPR 2019)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashpose.cfg.config import PoseConfig
from flashpose.registry import MODELS


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class HRModule(nn.Module):
    """High-Resolution module with multi-resolution parallel convolutions and fusions."""

    def __init__(self, num_branches: int, channels: List[int], num_blocks: int = 4):
        super().__init__()
        self.num_branches = num_branches

        self.branches = nn.ModuleList()
        for i in range(num_branches):
            branch = nn.Sequential(*[
                BasicBlock(channels[i], channels[i]) for _ in range(num_blocks)
            ])
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j == i:
                    fuse_layer.append(nn.Identity())
                elif j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                    ))
                else:
                    downsample = []
                    for k in range(i - j):
                        out_ch = channels[i] if k == i - j - 1 else channels[j]
                        downsample.append(nn.Sequential(
                            nn.Conv2d(channels[j], out_ch, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True) if k < i - j - 1 else nn.Identity(),
                        ))
                    fuse_layer.append(nn.Sequential(*downsample))
            self.fuse_layers.append(fuse_layer)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        branch_outputs = [self.branches[i](inputs[i]) for i in range(self.num_branches)]

        fused = []
        for i in range(self.num_branches):
            y = None
            for j in range(self.num_branches):
                x_j = self.fuse_layers[i][j](branch_outputs[j])
                if j > i:
                    x_j = F.interpolate(x_j, size=branch_outputs[i].shape[2:], mode="bilinear", align_corners=False)
                y = x_j if y is None else y + x_j
            fused.append(self.relu(y))

        return fused


class TransitionLayer(nn.Module):
    """Transition between HRNet stages, creating new resolution branches."""

    def __init__(self, in_channels: List[int], out_channels: List[int]):
        super().__init__()
        self.transitions = nn.ModuleList()

        for i, out_ch in enumerate(out_channels):
            if i < len(in_channels):
                if in_channels[i] != out_ch:
                    self.transitions.append(nn.Sequential(
                        nn.Conv2d(in_channels[i], out_ch, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    self.transitions.append(nn.Identity())
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels[-1], out_ch, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                self.transitions.append(downsample)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i, trans in enumerate(self.transitions):
            if i < len(inputs):
                outputs.append(trans(inputs[i]))
            else:
                outputs.append(trans(inputs[-1]))
        return outputs


@MODELS.register("HRNet")
class HRNet(nn.Module):
    """High-Resolution Network for human pose estimation.

    Maintains high-resolution representations through parallel multi-resolution
    subnetworks with repeated multi-scale fusions.

    Variants:
        - hrnet_w32: channels=[32, 64, 128, 256]
        - hrnet_w48: channels=[48, 96, 192, 384]
    """

    CONFIGS = {
        "hrnet_w32": [32, 64, 128, 256],
        "hrnet_w48": [48, 96, 192, 384],
    }

    def __init__(self, config: PoseConfig):
        super().__init__()
        backbone_name = config.backbone if config.backbone in self.CONFIGS else "hrnet_w32"
        channels = self.CONFIGS[backbone_name]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for _ in range(4)])

        self.transition1 = TransitionLayer([64], channels[:2])
        self.stage2 = HRModule(2, channels[:2], num_blocks=4)

        self.transition2 = TransitionLayer(channels[:2], channels[:3])
        self.stage3 = HRModule(3, channels[:3], num_blocks=4)

        self.transition3 = TransitionLayer(channels[:3], channels[:4])
        self.stage4 = HRModule(4, channels[:4], num_blocks=4)

        self._out_channels = channels[0]

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning highest-resolution feature map.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Feature map (B, C, H/4, W/4) at 1/4 input resolution.
        """
        x = self.stem(x)
        x = self.layer1(x)

        x_list = self.transition1([x])
        x_list = self.stage2(x_list)

        x_list = self.transition2(x_list)
        x_list = self.stage3(x_list)

        x_list = self.transition3(x_list)
        x_list = self.stage4(x_list)

        return x_list[0]
