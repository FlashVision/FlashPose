"""Animal Pose Estimation with AP-10K dataset support.

Implements species-agnostic quadruped pose estimation with configurable
skeleton topology for various animal types.

Reference: "AP-10K: A Benchmark for Animal Pose Estimation in the Wild"
           (Yu et al., NeurIPS 2021)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from flashpose.cfg.config import PoseConfig
from flashpose.registry import TASKS


ANIMAL_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "throat", "withers", "tail_base",
    "left_front_elbow", "left_front_knee", "left_front_paw",
    "right_front_elbow", "right_front_knee", "right_front_paw",
    "left_back_elbow", "left_back_knee", "left_back_paw",
    "right_back_elbow", "right_back_knee", "right_back_paw",
]

ANIMAL_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (5, 6), (6, 7),
    (5, 8), (8, 9), (9, 10),
    (5, 11), (11, 12), (12, 13),
    (7, 14), (14, 15), (15, 16),
    (7, 17), (17, 18), (18, 19),
]

SPECIES_CONFIGS = {
    "quadruped": {"num_keypoints": 20, "keypoints": ANIMAL_KEYPOINTS, "skeleton": ANIMAL_SKELETON},
    "dog": {"num_keypoints": 20, "keypoints": ANIMAL_KEYPOINTS, "skeleton": ANIMAL_SKELETON},
    "cat": {"num_keypoints": 20, "keypoints": ANIMAL_KEYPOINTS, "skeleton": ANIMAL_SKELETON},
    "horse": {"num_keypoints": 20, "keypoints": ANIMAL_KEYPOINTS, "skeleton": ANIMAL_SKELETON},
    "bird": {
        "num_keypoints": 14,
        "keypoints": ["beak", "left_eye", "right_eye", "crown", "nape", "left_wing_tip",
                      "right_wing_tip", "left_wing_shoulder", "right_wing_shoulder",
                      "belly", "tail", "left_leg", "right_leg", "left_foot", "right_foot"][:14],
        "skeleton": [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6),
                     (5, 7), (6, 8), (4, 9), (9, 10), (9, 11), (9, 12)],
    },
}


class AnimalPoseBackbone(nn.Module):
    """Lightweight backbone for animal pose estimation."""

    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            self._make_stage(base_ch, base_ch, 2),
            self._make_stage(base_ch, base_ch * 2, 2, stride=2),
            self._make_stage(base_ch * 2, base_ch * 4, 2, stride=2),
        )
        self.out_channels = base_ch * 4

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        for i in range(blocks):
            s = stride if i == 0 else 1
            ic = in_ch if i == 0 else out_ch
            layers.append(nn.Sequential(
                nn.Conv2d(ic, out_ch, 3, s, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            ))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class AnimalPoseHead(nn.Module):
    """Heatmap head for animal pose with deconvolution upsampling."""

    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        return self.head(x)


class AnimalPoseModel(nn.Module):
    """Species-agnostic animal pose estimator.

    Args:
        species: Animal species or 'quadruped' for generic.
        num_keypoints: Override keypoint count.
        base_channels: Backbone base channel width.
    """

    def __init__(self, species: str = "quadruped", num_keypoints: Optional[int] = None, base_channels: int = 64):
        super().__init__()
        config = SPECIES_CONFIGS.get(species, SPECIES_CONFIGS["quadruped"])
        self.species = species
        self.num_keypoints = num_keypoints or config["num_keypoints"]
        self.keypoint_names = config["keypoints"][:self.num_keypoints]
        self.skeleton = config["skeleton"]

        self.backbone = AnimalPoseBackbone(base_ch=base_channels)
        self.head = AnimalPoseHead(self.backbone.out_channels, self.num_keypoints)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        heatmaps = self.head(features)
        return {"heatmaps": heatmaps}

    def decode_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Decode heatmaps to keypoint coordinates.

        Args:
            heatmaps: (B, K, H, W) predicted heatmaps.

        Returns:
            (B, K, 2) keypoint coordinates normalized to [0, 1].
        """
        B, K, H, W = heatmaps.shape
        flat = heatmaps.reshape(B, K, -1)
        max_idx = flat.argmax(dim=-1)
        y = (max_idx // W).float() / (H - 1)
        x = (max_idx % W).float() / (W - 1)
        return torch.stack([x, y], dim=-1)


@TASKS.register("animal_pose")
class AnimalPoseTask:
    """Animal pose estimation task with AP-10K dataset support.

    Supports quadruped animals with 20 keypoints by default,
    configurable for different species.
    """

    def __init__(self, species: str = "quadruped", config: Optional[PoseConfig] = None):
        self.species = species
        sp_config = SPECIES_CONFIGS.get(species, SPECIES_CONFIGS["quadruped"])
        self.num_kpts = sp_config["num_keypoints"]
        self.keypoints = sp_config["keypoints"]
        self.skeleton = sp_config["skeleton"]
        self.config = config

    @property
    def name(self) -> str:
        return f"animal_pose_{self.species}"

    @property
    def num_keypoints(self) -> int:
        return self.num_kpts

    def get_model(self, base_channels: int = 64) -> AnimalPoseModel:
        """Build an animal pose model for this species."""
        return AnimalPoseModel(self.species, base_channels=base_channels)

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate animal pose predictions using AP metric.

        Args:
            predictions: (N, K, 2) predicted keypoints.
            ground_truth: (N, K, 2) ground truth keypoints.
            bbox: Optional (N, 4) bounding boxes for normalization.

        Returns:
            Dictionary of evaluation metrics.
        """
        if bbox is not None:
            areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
            scales = np.sqrt(areas)
        else:
            scales = np.ones(len(predictions)) * 100.0

        dists = np.linalg.norm(predictions - ground_truth, axis=-1)
        normalized = dists / scales[:, None]

        thresholds = [0.05, 0.1, 0.2]
        metrics = {}
        for thr in thresholds:
            pck = (normalized < thr).mean()
            metrics[f"PCK@{thr}"] = float(pck)

        metrics["mean_error"] = float(dists.mean())
        return metrics
