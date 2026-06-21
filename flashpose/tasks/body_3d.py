"""3D body pose estimation task (lifting from 2D or direct 3D prediction)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from flashpose.cfg.config import PoseConfig, get_config
from flashpose.data.keypoint_utils import H36M_KEYPOINTS, H36M_SKELETON
from flashpose.registry import TASKS


class PoseLifter(nn.Module):
    """Simple MLP-based 2D-to-3D pose lifting network.

    Takes 2D keypoints and predicts their 3D coordinates via fully connected layers.
    """

    def __init__(self, num_joints: int = 17, hidden_dim: int = 1024, num_layers: int = 4, dropout: float = 0.25):
        super().__init__()
        layers = []
        in_dim = num_joints * 2

        for i in range(num_layers):
            out_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])

        layers.append(nn.Linear(hidden_dim, num_joints * 3))
        self.net = nn.Sequential(*layers)
        self.num_joints = num_joints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lift 2D keypoints to 3D.

        Args:
            x: (B, J*2) flattened 2D keypoints.

        Returns:
            (B, J, 3) predicted 3D coordinates.
        """
        B = x.shape[0]
        out = self.net(x.flatten(1))
        return out.reshape(B, self.num_joints, 3)


@TASKS.register("body_3d")
class Body3DTask:
    """3D body pose estimation task.

    Supports direct 3D prediction and 2D-to-3D lifting approaches.
    Uses Human3.6M skeleton with 17 joints.
    """

    KEYPOINTS = H36M_KEYPOINTS
    SKELETON = H36M_SKELETON
    NUM_KEYPOINTS = 17

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or get_config(task="body_3d")

    @property
    def name(self) -> str:
        return "body_3d"

    @property
    def num_keypoints(self) -> int:
        return self.NUM_KEYPOINTS

    def get_default_config(self) -> PoseConfig:
        return get_config(task="body_3d")

    def build_lifter(self, hidden_dim: int = 1024, num_layers: int = 4) -> PoseLifter:
        """Build a 2D-to-3D pose lifting network.

        Args:
            hidden_dim: Hidden layer dimension.
            num_layers: Number of FC layers.

        Returns:
            PoseLifter module.
        """
        return PoseLifter(num_joints=self.NUM_KEYPOINTS, hidden_dim=hidden_dim, num_layers=num_layers)

    def evaluate(
        self,
        predictions_3d: np.ndarray,
        ground_truth_3d: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate 3D pose predictions.

        Args:
            predictions_3d: (N, K, 3) predicted 3D coordinates.
            ground_truth_3d: (N, K, 3) ground truth 3D coordinates.

        Returns:
            Dict with MPJPE and PA-MPJPE metrics.
        """
        from flashpose.analytics.metrics import compute_mpjpe, compute_pa_mpjpe

        mpjpe = compute_mpjpe(predictions_3d, ground_truth_3d)
        pa_mpjpe = compute_pa_mpjpe(predictions_3d, ground_truth_3d)

        return {"MPJPE": mpjpe, "PA-MPJPE": pa_mpjpe}

    def procrustes_align(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Align predicted pose to target using Procrustes analysis.

        Args:
            predicted: (K, 3) predicted 3D keypoints.
            target: (K, 3) ground truth 3D keypoints.

        Returns:
            Aligned (K, 3) prediction.
        """
        mu_pred = predicted.mean(axis=0)
        mu_target = target.mean(axis=0)
        pred_centered = predicted - mu_pred
        target_centered = target - mu_target

        H = pred_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        scale = np.trace(R @ H) / np.trace(pred_centered.T @ pred_centered)
        aligned = scale * (pred_centered @ R.T) + mu_target

        return aligned
