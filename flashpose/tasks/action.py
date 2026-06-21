"""Action recognition task based on skeleton sequences."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from flashpose.cfg.config import PoseConfig, get_config
from flashpose.registry import TASKS


class SkeletonGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for skeleton-based action recognition.

    Processes sequences of skeleton keypoints to classify actions using graph
    convolutions over the body structure and temporal convolutions along time.
    """

    def __init__(self, num_joints: int = 17, in_channels: int = 2, num_classes: int = 60, hidden_dim: int = 128):
        super().__init__()
        self.num_joints = num_joints
        self.num_classes = num_classes

        self.spatial_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, num_joints) for _ in range(3)
        ])

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * num_joints, hidden_dim * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify action from skeleton sequence.

        Args:
            x: (B, T, J, C) skeleton sequence (time, joints, coords).

        Returns:
            (B, num_classes) action logits.
        """
        B, T, J, C = x.shape

        x = self.spatial_embed(x)

        x_gcn = x.reshape(B * T, J, -1)
        for gcn in self.gcn_layers:
            x_gcn = gcn(x_gcn)
        x = x_gcn.reshape(B, T, J, -1)

        x = x.reshape(B, T, -1).permute(0, 2, 1)
        x = self.temporal_conv(x)
        logits = self.classifier(x)

        return logits


class GraphConvLayer(nn.Module):
    """Single graph convolution layer over skeleton joints."""

    def __init__(self, in_features: int, out_features: int, num_joints: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.adj = nn.Parameter(torch.eye(num_joints) + torch.randn(num_joints, num_joints) * 0.01)
        self.bn = nn.BatchNorm1d(num_joints)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply graph convolution.

        Args:
            x: (B, J, C) node features.

        Returns:
            (B, J, C_out) updated features.
        """
        adj_norm = torch.softmax(self.adj, dim=-1)
        x = adj_norm @ x
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@TASKS.register("action")
class ActionTask:
    """Skeleton-based action recognition task.

    Classifies human actions from sequences of 2D/3D skeleton keypoints
    using a Spatial-Temporal Graph Convolutional Network.
    """

    NTU_ACTIONS: List[str] = [
        "drink_water", "eat_meal", "brush_teeth", "brush_hair", "drop",
        "pickup", "throw", "sit_down", "stand_up", "clapping",
        "reading", "writing", "tear_up_paper", "wear_jacket", "take_off_jacket",
        "wear_a_shoe", "take_off_a_shoe", "wear_on_glasses", "take_off_glasses", "put_on_a_hat",
        "take_off_a_hat", "cheer_up", "hand_waving", "kicking_something", "reach_into_pocket",
        "hopping", "jump_up", "make_a_phone_call", "playing_with_phone", "typing_on_keyboard",
        "pointing_to_something", "taking_a_selfie", "check_time", "rub_two_hands", "nod_head",
        "shake_head", "wipe_face", "salute", "put_palms_together", "cross_hands",
        "sneeze", "staggering", "falling", "touch_head", "touch_chest",
        "touch_back", "touch_neck", "nausea", "fan_self", "punch",
        "kicking", "pushing", "pat_on_back", "point_finger", "hugging",
        "giving_something", "touch_pocket", "handshaking", "walking_towards", "walking_apart",
    ]

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or get_config(task="action")

    @property
    def name(self) -> str:
        return "action"

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    def get_default_config(self) -> PoseConfig:
        return get_config(task="action", num_classes=60)

    def build_model(self, num_joints: int = 17, in_channels: int = 2, num_classes: int = 60) -> SkeletonGCN:
        """Build the action recognition model.

        Args:
            num_joints: Number of skeleton joints.
            in_channels: Coordinate dimensions (2 for 2D, 3 for 3D).
            num_classes: Number of action categories.

        Returns:
            SkeletonGCN model.
        """
        return SkeletonGCN(num_joints=num_joints, in_channels=in_channels, num_classes=num_classes)

    def preprocess_sequence(
        self,
        keypoints_sequence: np.ndarray,
        target_length: int = 64,
    ) -> np.ndarray:
        """Normalize and pad/truncate a keypoint sequence to fixed length.

        Args:
            keypoints_sequence: (T, J, C) raw skeleton sequence.
            target_length: Desired sequence length.

        Returns:
            (target_length, J, C) normalized sequence.
        """
        T, J, C = keypoints_sequence.shape

        center = keypoints_sequence[:, 0:1, :]
        normalized = keypoints_sequence - center

        max_val = np.abs(normalized).max() + 1e-6
        normalized = normalized / max_val

        if T >= target_length:
            indices = np.linspace(0, T - 1, target_length).astype(int)
            return normalized[indices]
        else:
            padded = np.zeros((target_length, J, C), dtype=np.float32)
            padded[:T] = normalized
            return padded

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate action classification.

        Args:
            predictions: (N, num_classes) predicted logits or (N,) class indices.
            ground_truth: (N,) ground truth class indices.

        Returns:
            Dict with accuracy and top-5 accuracy.
        """
        if predictions.ndim == 2:
            pred_classes = predictions.argmax(axis=1)
            top5_preds = np.argsort(predictions, axis=1)[:, -5:]
        else:
            pred_classes = predictions
            top5_preds = predictions[:, np.newaxis]

        acc = float((pred_classes == ground_truth).mean())
        top5_acc = float(np.any(top5_preds == ground_truth[:, np.newaxis], axis=1).mean())

        return {"accuracy": acc, "top5_accuracy": top5_acc}
