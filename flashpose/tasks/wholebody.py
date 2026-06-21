"""Whole-body pose estimation task (133 keypoints: body + hands + face)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from flashpose.cfg.config import PoseConfig, get_config
from flashpose.registry import TASKS


@TASKS.register("wholebody")
class WholeBodyTask:
    """Whole-body pose estimation with 133 keypoints.

    Combines body (17), left hand (21), right hand (21), face (68),
    and foot (6) keypoints into a single unified prediction.
    """

    NUM_KEYPOINTS = 133

    BODY_INDICES = list(range(0, 17))
    LEFT_HAND_INDICES = list(range(91, 112))
    RIGHT_HAND_INDICES = list(range(112, 133))
    FACE_INDICES = list(range(23, 91))
    FOOT_INDICES = list(range(17, 23))

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or get_config(task="wholebody", num_keypoints=133)

    @property
    def name(self) -> str:
        return "wholebody"

    @property
    def num_keypoints(self) -> int:
        return self.NUM_KEYPOINTS

    def get_default_config(self) -> PoseConfig:
        return get_config(task="wholebody", num_keypoints=133, input_size=(384, 288))

    def split_predictions(self, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """Split whole-body predictions into component groups.

        Args:
            keypoints: (133, 2) whole-body keypoints.

        Returns:
            Dict with 'body', 'left_hand', 'right_hand', 'face', 'foot' arrays.
        """
        return {
            "body": keypoints[self.BODY_INDICES],
            "foot": keypoints[self.FOOT_INDICES],
            "face": keypoints[self.FACE_INDICES],
            "left_hand": keypoints[self.LEFT_HAND_INDICES],
            "right_hand": keypoints[self.RIGHT_HAND_INDICES],
        }

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate whole-body predictions with per-component metrics.

        Args:
            predictions: (N, 133, 2) predicted coordinates.
            ground_truth: (N, 133, 2) ground truth coordinates.

        Returns:
            Dict with overall and per-component AP/PCK metrics.
        """
        from flashpose.analytics.metrics import compute_pck

        metrics = {}
        metrics["PCK@0.5_overall"] = compute_pck(predictions, ground_truth, threshold=0.5)

        component_indices = {
            "body": self.BODY_INDICES,
            "face": self.FACE_INDICES,
            "left_hand": self.LEFT_HAND_INDICES,
            "right_hand": self.RIGHT_HAND_INDICES,
        }

        for name, indices in component_indices.items():
            comp_pred = predictions[:, indices]
            comp_gt = ground_truth[:, indices]
            metrics[f"PCK@0.5_{name}"] = compute_pck(comp_pred, comp_gt, threshold=0.5)

        return metrics
