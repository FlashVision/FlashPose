"""2D body pose estimation task."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from flashpose.cfg.config import PoseConfig, get_config
from flashpose.data.keypoint_utils import COCO_KEYPOINTS, COCO_SKELETON, COCO_FLIP_PAIRS
from flashpose.registry import TASKS


@TASKS.register("body_2d")
class Body2DTask:
    """2D body pose estimation with 17 COCO keypoints.

    Handles task-specific configuration, evaluation metrics,
    and keypoint post-processing.
    """

    KEYPOINTS = COCO_KEYPOINTS
    SKELETON = COCO_SKELETON
    FLIP_PAIRS = COCO_FLIP_PAIRS
    NUM_KEYPOINTS = 17

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or get_config(task="body_2d")

    @property
    def name(self) -> str:
        return "body_2d"

    @property
    def num_keypoints(self) -> int:
        return self.NUM_KEYPOINTS

    def get_default_config(self) -> PoseConfig:
        return get_config(task="body_2d", num_keypoints=17, input_size=(256, 192))

    def postprocess(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Post-process predictions: filter low-confidence and compute bbox.

        Args:
            keypoints: (K, 2) predicted keypoint coordinates.
            scores: (K,) confidence scores per keypoint.
            center: (2,) original bbox center.
            scale: (2,) original bbox scale.

        Returns:
            Dict with filtered keypoints, scores, and bounding box.
        """
        valid_mask = scores > 0.3
        valid_kps = keypoints[valid_mask]

        if len(valid_kps) > 0:
            bbox = np.array([
                valid_kps[:, 0].min(),
                valid_kps[:, 1].min(),
                valid_kps[:, 0].max(),
                valid_kps[:, 1].max(),
            ])
        else:
            bbox = np.zeros(4)

        return {
            "keypoints": keypoints,
            "scores": scores,
            "valid_mask": valid_mask,
            "bbox": bbox,
            "mean_score": float(scores[valid_mask].mean()) if valid_mask.any() else 0.0,
        }

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        bbox_sizes: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate predictions against ground truth.

        Args:
            predictions: (N, K, 2) predicted coordinates.
            ground_truth: (N, K, 2) ground truth coordinates.
            bbox_sizes: (N,) bounding box sizes for normalization.

        Returns:
            Dict of metric name -> value.
        """
        from flashpose.analytics.metrics import compute_pck, compute_ap

        pck_50 = compute_pck(predictions, ground_truth, threshold=0.5, bbox_sizes=bbox_sizes)
        pck_20 = compute_pck(predictions, ground_truth, threshold=0.2, bbox_sizes=bbox_sizes)
        ap = compute_ap(predictions, ground_truth)

        return {"PCK@0.5": pck_50, "PCK@0.2": pck_20, "AP": ap}
