"""Hand pose estimation task (21 keypoints)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from flashpose.cfg.config import PoseConfig, get_config
from flashpose.data.keypoint_utils import HAND_KEYPOINTS, HAND_SKELETON
from flashpose.registry import TASKS


@TASKS.register("hand")
class HandTask:
    """Hand pose estimation task with 21 keypoints.

    Detects finger joints including tips, DIPs, PIPs, MCPs, and wrist.
    """

    KEYPOINTS = HAND_KEYPOINTS
    SKELETON = HAND_SKELETON
    NUM_KEYPOINTS = 21

    FINGER_GROUPS = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or get_config(task="hand", num_keypoints=21)

    @property
    def name(self) -> str:
        return "hand"

    @property
    def num_keypoints(self) -> int:
        return self.NUM_KEYPOINTS

    def get_default_config(self) -> PoseConfig:
        return get_config(task="hand", num_keypoints=21, input_size=(256, 256))

    def is_finger_extended(self, keypoints: np.ndarray, finger: str) -> bool:
        """Determine if a finger is extended based on joint positions.

        Args:
            keypoints: (21, 2) hand keypoint positions.
            finger: Finger name (thumb, index, middle, ring, pinky).

        Returns:
            True if the finger tip is farther from wrist than the MCP.
        """
        indices = self.FINGER_GROUPS.get(finger)
        if indices is None:
            return False

        wrist = keypoints[0]
        mcp = keypoints[indices[0]]
        tip = keypoints[indices[-1]]

        dist_tip = np.linalg.norm(tip - wrist)
        dist_mcp = np.linalg.norm(mcp - wrist)

        return dist_tip > dist_mcp * 1.2

    def count_extended_fingers(self, keypoints: np.ndarray) -> int:
        """Count the number of extended fingers.

        Args:
            keypoints: (21, 2) hand keypoints.

        Returns:
            Number of extended fingers (0-5).
        """
        count = 0
        for finger in self.FINGER_GROUPS:
            if self.is_finger_extended(keypoints, finger):
                count += 1
        return count

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate hand pose predictions.

        Args:
            predictions: (N, 21, 2) predicted coordinates.
            ground_truth: (N, 21, 2) ground truth coordinates.

        Returns:
            Dict with PCK and per-finger metrics.
        """
        from flashpose.analytics.metrics import compute_pck

        overall_pck = compute_pck(predictions, ground_truth, threshold=0.2)
        metrics = {"PCK@0.2": overall_pck}

        for finger, indices in self.FINGER_GROUPS.items():
            finger_preds = predictions[:, indices]
            finger_gts = ground_truth[:, indices]
            finger_pck = compute_pck(finger_preds, finger_gts, threshold=0.2)
            metrics[f"PCK@0.2_{finger}"] = finger_pck

        return metrics
