"""Face landmark detection task (68 keypoints)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from flashpose.cfg.config import PoseConfig, get_config
from flashpose.registry import TASKS


@TASKS.register("face")
class FaceTask:
    """Face landmark detection with 68 keypoints.

    Groups: jaw (0-16), left eyebrow (17-21), right eyebrow (22-26),
    nose bridge (27-30), nose tip (31-35), left eye (36-41),
    right eye (42-47), outer lip (48-59), inner lip (60-67).
    """

    NUM_KEYPOINTS = 68

    GROUPS = {
        "jaw": list(range(0, 17)),
        "left_eyebrow": list(range(17, 22)),
        "right_eyebrow": list(range(22, 27)),
        "nose_bridge": list(range(27, 31)),
        "nose_tip": list(range(31, 36)),
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "outer_lip": list(range(48, 60)),
        "inner_lip": list(range(60, 68)),
    }

    SKELETON: List[tuple] = []

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or get_config(task="face", num_keypoints=68)
        self._build_skeleton()

    def _build_skeleton(self):
        """Build face skeleton connections from groups."""
        connections = []
        for group, indices in self.GROUPS.items():
            for i in range(len(indices) - 1):
                connections.append((indices[i], indices[i + 1]))
            if group in ("left_eye", "right_eye", "outer_lip", "inner_lip"):
                connections.append((indices[-1], indices[0]))
        self.SKELETON = connections

    @property
    def name(self) -> str:
        return "face"

    @property
    def num_keypoints(self) -> int:
        return self.NUM_KEYPOINTS

    def get_default_config(self) -> PoseConfig:
        return get_config(task="face", num_keypoints=68, input_size=(256, 256))

    def compute_eye_aspect_ratio(self, keypoints: np.ndarray) -> float:
        """Compute Eye Aspect Ratio (EAR) for blink detection.

        Args:
            keypoints: (68, 2) face landmarks.

        Returns:
            Average EAR for both eyes.
        """
        def _ear(eye_pts):
            v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
            v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
            h = np.linalg.norm(eye_pts[0] - eye_pts[3])
            return (v1 + v2) / (2.0 * h + 1e-6)

        left_eye = keypoints[36:42]
        right_eye = keypoints[42:48]

        left_ear = _ear(left_eye)
        right_ear = _ear(right_eye)

        return (left_ear + right_ear) / 2.0

    def compute_mouth_aspect_ratio(self, keypoints: np.ndarray) -> float:
        """Compute Mouth Aspect Ratio (MAR) for open mouth detection.

        Args:
            keypoints: (68, 2) face landmarks.

        Returns:
            MAR value.
        """
        mouth = keypoints[48:68]
        v1 = np.linalg.norm(mouth[2] - mouth[10])
        v2 = np.linalg.norm(mouth[4] - mouth[8])
        h = np.linalg.norm(mouth[0] - mouth[6])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate face landmark predictions using NME.

        Args:
            predictions: (N, 68, 2) predicted coordinates.
            ground_truth: (N, 68, 2) ground truth coordinates.

        Returns:
            Dict with Normalized Mean Error (NME) metrics.
        """
        N = len(predictions)
        errors = []

        for i in range(N):
            left_eye_center = ground_truth[i, 36:42].mean(axis=0)
            right_eye_center = ground_truth[i, 42:48].mean(axis=0)
            interocular = np.linalg.norm(left_eye_center - right_eye_center)
            if interocular < 1e-6:
                continue

            dist = np.linalg.norm(predictions[i] - ground_truth[i], axis=1).mean()
            nme = dist / interocular
            errors.append(nme)

        mean_nme = float(np.mean(errors)) if errors else 0.0
        return {"NME_interocular": mean_nme}
