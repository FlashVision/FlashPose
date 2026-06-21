"""Gesture recognition solution based on hand keypoints."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from flashpose.tasks.hand import HandTask


class GestureRecognizer:
    """Recognize hand gestures from hand keypoint detections.

    Uses geometric rules on finger extension patterns to classify
    common gestures without a trained neural network.

    Supported gestures: open_palm, fist, peace, thumbs_up, pointing, ok, rock.

    Example:
        recognizer = GestureRecognizer()
        result = recognizer.recognize(hand_keypoints)
    """

    GESTURES = [
        "open_palm",
        "fist",
        "peace",
        "thumbs_up",
        "pointing",
        "ok",
        "rock",
        "unknown",
    ]

    def __init__(self, confidence_threshold: float = 0.6):
        self.hand_task = HandTask()
        self.confidence_threshold = confidence_threshold
        self.history: List[str] = []

    def recognize(self, hand_keypoints: np.ndarray) -> Dict:
        """Recognize gesture from hand keypoints.

        Args:
            hand_keypoints: (21, 2) hand keypoint coordinates.

        Returns:
            Dict with 'gesture', 'confidence', and finger extension details.
        """
        fingers = {}
        for finger in self.hand_task.FINGER_GROUPS:
            fingers[finger] = self.hand_task.is_finger_extended(hand_keypoints, finger)

        extended_count = sum(fingers.values())
        gesture, confidence = self._classify_gesture(fingers, extended_count, hand_keypoints)

        self.history.append(gesture)
        if len(self.history) > 10:
            self.history = self.history[-10:]

        return {
            "gesture": gesture,
            "confidence": confidence,
            "fingers_extended": fingers,
            "num_extended": extended_count,
        }

    def _classify_gesture(
        self,
        fingers: Dict[str, bool],
        extended_count: int,
        keypoints: np.ndarray,
    ) -> tuple:
        """Classify gesture based on finger extension pattern."""
        if extended_count == 5:
            return "open_palm", 0.95

        if extended_count == 0:
            return "fist", 0.9

        if extended_count == 2 and fingers["index"] and fingers["middle"]:
            if not fingers["ring"] and not fingers["pinky"]:
                return "peace", 0.9

        if extended_count == 1 and fingers["thumb"]:
            thumb_tip = keypoints[4]
            wrist = keypoints[0]
            if thumb_tip[1] < wrist[1]:
                return "thumbs_up", 0.85

        if extended_count == 1 and fingers["index"]:
            return "pointing", 0.85

        if self._is_ok_gesture(keypoints, fingers):
            return "ok", 0.8

        if extended_count == 2 and fingers["index"] and fingers["pinky"]:
            return "rock", 0.85

        return "unknown", 0.3

    def _is_ok_gesture(self, keypoints: np.ndarray, fingers: Dict[str, bool]) -> bool:
        """Check if thumb and index form an 'OK' circle."""
        thumb_tip = keypoints[4]
        index_tip = keypoints[8]
        distance = np.linalg.norm(thumb_tip - index_tip)

        hand_size = np.linalg.norm(keypoints[0] - keypoints[9])
        if hand_size < 1e-6:
            return False

        normalized_dist = distance / hand_size
        return normalized_dist < 0.3 and fingers.get("middle", False)

    def get_stable_gesture(self, window: int = 5) -> Optional[str]:
        """Get the most frequent gesture in the recent history.

        Args:
            window: Number of recent frames to consider.

        Returns:
            Most frequent gesture or None if history is too short.
        """
        if len(self.history) < window:
            return None

        recent = self.history[-window:]
        counts = {}
        for g in recent:
            counts[g] = counts.get(g, 0) + 1

        best = max(counts, key=counts.get)
        if counts[best] / window >= self.confidence_threshold:
            return best
        return None

    def reset(self):
        """Clear gesture history."""
        self.history.clear()
