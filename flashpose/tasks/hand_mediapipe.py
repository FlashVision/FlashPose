"""MediaPipe-style hand pose estimation with palm detection + landmark model.

Implements a two-stage hand pose pipeline:
1. Palm detection: lightweight SSD-style detector for hand bounding boxes
2. Hand landmark: 21 keypoints per hand with depth estimation

Designed for real-time inference on mobile/edge devices.

Reference: "MediaPipe Hands: On-device Real-time Hand Tracking"
           (Zhang et al., CVPR 2020 Workshop)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flashpose.registry import TASKS


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        return F.relu(self.bn2(self.pw(x)), inplace=True)


class PalmDetector(nn.Module):
    """Lightweight palm detection network (SSD-style single-shot detector).

    Detects palm bounding boxes and handedness from input images.

    Args:
        in_channels: Input image channels.
        num_anchors: Number of anchor boxes per spatial location.
    """

    def __init__(self, in_channels: int = 3, num_anchors: int = 2):
        super().__init__()
        self.num_anchors = num_anchors

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 64, stride=2),
            DepthwiseSeparableConv(64, 128),
            DepthwiseSeparableConv(128, 128, stride=2),
            DepthwiseSeparableConv(128, 256),
            DepthwiseSeparableConv(256, 256, stride=2),
        )

        self.cls_head = nn.Conv2d(256, num_anchors * 2, 1)
        self.reg_head = nn.Conv2d(256, num_anchors * 4, 1)
        self.handedness_head = nn.Conv2d(256, num_anchors * 2, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        bbox_reg = self.reg_head(features)
        handedness = self.handedness_head(features)

        B, _, H, W = cls_logits.shape
        cls_logits = cls_logits.reshape(B, self.num_anchors, 2, H, W).permute(0, 1, 3, 4, 2).reshape(B, -1, 2)
        bbox_reg = bbox_reg.reshape(B, self.num_anchors, 4, H, W).permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
        handedness = handedness.reshape(B, self.num_anchors, 2, H, W).permute(0, 1, 3, 4, 2).reshape(B, -1, 2)

        return {"cls_logits": cls_logits, "bbox": bbox_reg, "handedness": handedness}


class HandLandmarkModel(nn.Module):
    """Hand landmark model predicting 21 keypoints with 3D coordinates.

    Each keypoint has (x, y, z) coordinates where z represents relative
    depth from the palm.

    Args:
        in_channels: Input channels.
        num_keypoints: Number of hand landmarks (21).
    """

    def __init__(self, in_channels: int = 3, num_keypoints: int = 21):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 64, stride=2),
            DepthwiseSeparableConv(64, 128),
            DepthwiseSeparableConv(128, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 256, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.landmark_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_keypoints * 3),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)

        landmarks = self.landmark_head(pooled).reshape(-1, self.num_keypoints, 3)
        confidence = self.confidence_head(pooled)

        return {"landmarks": landmarks, "confidence": confidence}


class HandMediaPipePipeline(nn.Module):
    """Full MediaPipe-style hand tracking pipeline.

    Two-stage: palm detection followed by landmark regression on
    cropped hand regions.

    Args:
        input_size: Expected input image size for palm detection.
        landmark_size: Crop size for landmark model.
    """

    def __init__(self, input_size: int = 256, landmark_size: int = 224):
        super().__init__()
        self.input_size = input_size
        self.landmark_size = landmark_size

        self.palm_detector = PalmDetector()
        self.landmark_model = HandLandmarkModel()

    def detect_palms(self, image: torch.Tensor, conf_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Run palm detection.

        Args:
            image: (B, 3, H, W) input image.
            conf_threshold: Detection confidence threshold.

        Returns:
            Detection results with boxes and scores.
        """
        det_out = self.palm_detector(image)
        scores = det_out["cls_logits"].softmax(dim=-1)[..., 1]
        return {"boxes": det_out["bbox"], "scores": scores, "handedness": det_out["handedness"]}

    def predict_landmarks(self, hand_crops: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict landmarks from cropped hand images.

        Args:
            hand_crops: (N, 3, landmark_size, landmark_size) hand crops.

        Returns:
            Landmark predictions.
        """
        return self.landmark_model(hand_crops)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """End-to-end inference: detect palms then predict landmarks.

        For training/testing, processes the full image through the landmark
        model directly (assumes pre-cropped hand images).

        Args:
            image: (B, 3, H, W) hand crops or full images.

        Returns:
            Dict with 'landmarks' (B, 21, 3) and 'confidence' (B, 1).
        """
        return self.landmark_model(image)


HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


@TASKS.register("hand_mediapipe")
class HandMediaPipeTask:
    """MediaPipe-style hand pose task with 21 landmarks per hand.

    Provides palm detection + hand landmark estimation with
    real-time performance target.
    """

    NUM_KEYPOINTS = 21
    KEYPOINT_NAMES = HAND_LANDMARK_NAMES
    CONNECTIONS = HAND_CONNECTIONS

    FINGER_GROUPS = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    def __init__(self, input_size: int = 224):
        self.input_size = input_size

    @property
    def name(self) -> str:
        return "hand_mediapipe"

    @property
    def num_keypoints(self) -> int:
        return self.NUM_KEYPOINTS

    def get_model(self) -> HandMediaPipePipeline:
        """Create a hand MediaPipe pipeline model."""
        return HandMediaPipePipeline(landmark_size=self.input_size)

    def is_finger_extended(self, landmarks: np.ndarray, finger: str) -> bool:
        """Determine if a finger is extended.

        Args:
            landmarks: (21, 3) hand landmarks.
            finger: Finger name.

        Returns:
            True if the finger tip is above (farther from palm than) its PIP.
        """
        indices = self.FINGER_GROUPS.get(finger)
        if indices is None:
            return False

        tip = landmarks[indices[-1]]
        pip = landmarks[indices[1]]
        wrist = landmarks[0]

        dist_tip = np.linalg.norm(tip[:2] - wrist[:2])
        dist_pip = np.linalg.norm(pip[:2] - wrist[:2])
        return dist_tip > dist_pip

    def count_fingers(self, landmarks: np.ndarray) -> int:
        """Count extended fingers."""
        return sum(1 for f in self.FINGER_GROUPS if self.is_finger_extended(landmarks, f))

    def recognize_gesture(self, landmarks: np.ndarray) -> str:
        """Basic gesture recognition from landmarks.

        Args:
            landmarks: (21, 3) hand landmarks.

        Returns:
            Recognized gesture name.
        """
        count = self.count_fingers(landmarks)
        if count == 0:
            return "fist"
        elif count == 1 and self.is_finger_extended(landmarks, "index"):
            return "pointing"
        elif count == 2 and self.is_finger_extended(landmarks, "index") and self.is_finger_extended(landmarks, "middle"):
            return "peace"
        elif count == 5:
            return "open_palm"
        elif count == 1 and self.is_finger_extended(landmarks, "thumb"):
            return "thumbs_up"
        return f"fingers_{count}"

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate hand landmark predictions.

        Args:
            predictions: (N, 21, 3) predicted landmarks.
            ground_truth: (N, 21, 3) ground-truth landmarks.

        Returns:
            Evaluation metrics.
        """
        dists = np.linalg.norm(predictions[:, :, :2] - ground_truth[:, :, :2], axis=-1)
        metrics = {
            "mean_error": float(dists.mean()),
            "PCK@0.1": float((dists < 0.1).mean()),
            "PCK@0.2": float((dists < 0.2).mean()),
        }

        for finger, indices in self.FINGER_GROUPS.items():
            finger_err = dists[:, indices].mean()
            metrics[f"error_{finger}"] = float(finger_err)

        return metrics
