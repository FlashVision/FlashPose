"""Action classification solution based on skeleton sequences."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from flashpose.tasks.action import ActionTask, SkeletonGCN


class ActionClassifier:
    """High-level action classification from skeleton sequences.

    Classifies human actions by analyzing temporal sequences of body
    keypoints using a spatial-temporal GCN.

    Example:
        classifier = ActionClassifier(model_path="action_model.pth")
        result = classifier.classify(keypoints_sequence)
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        num_joints: int = 17,
        num_classes: int = 60,
        sequence_length: int = 64,
        in_channels: int = 2,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.in_channels = in_channels

        self.task = ActionTask()
        self.model = self._load_model(model_path)
        self.buffer: List[np.ndarray] = []

    def _load_model(self, model_path: str) -> SkeletonGCN:
        model = self.task.build_model(
            num_joints=self.num_joints,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        )

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def classify(self, keypoints_sequence: np.ndarray) -> Dict:
        """Classify action from a keypoint sequence.

        Args:
            keypoints_sequence: (T, J, C) skeleton sequence.

        Returns:
            Dict with 'action', 'confidence', and 'top5' predictions.
        """
        processed = self.task.preprocess_sequence(keypoints_sequence, self.sequence_length)
        tensor = torch.from_numpy(processed).unsqueeze(0).float().to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]

        top5_indices = probs.topk(5).indices.cpu().numpy()
        top5_probs = probs.topk(5).values.cpu().numpy()

        pred_idx = int(top5_indices[0])
        action_name = self.task.NTU_ACTIONS[pred_idx] if pred_idx < len(self.task.NTU_ACTIONS) else f"action_{pred_idx}"

        return {
            "action": action_name,
            "action_id": pred_idx,
            "confidence": float(top5_probs[0]),
            "top5": [
                {"action": self.task.NTU_ACTIONS[idx] if idx < len(self.task.NTU_ACTIONS) else f"action_{idx}",
                 "confidence": float(prob)}
                for idx, prob in zip(top5_indices, top5_probs)
            ],
        }

    def update(self, keypoints: np.ndarray) -> Optional[Dict]:
        """Add a single frame and classify when buffer is full.

        Use this for streaming/real-time action recognition.

        Args:
            keypoints: (J, C) keypoints for a single frame.

        Returns:
            Classification result when buffer is full, else None.
        """
        self.buffer.append(keypoints)

        if len(self.buffer) >= self.sequence_length:
            sequence = np.array(self.buffer[-self.sequence_length:])
            result = self.classify(sequence)
            self.buffer = self.buffer[-self.sequence_length // 2:]
            return result

        return None

    def reset(self):
        """Clear the frame buffer."""
        self.buffer.clear()
