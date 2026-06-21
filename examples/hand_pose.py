"""Example: Hand pose estimation and gesture recognition.

Demonstrates hand keypoint detection and rule-based gesture classification.
"""

import numpy as np
import torch

from flashpose import FlashPose
from flashpose.cfg import get_config
from flashpose.tasks.hand import HandTask
from flashpose.solutions.gesture_recognizer import GestureRecognizer


def example_hand_model():
    """Run hand pose model."""
    config = get_config(model_name="RTMPose", task="hand", num_keypoints=21, input_size=(256, 256))
    model = FlashPose(config)
    model.eval()

    print(f"Hand Model: {config.model_name}")
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Keypoints: {config.num_keypoints}")

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)

    print(f"Output keys: {list(output.keys())}")


def example_gesture_recognition():
    """Demonstrate gesture recognition from hand keypoints."""
    recognizer = GestureRecognizer()

    open_palm = np.array([
        [128, 300],  # wrist
        [100, 260], [80, 220], [60, 180], [50, 140],  # thumb
        [110, 180], [110, 140], [110, 100], [110, 60],  # index
        [130, 175], [130, 135], [130, 95], [130, 55],   # middle
        [150, 180], [150, 140], [150, 100], [150, 60],  # ring
        [170, 190], [170, 155], [170, 120], [170, 85],  # pinky
    ], dtype=np.float32)

    result = recognizer.recognize(open_palm)
    print(f"Open Palm → Gesture: {result['gesture']} (conf: {result['confidence']:.2f})")
    print(f"  Fingers extended: {result['num_extended']}")

    fist = np.array([
        [128, 300],  # wrist
        [110, 270], [100, 260], [110, 270], [120, 280],  # thumb (curled)
        [115, 260], [115, 270], [115, 275], [115, 280],  # index (curled)
        [130, 258], [130, 268], [130, 273], [130, 278],  # middle (curled)
        [145, 260], [145, 270], [145, 275], [145, 280],  # ring (curled)
        [160, 265], [160, 275], [160, 280], [160, 285],  # pinky (curled)
    ], dtype=np.float32)

    result = recognizer.recognize(fist)
    print(f"Fist → Gesture: {result['gesture']} (conf: {result['confidence']:.2f})")
    print(f"  Fingers extended: {result['num_extended']}")


def example_finger_analysis():
    """Analyze individual finger extension."""
    task = HandTask()

    keypoints = np.random.randn(21, 2).astype(np.float32) * 50 + 128
    extended = task.count_extended_fingers(keypoints)
    print(f"Extended fingers: {extended}/5")

    for finger in task.FINGER_GROUPS:
        is_ext = task.is_finger_extended(keypoints, finger)
        print(f"  {finger}: {'extended' if is_ext else 'curled'}")


if __name__ == "__main__":
    print("=" * 50)
    print("FlashPose — Hand Pose & Gesture Example")
    print("=" * 50)
    print()

    print("1. Hand Model")
    print("-" * 30)
    example_hand_model()
    print()

    print("2. Gesture Recognition")
    print("-" * 30)
    example_gesture_recognition()
    print()

    print("3. Finger Analysis")
    print("-" * 30)
    example_finger_analysis()
