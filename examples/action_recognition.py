"""Example: Skeleton-based action recognition.

Demonstrates action classification from skeleton keypoint sequences
using a Spatial-Temporal Graph Convolutional Network.
"""

import numpy as np
import torch

from flashpose.solutions.action_classifier import ActionClassifier
from flashpose.tasks.action import ActionTask, SkeletonGCN


def example_action_model():
    """Build and run the action recognition model."""
    task = ActionTask()
    model = task.build_model(num_joints=17, in_channels=2, num_classes=60)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Action Model Parameters: {params:,}")
    print(f"Num Classes: 60 (NTU RGB+D)")

    batch_size = 4
    seq_length = 64
    x = torch.randn(batch_size, seq_length, 17, 2)

    with torch.no_grad():
        logits = model(x)

    probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)

    print(f"Input: {x.shape} → Output: {logits.shape}")
    print(f"Predicted actions: {[task.NTU_ACTIONS[p] for p in predictions.tolist()]}")


def example_streaming_classification():
    """Simulate real-time action recognition from streaming frames."""
    classifier = ActionClassifier(
        model_path="",
        device="cpu",
        num_joints=17,
        num_classes=60,
        sequence_length=32,
    )

    print("Simulating streaming action recognition...")
    print(f"Buffer length required: {classifier.sequence_length}")

    for frame_idx in range(40):
        keypoints = np.random.randn(17, 2).astype(np.float32) * 50 + 100
        result = classifier.update(keypoints)

        if result is not None:
            print(f"  Frame {frame_idx}: Action = {result['action']} (conf: {result['confidence']:.3f})")
            top3 = [(r["action"], f"{r['confidence']:.3f}") for r in result["top5"][:3]]
            print(f"    Top-3: {top3}")


def example_sequence_preprocessing():
    """Show how skeleton sequences are normalized."""
    task = ActionTask()

    raw_sequence = np.random.randn(120, 17, 2).astype(np.float32) * 100 + 300
    print(f"Raw sequence: shape={raw_sequence.shape}, range=[{raw_sequence.min():.1f}, {raw_sequence.max():.1f}]")

    processed = task.preprocess_sequence(raw_sequence, target_length=64)
    print(f"Processed: shape={processed.shape}, range=[{processed.min():.3f}, {processed.max():.3f}]")

    short_sequence = np.random.randn(20, 17, 2).astype(np.float32)
    padded = task.preprocess_sequence(short_sequence, target_length=64)
    print(f"Short (padded): {short_sequence.shape} → {padded.shape}")


def example_evaluation():
    """Evaluate action classification accuracy."""
    task = ActionTask()

    N = 200
    num_classes = 60
    predictions = np.random.randn(N, num_classes).astype(np.float32)
    ground_truth = np.random.randint(0, num_classes, N)

    metrics = task.evaluate(predictions, ground_truth)
    print(f"Top-1 Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']*100:.1f}%")


if __name__ == "__main__":
    print("=" * 50)
    print("FlashPose — Action Recognition Example")
    print("=" * 50)
    print()

    print("1. Action Model")
    print("-" * 30)
    example_action_model()
    print()

    print("2. Streaming Classification")
    print("-" * 30)
    example_streaming_classification()
    print()

    print("3. Sequence Preprocessing")
    print("-" * 30)
    example_sequence_preprocessing()
    print()

    print("4. Evaluation")
    print("-" * 30)
    example_evaluation()
