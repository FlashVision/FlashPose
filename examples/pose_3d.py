"""Example: 3D body pose estimation with lifting from 2D.

Demonstrates the 2D-to-3D pose lifting pipeline and Procrustes alignment.
"""

import numpy as np
import torch

from flashpose import FlashPose
from flashpose.cfg import get_config
from flashpose.tasks.body_3d import Body3DTask, PoseLifter
from flashpose.analytics.metrics import compute_mpjpe, compute_pa_mpjpe


def example_3d_model():
    """Run 3D pose model."""
    config = get_config(model_name="HRNet", task="body_3d")
    model = FlashPose(config)
    model.eval()

    print(f"3D Pose Model: {config.model_name}")
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Task: {config.task}")
    print(f"Lift from 2D: {config.lift_from_2d}")

    x = torch.randn(1, 3, *config.input_size)
    with torch.no_grad():
        output = model(x)

    print(f"Output keys: {list(output.keys())}")


def example_pose_lifting():
    """Demonstrate 2D-to-3D pose lifting."""
    task = Body3DTask()
    lifter = task.build_lifter(hidden_dim=1024, num_layers=4)
    lifter.eval()

    params = sum(p.numel() for p in lifter.parameters())
    print(f"Lifter parameters: {params:,}")

    batch_size = 8
    joints_2d = torch.randn(batch_size, 17, 2)

    with torch.no_grad():
        joints_3d = lifter(joints_2d)

    print(f"Input 2D: {joints_2d.shape} → Output 3D: {joints_3d.shape}")


def example_metrics():
    """Compute 3D pose evaluation metrics."""
    N = 100
    predictions = np.random.randn(N, 17, 3).astype(np.float32) * 100
    ground_truth = predictions + np.random.randn(N, 17, 3).astype(np.float32) * 20

    mpjpe = compute_mpjpe(predictions, ground_truth)
    pa_mpjpe = compute_pa_mpjpe(predictions, ground_truth)

    print(f"MPJPE: {mpjpe:.2f} mm")
    print(f"PA-MPJPE: {pa_mpjpe:.2f} mm")


def example_procrustes():
    """Demonstrate Procrustes alignment."""
    task = Body3DTask()

    gt = np.random.randn(17, 3) * 100
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    pred = (gt @ rotation.T) * 1.1 + np.array([50, -30, 20])

    error_before = np.linalg.norm(pred - gt, axis=1).mean()
    aligned = task.procrustes_align(pred, gt)
    error_after = np.linalg.norm(aligned - gt, axis=1).mean()

    print(f"Error before alignment: {error_before:.2f}")
    print(f"Error after alignment:  {error_after:.2f}")
    print(f"Improvement: {(1 - error_after/error_before)*100:.1f}%")


if __name__ == "__main__":
    print("=" * 50)
    print("FlashPose — 3D Pose Estimation Example")
    print("=" * 50)
    print()

    print("1. 3D Model")
    print("-" * 30)
    example_3d_model()
    print()

    print("2. Pose Lifting (2D → 3D)")
    print("-" * 30)
    example_pose_lifting()
    print()

    print("3. Evaluation Metrics")
    print("-" * 30)
    example_metrics()
    print()

    print("4. Procrustes Alignment")
    print("-" * 30)
    example_procrustes()
