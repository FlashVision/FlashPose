"""Example: 2D body pose estimation with FlashPose.

Demonstrates loading a model, running inference on an image,
and visualizing the predicted skeleton.
"""

import torch

from flashpose import FlashPose, Predictor, PoseEstimator
from flashpose.cfg import get_config
from flashpose.utils.visualize import draw_skeleton


def example_model_forward():
    """Basic model forward pass."""
    config = get_config(model_name="ViTPose", task="body_2d", num_keypoints=17)
    model = FlashPose(config)
    model.eval()

    print(f"Model: {config.model_name}")
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Input size: {config.input_size}")

    x = torch.randn(1, 3, *config.input_size)
    with torch.no_grad():
        output = model(x)

    if "heatmaps" in output:
        print(f"Output heatmaps: {output['heatmaps'].shape}")
    elif "keypoints" in output:
        print(f"Output keypoints: {output['keypoints'].shape}")


def example_pose_estimation():
    """High-level pose estimation on an image."""
    estimator = PoseEstimator(
        model_path="",
        device="cpu",
        task="body_2d",
        conf_threshold=0.3,
    )

    import numpy as np
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = estimator.estimate_single(dummy_image)

    print(f"Keypoints shape: {result['keypoints'].shape}")
    print(f"Scores shape: {result['scores'].shape}")
    print(f"Mean confidence: {result['scores'].mean():.3f}")


def example_different_backbones():
    """Compare different backbone architectures."""
    for model_name in ["ViTPose", "HRNet", "RTMPose"]:
        config = get_config(model_name=model_name, task="body_2d")
        model = FlashPose(config)
        print(f"{model_name}: {model.num_parameters:,} params")


if __name__ == "__main__":
    print("=" * 50)
    print("FlashPose — 2D Body Pose Example")
    print("=" * 50)
    print()

    print("1. Model Forward Pass")
    print("-" * 30)
    example_model_forward()
    print()

    print("2. Pose Estimation")
    print("-" * 30)
    example_pose_estimation()
    print()

    print("3. Backbone Comparison")
    print("-" * 30)
    example_different_backbones()
