"""Example: Benchmarking FlashPose models.

Compares inference speed, parameter count, and memory usage
across different architectures and configurations.
"""

import torch

from flashpose import FlashPose, Benchmark
from flashpose.cfg import get_config


def example_single_benchmark():
    """Benchmark a single model."""
    print("Benchmarking ViTPose-Base...")
    bench = Benchmark(model_path="", device="cpu", task="body_2d")
    results = bench.run(num_iterations=20, batch_size=1, warmup=5)

    for key, value in results.items():
        print(f"  {key}: {value}")


def example_architecture_comparison():
    """Compare all supported architectures."""
    print("\nArchitecture Comparison:")
    print("-" * 60)
    print(f"{'Model':<12} {'Params':>12} {'Input':>10} {'Head':>8}")
    print("-" * 60)

    configs = [
        ("ViTPose", "vit_base", "heatmap", (256, 192)),
        ("ViTPose", "vit_small", "heatmap", (256, 192)),
        ("HRNet", "hrnet_w32", "heatmap", (256, 192)),
        ("HRNet", "hrnet_w48", "heatmap", (256, 192)),
        ("RTMPose", "rtmpose_s", "simcc", (256, 192)),
        ("RTMPose", "rtmpose_m", "simcc", (256, 192)),
    ]

    for model_name, backbone, head, input_size in configs:
        cfg = get_config(model_name=model_name, backbone=backbone, head=head, input_size=input_size)
        model = FlashPose(cfg)
        params = model.num_parameters
        print(f"  {model_name:<10} {params:>12,} {f'{input_size[0]}x{input_size[1]}':>10} {head:>8}")


def example_task_comparison():
    """Compare model sizes across different tasks."""
    print("\nTask Comparison (ViTPose-Base):")
    print("-" * 50)

    tasks = [
        ("body_2d", 17),
        ("hand", 21),
        ("face", 68),
        ("wholebody", 133),
    ]

    for task, num_kps in tasks:
        cfg = get_config(model_name="ViTPose", task=task, num_keypoints=num_kps)
        model = FlashPose(cfg)
        params = model.num_parameters

        x = torch.randn(1, 3, *cfg.input_size)
        with torch.no_grad():
            output = model(x)

        out_shape = next(iter(output.values())).shape
        print(f"  {task:<12} kps={num_kps:<4} params={params:>12,} output={list(out_shape)}")


def example_lora_comparison():
    """Compare full model vs LoRA fine-tuning parameter counts."""
    from flashpose.models.lora import apply_lora

    print("\nLoRA Efficiency:")
    print("-" * 50)

    cfg = get_config(model_name="ViTPose", task="body_2d")
    model = FlashPose(cfg)
    full_params = model.num_parameters

    model_lora = FlashPose(cfg)
    model_lora = apply_lora(model_lora, rank=8, alpha=16.0)
    lora_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)

    print(f"  Full fine-tuning: {full_params:,} trainable params")
    print(f"  LoRA (rank=8):    {lora_params:,} trainable params")
    print(f"  Reduction:        {(1 - lora_params/full_params)*100:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("FlashPose — Model Benchmarking")
    print("=" * 60)
    print()

    example_single_benchmark()
    print()
    example_architecture_comparison()
    print()
    example_task_comparison()
    print()
    example_lora_comparison()
