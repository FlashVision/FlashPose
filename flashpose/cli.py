"""FlashPose CLI — command-line interface for pose estimation, training, prediction, and export."""

import argparse
import sys


def _colored(text, color):
    """Simple ANSI color helper."""
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashPose", "bold") + f" v{_get_version()}")
    print(_colored("2D/3D human pose estimation, hand/face keypoints, action & gesture recognition", "blue"))
    print()


def _get_version():
    from flashpose import __version__
    return __version__


def cmd_version(args):
    """Print version info."""
    _print_banner()


def cmd_settings(args):
    """Print system settings and environment info."""
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:      {platform.python_version()}")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Machine:     {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  CUDA:        {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:        {mem:.1f} GB")
    else:
        print("  GPU:         None (CPU only)")
    print(f"  CPU cores:   {__import__('os').cpu_count()}")


def cmd_check(args):
    """Verify installation — imports, GPU, and basic inference."""
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashpose  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashpose package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashpose package: {e}")
        errors.append(str(e))

    try:
        from flashpose.engine import Trainer, Predictor, Exporter, Validator  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Predictor, Exporter, Validator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashpose.solutions import PoseEstimator, ActionClassifier, GestureRecognizer  # noqa: F401
        print(f"  {_colored('✓', 'green')} solutions (PoseEstimator, ActionClassifier, GestureRecognizer)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} solutions: {e}")
        errors.append(str(e))

    try:
        from flashpose.analytics import Benchmark  # noqa: F401
        print(f"  {_colored('✓', 'green')} analytics (Benchmark)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} analytics: {e}")
        errors.append(str(e))

    try:
        import torch
        from flashpose.cfg import get_config
        from flashpose.models.flashpose_model import FlashPose
        cfg = get_config(model_name="ViTPose", task="body_2d", num_keypoints=17)
        model = FlashPose(cfg)
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, cfg.input_size[0], cfg.input_size[1])
            model(x)
        print(f"  {_colored('✓', 'green')} model forward pass (ViTPose, body_2d)")
    except Exception as e:
        print(f"  {_colored('✗', 'red')} model forward pass: {e}")
        errors.append(str(e))

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (training will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashPose is ready.", "green"))


def cmd_train(args):
    """Train a FlashPose model."""
    from flashpose.engine.trainer import Trainer

    if args.config:
        from flashpose.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(config=cfg, device=args.device)
    else:
        kwargs = {
            "model_name": args.model_name,
            "task": args.task,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "save_dir": args.save_dir,
        }
        if args.lora:
            kwargs["lora"] = True
        if args.amp:
            kwargs["amp"] = True
        if args.lr:
            kwargs["lr"] = args.lr
        if args.workers is not None:
            kwargs["workers"] = args.workers
        if args.pretrained:
            kwargs["pretrained"] = args.pretrained
        trainer = Trainer(**kwargs)

    trainer.train()


def cmd_predict(args):
    """Run pose estimation on an image, video, or directory."""
    from flashpose.engine.predictor import Predictor

    predictor = Predictor(
        model_path=args.model,
        device=args.device,
        task=args.task,
    )

    results = predictor.predict(args.source, output_dir=args.output)
    n = len(results) if isinstance(results, list) else 1
    print(f"\n{_colored(f'Processed {n} frame(s)', 'green')}")


def cmd_estimate(args):
    """High-level pose estimation with visualization."""
    from flashpose.solutions.pose_estimator import PoseEstimator

    estimator = PoseEstimator(
        model_path=args.model,
        device=args.device,
        task=args.task,
    )

    results = estimator.run(args.source, output_dir=args.output, visualize=not args.no_vis)
    n = len(results) if isinstance(results, list) else 1
    print(f"\n{_colored(f'Estimated poses in {n} frame(s)', 'green')}")


def cmd_export(args):
    """Export model to ONNX."""
    from flashpose.engine.exporter import Exporter
    exporter = Exporter(model_path=args.model, task=args.task)
    path = exporter.export(output=args.output, simplify=args.simplify)
    print(f"\n{_colored('✓', 'green')} Exported: {path}")


def cmd_benchmark(args):
    """Benchmark pose estimation model."""
    from flashpose.analytics.benchmark import Benchmark

    bench = Benchmark(
        model_path=args.model,
        device=args.device,
        task=args.task,
    )
    results = bench.run(num_iterations=args.iterations, batch_size=args.batch_size)
    print(f"\n{_colored('Benchmark Results:', 'bold')}")
    for key, value in results.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        prog="flashpose",
        description="FlashPose: 2D/3D human pose estimation, hand/face keypoints, action & gesture recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashpose check                              Verify installation
  flashpose train --config configs/flashpose_body_17.yaml
  flashpose predict --model best.pth --source photo.jpg --task body_2d
  flashpose estimate --model best.pth --source video.mp4
  flashpose export --model best.pth --output model.onnx --simplify
  flashpose benchmark --model best.pth --task body_2d

Documentation: https://github.com/FlashVision/FlashPose
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show version info")

    # settings
    subparsers.add_parser("settings", help="Show system settings (Python, PyTorch, CUDA, GPU)")

    # check
    subparsers.add_parser("check", help="Verify installation and run health check")

    # train
    train_p = subparsers.add_parser("train", help="Train a FlashPose model")
    train_p.add_argument("--config", default=None, help="Path to YAML config")
    train_p.add_argument("--model-name", default="ViTPose", choices=["ViTPose", "HRNet", "RTMPose"])
    train_p.add_argument("--task", default="body_2d", choices=["body_2d", "body_3d", "hand", "face", "wholebody", "action"])
    train_p.add_argument("--epochs", type=int, default=210, help="Training epochs (default: 210)")
    train_p.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    train_p.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_p.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    train_p.add_argument("--save-dir", default="workspace/pose", help="Output directory")
    train_p.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    train_p.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    train_p.add_argument("--amp", action="store_true", help="Enable mixed precision (FP16)")
    train_p.add_argument("--pretrained", default="", help="Path to pretrained weights")

    # predict
    pred_p = subparsers.add_parser("predict", help="Run pose prediction on image/video/directory")
    pred_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    pred_p.add_argument("--source", required=True, help="Image path, video path, or directory")
    pred_p.add_argument("--task", default="body_2d", choices=["body_2d", "body_3d", "hand", "face", "wholebody"])
    pred_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    pred_p.add_argument("--output", default=None, help="Output directory for results")

    # estimate
    est_p = subparsers.add_parser("estimate", help="High-level pose estimation with visualization")
    est_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    est_p.add_argument("--source", required=True, help="Image path, video path, or directory")
    est_p.add_argument("--task", default="body_2d", choices=["body_2d", "body_3d", "hand", "face", "wholebody"])
    est_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    est_p.add_argument("--output", default=None, help="Output directory")
    est_p.add_argument("--no-vis", action="store_true", help="Disable visualization output")

    # export
    exp_p = subparsers.add_parser("export", help="Export model to ONNX format")
    exp_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    exp_p.add_argument("--task", default="body_2d", choices=["body_2d", "body_3d", "hand", "face", "wholebody"])
    exp_p.add_argument("--output", default="flashpose.onnx", help="Output path")
    exp_p.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Benchmark model performance")
    bench_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    bench_p.add_argument("--task", default="body_2d", choices=["body_2d", "body_3d", "hand", "face", "wholebody"])
    bench_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    bench_p.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    bench_p.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "train": cmd_train,
        "predict": cmd_predict,
        "estimate": cmd_estimate,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
