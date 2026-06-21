# Quick Start

## 1. Run Pose Estimation

```python
from flashpose import PoseEstimator

estimator = PoseEstimator(model_path="best.pth", task="body_2d")
results = estimator.run("photo.jpg", output_dir="output/", visualize=True)

for result in results:
    print(f"Keypoints: {result['keypoints'].shape}")
    print(f"Mean confidence: {result['scores'].mean():.3f}")
```

## 2. Train a Model

```bash
flashpose train --config configs/flashpose_body_17.yaml --device cuda
```

Or in Python:

```python
from flashpose import Trainer
from flashpose.cfg import load_yaml_config

config = load_yaml_config("configs/flashpose_body_17.yaml")
trainer = Trainer(config=config)
trainer.train(train_loader, val_loader)
```

## 3. Export for Deployment

```bash
flashpose export --model workspace/pose/best.pth --output model.onnx --simplify
```

## 4. Benchmark Performance

```bash
flashpose benchmark --model best.pth --task body_2d --iterations 100
```
