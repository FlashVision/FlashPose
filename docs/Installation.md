# Installation

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA 11.8+ (recommended for GPU acceleration)

## pip Install

```bash
git clone https://github.com/FlashVision/FlashPose.git
cd FlashPose
pip install -e .
```

## Full Install (with export, analytics, 3D support)

```bash
pip install -e ".[all]"
```

## Development Install

```bash
pip install -e ".[dev]"
pre-commit install
```

## Docker

```bash
cd docker
docker compose build
docker compose run flashpose check
```

## Environment Script

```bash
bash setup_env.sh
source .venv/bin/activate
```

## Verify

```bash
flashpose check
flashpose settings
```
