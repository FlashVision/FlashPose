# Contributing to FlashPose

We welcome contributions! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/FlashVision/FlashPose.git
cd FlashPose
bash setup_env.sh
source .venv/bin/activate
pre-commit install
```

## Development Workflow

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run tests**: `pytest tests/ -v`
5. **Run linting**: `ruff check flashpose/`
6. **Commit** with clear messages
7. **Push** and open a Pull Request

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 120 characters
- Type hints encouraged for public APIs
- Docstrings in Google style

## Testing

```bash
pytest tests/ -v
pytest tests/test_models.py -v -k "test_forward"
```

## Adding a New Model Architecture

1. Create `flashpose/models/architectures/your_model.py`
2. Register with `@MODELS.register("YourModel")`
3. Ensure it has an `out_channels` property
4. Add tests in `tests/test_models.py`
5. Update documentation

## Adding a New Task

1. Create `flashpose/tasks/your_task.py`
2. Register with `@TASKS.register("your_task")`
3. Implement `evaluate()` method
4. Add tests in `tests/test_tasks.py`

## Reporting Issues

Please include:
- FlashPose version (`flashpose version`)
- Python version
- PyTorch version
- Hardware info (`flashpose settings`)
- Steps to reproduce
