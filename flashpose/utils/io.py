"""I/O utilities for checkpoint loading, saving, and file management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the .pth checkpoint file.
        model: Model to load state_dict into.
        optimizer: Optimizer to load state_dict into.
        map_location: Device mapping for loading.

    Returns:
        Checkpoint dictionary with metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    if model is not None:
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    config: Optional[Dict] = None,
    **extra_info,
) -> str:
    """Save a training checkpoint.

    Args:
        path: Output path for the checkpoint.
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch number.
        best_metric: Best metric achieved so far.
        config: Configuration dictionary.
        **extra_info: Additional metadata to store.

    Returns:
        Path to saved checkpoint.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "best_metric": best_metric,
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    if config is not None:
        state["config"] = config

    state.update(extra_info)
    torch.save(state, path)

    return path


def auto_download(url: str, dest_dir: str, filename: Optional[str] = None) -> str:
    """Download a file if it doesn't exist locally.

    Args:
        url: URL to download from.
        dest_dir: Destination directory.
        filename: Override filename. If None, uses URL basename.

    Returns:
        Path to the downloaded file.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if filename is None:
        filename = os.path.basename(url.split("?")[0])

    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path):
        return dest_path

    try:
        import urllib.request
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

    return dest_path


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters by component.

    Args:
        model: PyTorch model.

    Returns:
        Dict with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }
