"""Configuration system for FlashPose models and training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class PoseConfig:
    """Unified configuration for FlashPose models, training, and inference."""

    # Model
    model_name: str = "ViTPose"
    backbone: str = "vit_base"
    head: str = "heatmap"
    input_size: Tuple[int, int] = (256, 192)
    num_keypoints: int = 17
    heatmap_size: Tuple[int, int] = (64, 48)

    # Task
    task: str = "body_2d"
    dataset: str = "coco"

    # Training
    epochs: int = 210
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    amp: bool = True

    # LoRA
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    # Data
    train_ann: str = ""
    val_ann: str = ""
    img_dir: str = ""
    workers: int = 4
    flip_test: bool = True
    half_body_prob: float = 0.3

    # Augmentation
    scale_factor: float = 0.35
    rotation_factor: int = 40
    color_jitter: float = 0.3

    # Device
    device: str = "cuda"

    # Output
    save_dir: str = "workspace/pose"
    log_interval: int = 50
    save_interval: int = 10

    # 3D specific
    lift_from_2d: bool = False
    num_joints_3d: int = 17
    depth_range: Tuple[float, float] = (0.0, 10.0)

    # Action recognition
    num_classes: int = 60
    sequence_length: int = 64
    temporal_stride: int = 1

    # Extra
    pretrained: str = ""
    resume: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                result[k] = str(v)
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PoseConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


def get_config(
    model_name: str = "ViTPose",
    task: str = "body_2d",
    num_keypoints: int = 17,
    input_size: Tuple[int, int] = (256, 192),
    **kwargs,
) -> PoseConfig:
    """Create a PoseConfig with sensible defaults based on model and task.

    Args:
        model_name: Architecture name (ViTPose, HRNet, RTMPose).
        task: Task type (body_2d, body_3d, hand, face, wholebody, action).
        num_keypoints: Number of keypoints to predict.
        input_size: Input image size (H, W).
        **kwargs: Additional config overrides.

    Returns:
        PoseConfig instance.
    """
    heatmap_h = input_size[0] // 4
    heatmap_w = input_size[1] // 4

    defaults: Dict[str, Any] = {
        "model_name": model_name,
        "task": task,
        "num_keypoints": num_keypoints,
        "input_size": input_size,
        "heatmap_size": (heatmap_h, heatmap_w),
    }

    if task == "hand":
        defaults.update({"num_keypoints": 21, "input_size": (256, 256), "heatmap_size": (64, 64)})
    elif task == "face":
        defaults.update({"num_keypoints": 68, "input_size": (256, 256), "heatmap_size": (64, 64)})
    elif task == "wholebody":
        defaults.update({"num_keypoints": 133, "input_size": (384, 288), "heatmap_size": (96, 72)})
    elif task == "body_3d":
        defaults.update({"lift_from_2d": True, "num_joints_3d": 17})
    elif task == "action":
        defaults.update({"num_classes": 60, "sequence_length": 64})

    if model_name == "RTMPose":
        defaults["head"] = "simcc"
    elif model_name == "HRNet":
        defaults["head"] = "heatmap"
    else:
        defaults["head"] = "heatmap"

    defaults.update(kwargs)
    return PoseConfig.from_dict(defaults)


def load_yaml_config(path: str) -> PoseConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        PoseConfig instance.
    """
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "input_size" in data and isinstance(data["input_size"], list):
        data["input_size"] = tuple(data["input_size"])
    if "heatmap_size" in data and isinstance(data["heatmap_size"], list):
        data["heatmap_size"] = tuple(data["heatmap_size"])
    if "depth_range" in data and isinstance(data["depth_range"], list):
        data["depth_range"] = tuple(data["depth_range"])

    return PoseConfig.from_dict(data)
