"""Model export engine for converting FlashPose to ONNX and other formats."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from flashpose.cfg.config import PoseConfig
from flashpose.models.flashpose_model import FlashPose
from flashpose.models.lora import merge_lora_weights


class Exporter:
    """Export FlashPose models to ONNX format for deployment.

    Supports LoRA weight merging, graph simplification, and
    dynamic axis configuration.
    """

    def __init__(
        self,
        model_path: str = "",
        config: Optional[PoseConfig] = None,
        task: str = "body_2d",
    ):
        self.model_path = model_path
        self.task = task

        if config is None and model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            config = PoseConfig.from_dict(checkpoint.get("config", {}))

        self.config = config or PoseConfig(task=task)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> nn.Module:
        model = FlashPose(self.config)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)

        model = merge_lora_weights(model)
        model.eval()
        return model

    def export(
        self,
        output: str = "flashpose.onnx",
        simplify: bool = False,
        opset_version: int = 13,
        dynamic_batch: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> str:
        """Export the model to ONNX format.

        Args:
            output: Output file path.
            simplify: Whether to simplify the ONNX graph using onnxsim.
            opset_version: ONNX opset version.
            dynamic_batch: Whether to enable dynamic batch size.
            input_size: Override input size (H, W). Uses config default if None.

        Returns:
            Path to the exported ONNX file.
        """
        if input_size is None:
            input_size = self.config.input_size

        h, w = input_size
        dummy_input = torch.randn(1, 3, h, w)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes["input"] = {0: "batch_size"}
            dynamic_axes["output"] = {0: "batch_size"}

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes if dynamic_axes else None,
            do_constant_folding=True,
        )

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                model_onnx = onnx.load(str(output_path))
                model_simplified, check = onnx_simplify(model_onnx)
                if check:
                    onnx.save(model_simplified, str(output_path))
                    print(f"  ONNX simplified successfully")
                else:
                    print(f"  ONNX simplification check failed, keeping original")
            except ImportError:
                print("  onnxsim not installed, skipping simplification")

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Model: {self.config.model_name} | Input: {h}x{w}")
        print(f"  Output: {output_path} ({file_size:.1f} MB)")

        return str(output_path)
