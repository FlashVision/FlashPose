"""Benchmarking utilities for FlashPose model performance."""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from flashpose.cfg.config import PoseConfig
from flashpose.models.flashpose_model import FlashPose


class Benchmark:
    """Benchmark FlashPose model latency, throughput, and memory usage.

    Measures inference speed (FPS), per-frame latency, peak GPU memory,
    and model parameter count.

    Example:
        bench = Benchmark(model_path="best.pth")
        results = bench.run(num_iterations=100)
    """

    def __init__(
        self,
        model_path: str = "",
        config: Optional[PoseConfig] = None,
        device: str = "cuda",
        task: str = "body_2d",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task = task

        if config is None and model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            config = PoseConfig.from_dict(checkpoint.get("config", {}))

        self.config = config or PoseConfig(task=task)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> FlashPose:
        model = FlashPose(self.config)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def run(
        self,
        num_iterations: int = 100,
        batch_size: int = 1,
        warmup: int = 10,
    ) -> Dict[str, str]:
        """Run comprehensive benchmark.

        Args:
            num_iterations: Number of inference iterations to average.
            batch_size: Batch size for benchmarking.
            warmup: Number of warmup iterations (excluded from timing).

        Returns:
            Dict with benchmark results as formatted strings.
        """
        input_h, input_w = self.config.input_size
        dummy = torch.randn(batch_size, 3, input_h, input_w, device=self.device)

        for _ in range(warmup):
            self.model(dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        latencies = []
        for _ in range(num_iterations):
            if self.device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                self.model(dummy)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            else:
                start = time.perf_counter()
                self.model(dummy)
                latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        mean_latency = latencies.mean()
        std_latency = latencies.std()
        fps = 1000.0 / mean_latency * batch_size

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        results = {
            "Model": self.config.model_name,
            "Task": self.task,
            "Input Size": f"{input_h}x{input_w}",
            "Batch Size": str(batch_size),
            "Device": str(self.device),
            "Mean Latency": f"{mean_latency:.2f} ms",
            "Std Latency": f"{std_latency:.2f} ms",
            "FPS": f"{fps:.1f}",
            "Total Params": f"{total_params:,}",
            "Trainable Params": f"{trainable_params:,}",
        }

        if self.device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            results["Peak GPU Memory"] = f"{peak_mem:.1f} MB"

        return results

    def compare_models(
        self,
        model_names: list = None,
        num_iterations: int = 50,
    ) -> Dict[str, Dict[str, str]]:
        """Compare multiple model configurations.

        Args:
            model_names: List of model names to compare.
            num_iterations: Iterations per model.

        Returns:
            Nested dict of model_name -> metric_name -> value.
        """
        if model_names is None:
            model_names = ["ViTPose", "HRNet", "RTMPose"]

        from flashpose.cfg.config import get_config

        results = {}
        for name in model_names:
            cfg = get_config(model_name=name, task=self.task)
            model = FlashPose(cfg).to(self.device).eval()
            self.model = model
            self.config = cfg
            results[name] = self.run(num_iterations=num_iterations)

        return results
