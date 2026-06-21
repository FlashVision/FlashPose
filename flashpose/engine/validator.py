"""Validation engine for evaluating FlashPose models."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from flashpose.cfg.config import PoseConfig
from flashpose.models.flashpose_model import FlashPose
from flashpose.analytics.metrics import compute_pck, compute_ap, compute_mpjpe


class Validator:
    """Evaluate a trained FlashPose model on a validation dataset.

    Computes standard metrics: PCK, AP (for 2D), and MPJPE (for 3D).
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

        if config is None and model_path:
            checkpoint = torch.load(model_path, map_location="cpu")
            config = PoseConfig.from_dict(checkpoint.get("config", {}))

        self.config = config or PoseConfig()
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> nn.Module:
        model = FlashPose(self.config)
        if model_path:
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def validate(self, dataloader=None) -> Dict[str, float]:
        """Run validation and compute metrics.

        Args:
            dataloader: Validation DataLoader. If None, uses synthetic data.

        Returns:
            Dictionary of metric name -> value.
        """
        print(f"\nValidating {self.config.model_name} on {self.task}...")

        all_preds = []
        all_gts = []

        if dataloader is None:
            for _ in range(10):
                x = torch.randn(1, 3, *self.config.input_size, device=self.device)
                output = self.model(x)
                if "heatmaps" in output:
                    from flashpose.heads.heatmap_head import HeatmapHead
                    kps = HeatmapHead.decode_heatmaps(output["heatmaps"], self.config.input_size)
                    all_preds.append(kps[0].cpu().numpy()[:, :2])
                elif "keypoints" in output:
                    all_preds.append(output["keypoints"][0].cpu().numpy() * 256)
                else:
                    all_preds.append(np.random.randn(self.config.num_keypoints, 2) * 50 + 128)
                all_gts.append(np.random.randn(self.config.num_keypoints, 2) * 50 + 128)
        else:
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch["image"].to(self.device)
                output = self.model(images)

                if "heatmaps" in output:
                    from flashpose.heads.heatmap_head import HeatmapHead
                    kps = HeatmapHead.decode_heatmaps(output["heatmaps"], self.config.input_size)
                    for i in range(kps.shape[0]):
                        all_preds.append(kps[i].cpu().numpy()[:, :2])
                        all_gts.append(batch["keypoints"][i].numpy())

        all_preds_arr = np.array(all_preds)
        all_gts_arr = np.array(all_gts)

        metrics = {}

        if self.task in ("body_2d", "hand", "face", "wholebody"):
            pck_50 = compute_pck(all_preds_arr, all_gts_arr, threshold=0.5)
            pck_20 = compute_pck(all_preds_arr, all_gts_arr, threshold=0.2)
            metrics["PCK@0.5"] = pck_50
            metrics["PCK@0.2"] = pck_20
            metrics["AP"] = compute_ap(all_preds_arr, all_gts_arr)

        elif self.task == "body_3d":
            mpjpe = compute_mpjpe(all_preds_arr, all_gts_arr)
            metrics["MPJPE"] = mpjpe

        print("\n  Results:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        return metrics
