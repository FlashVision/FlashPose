"""Prediction engine for running inference with FlashPose models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from flashpose.cfg.config import PoseConfig
from flashpose.models.flashpose_model import FlashPose
from flashpose.data.transforms import get_affine_transform, affine_transform_point
from flashpose.utils.visualize import draw_skeleton


class Predictor:
    """Run pose estimation inference on images, videos, or directories.

    Handles model loading, preprocessing, inference, postprocessing,
    and optional visualization.
    """

    def __init__(
        self,
        model_path: str = "",
        config: Optional[PoseConfig] = None,
        device: str = "cuda",
        task: str = "body_2d",
        conf_threshold: float = 0.3,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task = task
        self.conf_threshold = conf_threshold

        if config is None and model_path:
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
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        source: str,
        output_dir: Optional[str] = None,
        visualize: bool = True,
    ) -> List[Dict]:
        """Run pose estimation on a source (image, video, or directory).

        Args:
            source: Path to an image file, video file, or directory.
            output_dir: Directory to save visualized results.
            visualize: Whether to draw skeleton overlays.

        Returns:
            List of result dicts, one per frame/image.
        """
        source_path = Path(source)

        if source_path.is_dir():
            image_paths = sorted(
                p for p in source_path.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            )
            return [self._predict_image(str(p), output_dir, visualize) for p in image_paths]

        elif source_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            return self._predict_video(str(source_path), output_dir, visualize)

        else:
            return [self._predict_image(str(source_path), output_dir, visualize)]

    def _predict_image(self, image_path: str, output_dir: Optional[str], visualize: bool) -> Dict:
        """Run prediction on a single image."""
        img = cv2.imread(image_path)
        if img is None:
            return {"keypoints": np.array([]), "scores": np.array([]), "image_path": image_path}

        keypoints, scores = self._infer(img)

        result = {
            "keypoints": keypoints,
            "scores": scores,
            "image_path": image_path,
        }

        if visualize and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            vis_img = draw_skeleton(img, keypoints, scores, threshold=self.conf_threshold, task=self.task)
            out_path = os.path.join(output_dir, Path(image_path).name)
            cv2.imwrite(out_path, vis_img)
            result["output_path"] = out_path

        return result

    def _predict_video(self, video_path: str, output_dir: Optional[str], visualize: bool) -> List[Dict]:
        """Run prediction on a video file."""
        cap = cv2.VideoCapture(video_path)
        results = []

        writer = None
        if visualize and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, Path(video_path).stem + "_pose.mp4")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = self._infer(frame)
            results.append({"keypoints": keypoints, "scores": scores, "frame_idx": frame_idx})

            if writer is not None:
                vis_frame = draw_skeleton(frame, keypoints, scores, threshold=self.conf_threshold, task=self.task)
                writer.write(vis_frame)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        return results

    @torch.no_grad()
    def _infer(self, image: np.ndarray) -> tuple:
        """Run model inference on a single BGR image.

        Returns:
            Tuple of (keypoints: (K, 2), scores: (K,)).
        """
        h, w = image.shape[:2]
        input_h, input_w = self.config.input_size

        center = np.array([w / 2, h / 2], dtype=np.float32)
        scale = np.array([w, h], dtype=np.float32) * 1.25

        trans = get_affine_transform(center, scale, 0.0, (input_w, input_h))
        inp = cv2.warpAffine(image, trans, (input_w, input_h), flags=cv2.INTER_LINEAR)

        inp = inp.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = (inp - mean) / std
        inp = inp.transpose(2, 0, 1)

        tensor = torch.from_numpy(inp).unsqueeze(0).float().to(self.device)
        output = self.model(tensor)

        if "heatmaps" in output:
            from flashpose.heads.heatmap_head import HeatmapHead
            kps = HeatmapHead.decode_heatmaps(output["heatmaps"], (input_h, input_w))
            keypoints = kps[0, :, :2].cpu().numpy()
            scores = kps[0, :, 2].cpu().numpy()
        elif "simcc_x" in output:
            head = self.model.head
            kps = head.decode(output["simcc_x"], output["simcc_y"], (input_h, input_w))
            keypoints = kps[0].cpu().numpy()
            scores = np.ones(len(keypoints))
        elif "keypoints" in output:
            keypoints = output["keypoints"][0].cpu().numpy() * np.array([input_w, input_h])
            scores = np.ones(len(keypoints))
        else:
            keypoints = np.zeros((self.config.num_keypoints, 2))
            scores = np.zeros(self.config.num_keypoints)

        inv_trans = get_affine_transform(center, scale, 0.0, (input_w, input_h), inv=True)
        for i in range(len(keypoints)):
            keypoints[i] = affine_transform_point(keypoints[i], inv_trans)

        return keypoints, scores
