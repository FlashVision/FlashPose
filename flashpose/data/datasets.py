"""Dataset implementations for COCO-Pose, MPII, and Human3.6M."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from flashpose.data.keypoint_utils import COCO_FLIP_PAIRS, MPII_FLIP_PAIRS
from flashpose.data.transforms import PoseTransforms
from flashpose.registry import DATASETS


@DATASETS.register("COCO")
class COCOPoseDataset(Dataset):
    """COCO Keypoint Detection dataset (17 keypoints).

    Expected structure:
        img_dir/
            <image_id>.jpg
        ann_file: COCO-format JSON with keypoint annotations.
    """

    NUM_KEYPOINTS = 17
    FLIP_PAIRS = COCO_FLIP_PAIRS

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        input_size: Tuple[int, int] = (256, 192),
        train: bool = True,
        transforms: Optional[PoseTransforms] = None,
    ):
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.input_size = input_size
        self.train = train
        self.transforms = transforms or PoseTransforms(input_size=input_size, train=train)
        self.data = self._load_annotations()

    def _load_annotations(self) -> List[Dict]:
        """Parse COCO keypoint annotations into a flat list of person instances."""
        if not os.path.exists(self.ann_file):
            return []

        with open(self.ann_file, "r") as f:
            coco = json.load(f)

        img_map = {img["id"]: img for img in coco.get("images", [])}
        records = []

        for ann in coco.get("annotations", []):
            if ann.get("num_keypoints", 0) == 0:
                continue
            if ann.get("iscrowd", 0):
                continue

            img_info = img_map.get(ann["image_id"])
            if img_info is None:
                continue

            keypoints = np.array(ann["keypoints"]).reshape(-1, 3).astype(np.float32)
            vis = keypoints[:, 2].copy()
            vis[vis > 0] = 1.0

            bbox = ann.get("bbox", [0, 0, 0, 0])
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            w = bbox[2]
            h = bbox[3]

            aspect_ratio = self.input_size[1] / self.input_size[0]
            if w > aspect_ratio * h:
                h = w / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio

            scale = np.array([w, h], dtype=np.float32) * 1.25

            records.append({
                "image_file": os.path.join(self.img_dir, img_info["file_name"]),
                "center": np.array([cx, cy], dtype=np.float32),
                "scale": scale,
                "keypoints": keypoints[:, :2],
                "joints_vis": vis,
                "bbox": np.array(bbox, dtype=np.float32),
            })

        return records

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.data[idx]

        img = cv2.imread(record["image_file"])
        if img is None:
            img = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)

        result = self.transforms(
            image=img,
            center=record["center"],
            scale=record["scale"],
            keypoints=record["keypoints"],
            joints_vis=record["joints_vis"],
        )

        target_heatmap = self._generate_heatmaps(result["keypoints"], result["target_weight"])

        return {
            "image": torch.from_numpy(result["image"]).float(),
            "target": torch.from_numpy(target_heatmap).float(),
            "target_weight": torch.from_numpy(result["target_weight"]).float(),
            "keypoints": torch.from_numpy(result["keypoints"]).float(),
        }

    def _generate_heatmaps(
        self,
        keypoints: np.ndarray,
        target_weight: np.ndarray,
        sigma: float = 2.0,
    ) -> np.ndarray:
        """Generate ground-truth Gaussian heatmaps for each keypoint.

        Args:
            keypoints: (K, 2) keypoint coordinates in crop space.
            target_weight: (K,) visibility weights.
            sigma: Gaussian sigma.

        Returns:
            (K, heatmap_h, heatmap_w) heatmap array.
        """
        heatmap_h = self.input_size[0] // 4
        heatmap_w = self.input_size[1] // 4
        num_joints = len(keypoints)

        heatmaps = np.zeros((num_joints, heatmap_h, heatmap_w), dtype=np.float32)
        feat_stride = np.array([self.input_size[1] / heatmap_w, self.input_size[0] / heatmap_h])

        for i in range(num_joints):
            if target_weight[i] < 0.5:
                continue

            mu_x = keypoints[i, 0] / feat_stride[0]
            mu_y = keypoints[i, 1] / feat_stride[1]

            ul = [int(mu_x - 3 * sigma), int(mu_y - 3 * sigma)]
            br = [int(mu_x + 3 * sigma + 1), int(mu_y + 3 * sigma + 1)]

            if ul[0] >= heatmap_w or ul[1] >= heatmap_h or br[0] < 0 or br[1] < 0:
                target_weight[i] = 0
                continue

            size = 6 * sigma + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0, y0 = 3 * sigma, 3 * sigma
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            g_x = max(0, -ul[0]), min(br[0], heatmap_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_h) - ul[1]
            img_x = max(0, ul[0]), min(br[0], heatmap_w)
            img_y = max(0, ul[1]), min(br[1], heatmap_h)

            heatmaps[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return heatmaps


@DATASETS.register("MPII")
class MPIIDataset(Dataset):
    """MPII Human Pose dataset (16 keypoints).

    Expected structure:
        img_dir/: MPII images
        ann_file: JSON with per-image annotations [{image, center, scale, joints, joints_vis}]
    """

    NUM_KEYPOINTS = 16
    FLIP_PAIRS = MPII_FLIP_PAIRS

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        input_size: Tuple[int, int] = (256, 256),
        train: bool = True,
        transforms: Optional[PoseTransforms] = None,
    ):
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.input_size = input_size
        self.train = train
        self.transforms = transforms or PoseTransforms(input_size=input_size, train=train)
        self.data = self._load_annotations()

    def _load_annotations(self) -> List[Dict]:
        if not os.path.exists(self.ann_file):
            return []

        with open(self.ann_file, "r") as f:
            annotations = json.load(f)

        records = []
        for ann in annotations:
            center = np.array(ann["center"], dtype=np.float32)
            scale = np.array([ann["scale"] * 200, ann["scale"] * 200], dtype=np.float32)
            joints = np.array(ann["joints"], dtype=np.float32)
            joints_vis = np.array(ann["joints_vis"], dtype=np.float32)

            records.append({
                "image_file": os.path.join(self.img_dir, ann["image"]),
                "center": center,
                "scale": scale,
                "keypoints": joints,
                "joints_vis": joints_vis,
            })

        return records

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.data[idx]

        img = cv2.imread(record["image_file"])
        if img is None:
            img = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)

        result = self.transforms(
            image=img,
            center=record["center"],
            scale=record["scale"],
            keypoints=record["keypoints"],
            joints_vis=record["joints_vis"],
        )

        heatmap_h = self.input_size[0] // 4
        heatmap_w = self.input_size[1] // 4
        target_heatmap = np.zeros((self.NUM_KEYPOINTS, heatmap_h, heatmap_w), dtype=np.float32)

        return {
            "image": torch.from_numpy(result["image"]).float(),
            "target": torch.from_numpy(target_heatmap).float(),
            "target_weight": torch.from_numpy(result["target_weight"]).float(),
            "keypoints": torch.from_numpy(result["keypoints"]).float(),
        }


@DATASETS.register("H36M")
class H36MDataset(Dataset):
    """Human3.6M dataset for 3D pose estimation (17 joints).

    Supports loading 2D projections and 3D ground-truth poses.
    """

    NUM_KEYPOINTS = 17

    def __init__(
        self,
        ann_file: str,
        img_dir: str = "",
        input_size: Tuple[int, int] = (256, 256),
        train: bool = True,
        transforms: Optional[PoseTransforms] = None,
    ):
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.input_size = input_size
        self.train = train
        self.transforms = transforms or PoseTransforms(input_size=input_size, train=train)
        self.data = self._load_annotations()

    def _load_annotations(self) -> List[Dict]:
        if not os.path.exists(self.ann_file):
            return []

        with open(self.ann_file, "r") as f:
            annotations = json.load(f)

        records = []
        for ann in annotations:
            joints_2d = np.array(ann["joints_2d"], dtype=np.float32)
            joints_3d = np.array(ann["joints_3d"], dtype=np.float32)
            center = np.array(ann.get("center", joints_2d.mean(axis=0)), dtype=np.float32)
            scale = np.array(ann.get("scale", [256.0, 256.0]), dtype=np.float32)

            records.append({
                "image_file": os.path.join(self.img_dir, ann.get("image", "")),
                "center": center,
                "scale": scale,
                "keypoints": joints_2d,
                "joints_3d": joints_3d,
                "joints_vis": np.ones(self.NUM_KEYPOINTS, dtype=np.float32),
            })

        return records

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.data[idx]

        img = cv2.imread(record["image_file"]) if record["image_file"] else None
        if img is None:
            img = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)

        result = self.transforms(
            image=img,
            center=record["center"],
            scale=record["scale"],
            keypoints=record["keypoints"],
            joints_vis=record["joints_vis"],
        )

        return {
            "image": torch.from_numpy(result["image"]).float(),
            "keypoints_2d": torch.from_numpy(result["keypoints"]).float(),
            "keypoints_3d": torch.from_numpy(record["joints_3d"]).float(),
            "target_weight": torch.from_numpy(result["target_weight"]).float(),
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader with sensible defaults for pose training.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False,
    )
