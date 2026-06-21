"""Data augmentation and transforms for pose estimation training."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: np.ndarray | None = None,
    inv: bool = False,
) -> np.ndarray:
    """Compute 2x3 affine transformation matrix for top-down pose crops.

    Args:
        center: (2,) center of the bounding box.
        scale: (2,) scale (width, height) in pixels.
        rot: Rotation angle in degrees.
        output_size: (width, height) of the output image.
        shift: (2,) translation offset as fraction of scale.
        inv: If True, compute the inverse transform.

    Returns:
        2x3 affine transformation matrix.
    """
    if shift is None:
        shift = np.array([0.0, 0.0])

    src_w = scale[0]
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180.0
    src_dir = _get_direction([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0] = center + scale * shift
    src[1] = center + src_dir + scale * shift
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2] = _get_third_point(src[0], src[1])
    dst[2] = _get_third_point(dst[0], dst[1])

    if inv:
        trans = cv2.getAffineTransform(dst.astype(np.float32), src.astype(np.float32))
    else:
        trans = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))

    return trans


def _get_direction(src_point: list, rot_rad: float) -> np.ndarray:
    """Rotate a point by given angle."""
    sin_val = np.sin(rot_rad)
    cos_val = np.cos(rot_rad)
    src_result = np.array([
        src_point[0] * cos_val - src_point[1] * sin_val,
        src_point[0] * sin_val + src_point[1] * cos_val,
    ])
    return src_result


def _get_third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Get the third point to uniquely define an affine transform from 2 points."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]])


def affine_transform_point(pt: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine transform to a single 2D point.

    Args:
        pt: (2,) point coordinates.
        trans: (2, 3) affine matrix.

    Returns:
        Transformed (2,) point.
    """
    new_pt = np.array([pt[0], pt[1], 1.0])
    new_pt = trans @ new_pt
    return new_pt[:2]


class TopDownAffine:
    """Top-down affine transformation for single-person pose estimation.

    Crops and resizes a person bounding box to the model input size using
    an affine transformation, and transforms keypoints accordingly.
    """

    def __init__(self, input_size: Tuple[int, int] = (256, 192)):
        self.input_size = input_size  # (H, W)

    def __call__(
        self,
        image: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        keypoints: np.ndarray | None = None,
        rotation: float = 0.0,
    ) -> dict:
        """Apply the affine crop transform.

        Args:
            image: Input BGR image (H, W, 3).
            center: (2,) bounding box center.
            scale: (2,) bounding box scale (w, h) in pixels.
            keypoints: (K, 2) or (K, 3) keypoint coordinates.
            rotation: Rotation angle in degrees.

        Returns:
            Dict with 'image', 'keypoints', 'transform' keys.
        """
        h, w = self.input_size
        trans = get_affine_transform(center, scale, rotation, (w, h))

        cropped = cv2.warpAffine(
            image, trans, (w, h), flags=cv2.INTER_LINEAR
        )

        transformed_kps = None
        if keypoints is not None:
            transformed_kps = keypoints.copy()
            for i in range(len(transformed_kps)):
                transformed_kps[i, :2] = affine_transform_point(
                    transformed_kps[i, :2], trans
                )

        return {
            "image": cropped,
            "keypoints": transformed_kps,
            "transform": trans,
        }


class PoseTransforms:
    """Compose augmentation pipeline for pose estimation training.

    Includes random scaling, rotation, flipping, and color jitter.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 192),
        scale_factor: float = 0.35,
        rotation_factor: int = 40,
        flip: bool = True,
        color_jitter: float = 0.3,
        half_body_prob: float = 0.3,
        train: bool = True,
    ):
        self.input_size = input_size
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip = flip
        self.color_jitter = color_jitter
        self.half_body_prob = half_body_prob
        self.train = train
        self.affine = TopDownAffine(input_size)

    def __call__(
        self,
        image: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        keypoints: np.ndarray,
        joints_vis: np.ndarray | None = None,
    ) -> dict:
        """Apply augmentation pipeline.

        Args:
            image: Input BGR image.
            center: (2,) bbox center.
            scale: (2,) bbox scale.
            keypoints: (K, 2) or (K, 3) keypoints.
            joints_vis: (K,) visibility per joint.

        Returns:
            Dict with 'image', 'keypoints', 'target_weight'.
        """
        s = scale.copy()
        r = 0.0

        if self.train:
            sf = self.scale_factor
            rf = self.rotation_factor

            s_aug = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            s = s * s_aug

            if np.random.random() < 0.6:
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

        result = self.affine(image, center, s, keypoints, rotation=r)

        img = result["image"]
        kps = result["keypoints"]

        if self.train and self.flip and np.random.random() < 0.5:
            img = img[:, ::-1].copy()
            if kps is not None:
                kps[:, 0] = self.input_size[1] - 1 - kps[:, 0]

        if self.train and self.color_jitter > 0:
            img = self._color_jitter(img)

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        target_weight = np.ones(len(kps), dtype=np.float32) if joints_vis is None else joints_vis.copy()

        return {
            "image": img,
            "keypoints": kps,
            "target_weight": target_weight,
        }

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness, contrast, and saturation jitter."""
        if np.random.random() < 0.5:
            factor = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        if np.random.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return image
