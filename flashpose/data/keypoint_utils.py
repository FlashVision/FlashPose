"""Keypoint definitions, skeleton structures, and utility functions for pose estimation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

COCO_KEYPOINTS: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_SKELETON: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]

COCO_FLIP_PAIRS: List[Tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
]

MPII_KEYPOINTS: List[str] = [
    "right_ankle", "right_knee", "right_hip", "left_hip",
    "left_knee", "left_ankle", "pelvis", "thorax",
    "upper_neck", "head_top", "right_wrist", "right_elbow",
    "right_shoulder", "left_shoulder", "left_elbow", "left_wrist",
]

MPII_FLIP_PAIRS: List[Tuple[int, int]] = [
    (0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13),
]

H36M_KEYPOINTS: List[str] = [
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine",
    "neck", "head", "head_top", "left_shoulder",
    "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
]

H36M_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
]

HAND_KEYPOINTS: List[str] = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

HAND_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def flip_keypoints(
    keypoints: np.ndarray,
    image_width: int,
    flip_pairs: List[Tuple[int, int]],
) -> np.ndarray:
    """Flip keypoints horizontally for test-time augmentation.

    Args:
        keypoints: Array of shape (N, K, 2) or (K, 2) with (x, y) coordinates.
        image_width: Width of the image for mirroring x coordinates.
        flip_pairs: List of (left, right) index pairs to swap.

    Returns:
        Flipped keypoints with same shape as input.
    """
    flipped = keypoints.copy()
    single = flipped.ndim == 2
    if single:
        flipped = flipped[np.newaxis]

    flipped[..., 0] = image_width - 1 - flipped[..., 0]

    for left, right in flip_pairs:
        temp = flipped[:, left].copy()
        flipped[:, left] = flipped[:, right]
        flipped[:, right] = temp

    return flipped[0] if single else flipped


def half_body_transform(
    joints: np.ndarray,
    joints_vis: np.ndarray,
    num_joints: int,
    upper_body_ids: List[int] | None = None,
    lower_body_ids: List[int] | None = None,
) -> Tuple[np.ndarray, float]:
    """Compute center and scale for half-body augmentation.

    When enough visible joints exist in either the upper or lower body,
    crop around that subset to simulate partial occlusion.

    Args:
        joints: (K, 2) keypoint locations.
        joints_vis: (K,) visibility flags.
        num_joints: Total number of joints.
        upper_body_ids: Indices of upper body joints.
        lower_body_ids: Indices of lower body joints.

    Returns:
        Tuple of (center, scale) for the half-body crop.
    """
    if upper_body_ids is None:
        upper_body_ids = list(range(0, num_joints // 2))
    if lower_body_ids is None:
        lower_body_ids = list(range(num_joints // 2, num_joints))

    upper_joints = []
    lower_joints = []

    for idx in range(num_joints):
        if joints_vis[idx] > 0:
            if idx in upper_body_ids:
                upper_joints.append(joints[idx])
            else:
                lower_joints.append(joints[idx])

    if np.random.random() < 0.5 and len(upper_joints) > 2:
        selected = np.array(upper_joints)
    elif len(lower_joints) > 2:
        selected = np.array(lower_joints)
    elif len(upper_joints) > 2:
        selected = np.array(upper_joints)
    else:
        return joints.mean(axis=0), 1.0

    center = selected.mean(axis=0)
    extent = selected.max(axis=0) - selected.min(axis=0)
    scale = max(extent[0], extent[1]) * 1.5

    return center, max(scale, 1.0)


def keypoints_to_bbox(
    keypoints: np.ndarray,
    vis: np.ndarray | None = None,
    expansion: float = 1.25,
) -> np.ndarray:
    """Convert visible keypoints to a bounding box.

    Args:
        keypoints: (K, 2) array of (x, y) coordinates.
        vis: (K,) visibility flags; if None, all are treated as visible.
        expansion: Factor to expand the bounding box.

    Returns:
        Array [x1, y1, x2, y2] of the bounding box.
    """
    if vis is None:
        vis = np.ones(len(keypoints))

    valid = keypoints[vis > 0]
    if len(valid) == 0:
        return np.zeros(4)

    x_min, y_min = valid.min(axis=0)
    x_max, y_max = valid.max(axis=0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = (x_max - x_min) * expansion
    h = (y_max - y_min) * expansion

    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def oks_nms(
    keypoints_list: List[np.ndarray],
    scores: List[float],
    oks_threshold: float = 0.9,
    sigmas: np.ndarray | None = None,
    area: np.ndarray | None = None,
) -> List[int]:
    """Object Keypoint Similarity based NMS for pose estimation.

    Args:
        keypoints_list: List of (K, 3) arrays with (x, y, score).
        scores: Detection scores for each pose.
        oks_threshold: OKS threshold for suppression.
        sigmas: Per-keypoint standard deviations. Defaults to COCO sigmas.
        area: Bounding box areas for normalization.

    Returns:
        Indices of kept detections.
    """
    if sigmas is None:
        sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
        ])

    n = len(keypoints_list)
    if n == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        remaining = order[1:]
        if len(remaining) == 0:
            break

        suppress = []
        for j_idx, j in enumerate(remaining):
            kp_i = keypoints_list[i][:, :2]
            kp_j = keypoints_list[j][:, :2]
            vis_i = keypoints_list[i][:, 2]

            if area is not None:
                s = area[i]
            else:
                bbox = keypoints_to_bbox(kp_i)
                s = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)

            dist = np.sum((kp_i - kp_j) ** 2, axis=1)
            vars_k = (sigmas * 2) ** 2
            e = dist / (2 * s * vars_k + 1e-6)
            oks = np.mean(np.exp(-e) * (vis_i > 0))

            if oks > oks_threshold:
                suppress.append(j_idx)

        mask = np.ones(len(remaining), dtype=bool)
        mask[suppress] = False
        order = remaining[mask]

    return keep
