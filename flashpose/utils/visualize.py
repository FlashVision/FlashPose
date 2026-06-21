"""Visualization utilities for skeleton drawing and pose overlay."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from flashpose.data.keypoint_utils import (
    COCO_SKELETON,
    HAND_SKELETON,
    H36M_SKELETON,
)

COCO_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170),
]

LIMB_COLORS = [
    (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
    (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
    (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
    (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255),
]


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.3,
    task: str = "body_2d",
    line_thickness: int = 2,
    point_radius: int = 4,
) -> np.ndarray:
    """Draw skeleton on an image.

    Args:
        image: BGR image (H, W, 3).
        keypoints: (K, 2) keypoint coordinates (x, y).
        scores: (K,) confidence scores per keypoint.
        threshold: Minimum confidence to draw a keypoint.
        task: Task type to determine skeleton structure.
        line_thickness: Thickness of limb lines.
        point_radius: Radius of keypoint circles.

    Returns:
        Annotated BGR image.
    """
    img = image.copy()

    skeleton = _get_skeleton(task)
    colors = COCO_COLORS

    for idx, (start, end) in enumerate(skeleton):
        if start >= len(keypoints) or end >= len(keypoints):
            continue
        if scores[start] < threshold or scores[end] < threshold:
            continue

        pt1 = tuple(keypoints[start].astype(int))
        pt2 = tuple(keypoints[end].astype(int))
        color = LIMB_COLORS[idx % len(LIMB_COLORS)]
        cv2.line(img, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    for i, (kp, score) in enumerate(zip(keypoints, scores)):
        if score < threshold:
            continue
        x, y = int(kp[0]), int(kp[1])
        color = colors[i % len(colors)]
        cv2.circle(img, (x, y), point_radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), point_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_hand(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    line_thickness: int = 2,
    point_radius: int = 3,
) -> np.ndarray:
    """Draw hand skeleton with finger-specific colors.

    Args:
        image: BGR image.
        keypoints: (21, 2) hand keypoints.
        scores: (21,) confidence scores.
        threshold: Min confidence to draw.
        line_thickness: Line thickness.
        point_radius: Point radius.

    Returns:
        Annotated image.
    """
    img = image.copy()
    if scores is None:
        scores = np.ones(len(keypoints))

    finger_colors = {
        "thumb": (255, 128, 0),
        "index": (0, 255, 0),
        "middle": (255, 0, 128),
        "ring": (0, 128, 255),
        "pinky": (255, 255, 0),
        "palm": (200, 200, 200),
    }

    finger_ranges = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]

    for (start_idx, end_idx), name in zip(finger_ranges, finger_names):
        color = finger_colors[name]
        prev = 0
        for idx in range(start_idx, end_idx + 1):
            if idx >= len(keypoints):
                break
            if scores[prev] >= threshold and scores[idx] >= threshold:
                pt1 = tuple(keypoints[prev].astype(int))
                pt2 = tuple(keypoints[idx].astype(int))
                cv2.line(img, pt1, pt2, color, line_thickness, cv2.LINE_AA)
            prev = idx

    for i, (kp, score) in enumerate(zip(keypoints, scores)):
        if score < threshold:
            continue
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), point_radius, (255, 255, 255), -1, cv2.LINE_AA)

    return img


def draw_face(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    point_radius: int = 1,
) -> np.ndarray:
    """Draw face landmarks.

    Args:
        image: BGR image.
        keypoints: (68, 2) face landmarks.
        scores: (68,) confidence scores.
        threshold: Min confidence.
        point_radius: Point radius.

    Returns:
        Annotated image.
    """
    img = image.copy()
    if scores is None:
        scores = np.ones(len(keypoints))

    group_colors = {
        "jaw": (0, 255, 0),
        "eyebrow": (255, 128, 0),
        "nose": (255, 0, 128),
        "eye": (0, 128, 255),
        "lip": (255, 0, 0),
    }

    for i, (kp, score) in enumerate(zip(keypoints, scores)):
        if score < threshold:
            continue
        x, y = int(kp[0]), int(kp[1])

        if i < 17:
            color = group_colors["jaw"]
        elif i < 27:
            color = group_colors["eyebrow"]
        elif i < 36:
            color = group_colors["nose"]
        elif i < 48:
            color = group_colors["eye"]
        else:
            color = group_colors["lip"]

        cv2.circle(img, (x, y), point_radius, color, -1, cv2.LINE_AA)

    return img


def draw_pose_3d(
    keypoints_3d: np.ndarray,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Render 3D pose as a 2D image using matplotlib projection.

    Args:
        keypoints_3d: (K, 3) 3D joint coordinates.
        skeleton: List of (start, end) bone connections.
        figsize: Figure size.

    Returns:
        Rendered image as numpy array.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # noqa: F401
    except ImportError:
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(img, "matplotlib required for 3D viz", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        return img

    if skeleton is None:
        skeleton = H36M_SKELETON

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], c="red", s=20)

    for start, end in skeleton:
        if start < len(keypoints_3d) and end < len(keypoints_3d):
            ax.plot(
                [keypoints_3d[start, 0], keypoints_3d[end, 0]],
                [keypoints_3d[start, 1], keypoints_3d[end, 1]],
                [keypoints_3d[start, 2], keypoints_3d[end, 2]],
                "b-", linewidth=2,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=-80, azim=-90)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _get_skeleton(task: str) -> List[Tuple[int, int]]:
    """Get skeleton connections for a given task."""
    skeletons = {
        "body_2d": COCO_SKELETON,
        "body_3d": H36M_SKELETON,
        "hand": HAND_SKELETON,
        "wholebody": COCO_SKELETON,
    }
    return skeletons.get(task, COCO_SKELETON)
