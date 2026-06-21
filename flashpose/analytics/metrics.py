"""Evaluation metrics for pose estimation: PCK, AP, MPJPE, PA-MPJPE."""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_pck(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
    bbox_sizes: Optional[np.ndarray] = None,
    normalize_by: str = "bbox",
) -> float:
    """Compute Percentage of Correct Keypoints (PCK).

    A keypoint is correct if its distance to ground truth is within
    a threshold fraction of a normalization factor (bbox size or torso).

    Args:
        predictions: (N, K, 2) predicted keypoint coordinates.
        ground_truth: (N, K, 2) ground truth coordinates.
        threshold: Distance threshold as fraction of normalization factor.
        bbox_sizes: (N,) normalization sizes. If None, uses bbox diagonal.
        normalize_by: Normalization method ('bbox' or 'torso').

    Returns:
        PCK score in [0, 1].
    """
    if len(predictions) == 0:
        return 0.0

    predictions = np.asarray(predictions, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    if predictions.ndim == 2:
        predictions = predictions[np.newaxis]
        ground_truth = ground_truth[np.newaxis]

    N, K, _ = predictions.shape
    distances = np.linalg.norm(predictions - ground_truth, axis=-1)  # (N, K)

    if bbox_sizes is not None:
        norms = np.asarray(bbox_sizes, dtype=np.float64).reshape(N, 1)
    else:
        mins = ground_truth.min(axis=1)
        maxs = ground_truth.max(axis=1)
        bbox_diag = np.linalg.norm(maxs - mins, axis=-1)
        norms = bbox_diag.reshape(N, 1)

    norms = np.maximum(norms, 1e-6)
    normalized_distances = distances / norms

    correct = (normalized_distances < threshold).astype(np.float64)
    pck = correct.mean()

    return float(pck)


def compute_ap(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    oks_threshold: float = 0.5,
    sigmas: Optional[np.ndarray] = None,
) -> float:
    """Compute Average Precision (AP) using Object Keypoint Similarity (OKS).

    Args:
        predictions: (N, K, 2) predicted keypoint coordinates.
        ground_truth: (N, K, 2) ground truth coordinates.
        oks_threshold: OKS threshold for a true positive.
        sigmas: (K,) per-keypoint standard deviations.

    Returns:
        AP score in [0, 1].
    """
    if len(predictions) == 0:
        return 0.0

    predictions = np.asarray(predictions, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    if predictions.ndim == 2:
        predictions = predictions[np.newaxis]
        ground_truth = ground_truth[np.newaxis]

    N, K, _ = predictions.shape

    if sigmas is None:
        if K == 17:
            sigmas = np.array([
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
            ])
        else:
            sigmas = np.full(K, 0.05)

    oks_scores = []
    for i in range(N):
        gt_kps = ground_truth[i]
        pred_kps = predictions[i]

        bbox = np.array([gt_kps.min(axis=0), gt_kps.max(axis=0)])
        area = max(np.prod(bbox[1] - bbox[0]), 1.0)

        dist_sq = np.sum((pred_kps - gt_kps) ** 2, axis=1)
        vars_k = (sigmas * 2) ** 2
        e = dist_sq / (2 * area * vars_k + 1e-6)
        oks = np.mean(np.exp(-e))
        oks_scores.append(oks)

    oks_scores = np.array(oks_scores)
    ap = float((oks_scores >= oks_threshold).mean())

    return ap


def compute_mpjpe(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Compute Mean Per Joint Position Error (MPJPE) for 3D pose.

    Args:
        predictions: (N, K, 3) or (N, K, 2) predicted joint positions.
        ground_truth: (N, K, 3) or (N, K, 2) ground truth positions.

    Returns:
        MPJPE in the same units as the input (typically mm).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    if predictions.ndim == 2:
        predictions = predictions[np.newaxis]
        ground_truth = ground_truth[np.newaxis]

    per_joint_error = np.linalg.norm(predictions - ground_truth, axis=-1)
    mpjpe = per_joint_error.mean()

    return float(mpjpe)


def compute_pa_mpjpe(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Compute Procrustes-Aligned MPJPE (PA-MPJPE).

    Aligns each prediction to its ground truth using Procrustes analysis
    before computing MPJPE. Measures shape accuracy independent of global pose.

    Args:
        predictions: (N, K, 3) predicted 3D joint positions.
        ground_truth: (N, K, 3) ground truth 3D positions.

    Returns:
        PA-MPJPE value.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    if predictions.ndim == 2:
        predictions = predictions[np.newaxis]
        ground_truth = ground_truth[np.newaxis]

    N = len(predictions)
    errors = []

    for i in range(N):
        pred = predictions[i]
        gt = ground_truth[i]

        aligned = _procrustes_align(pred, gt)
        error = np.linalg.norm(aligned - gt, axis=-1).mean()
        errors.append(error)

    return float(np.mean(errors))


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align source to target using Procrustes (rotation + scale + translation)."""
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    s_centered = source - mu_s
    t_centered = target - mu_t

    norm_s = np.linalg.norm(s_centered)
    norm_t = np.linalg.norm(t_centered)

    if norm_s < 1e-8 or norm_t < 1e-8:
        return source

    s_norm = s_centered / norm_s
    t_norm = t_centered / norm_t

    H = s_norm.T @ t_norm
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = norm_t * np.sum(S)
    aligned = scale * (s_norm @ R.T) + mu_t

    return aligned
