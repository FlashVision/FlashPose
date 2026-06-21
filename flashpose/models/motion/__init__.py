"""MotionBERT — Motion-centric BERT for 3D pose, mesh recovery, and action recognition.

Implements a transformer-based model that processes 2D pose sequences
to perform 3D pose lifting, mesh recovery, and action recognition.

Reference: "MotionBERT: A Unified Perspective on Learning Human Motion
Representations" (Zhu et al., ICCV 2023)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEmbedding(nn.Module):
    """Embed 2D pose sequences into transformer-compatible tokens."""

    def __init__(self, num_joints: int = 17, joint_dim: int = 2, embed_dim: int = 512):
        super().__init__()
        self.joint_embed = nn.Linear(joint_dim, embed_dim)
        self.joint_token = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim))
        nn.init.trunc_normal_(self.joint_token, std=0.02)

    def forward(self, pose_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_seq: (B, T, J, D) — batch of temporal pose sequences.

        Returns:
            (B, T, J, embed_dim) token embeddings.
        """
        tokens = self.joint_embed(pose_seq)
        tokens = tokens + self.joint_token
        return tokens


class TemporalAttention(nn.Module):
    """Multi-head attention across temporal dimension for each joint."""

    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, D)

        Returns:
            (B, T, J, D)
        """
        B, T, J, D = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * J, T, D)
        h = self.norm(x_reshaped)
        out = x_reshaped + self.attn(h, h, h, need_weights=False)[0]
        return out.reshape(B, J, T, D).permute(0, 2, 1, 3)


class SpatialAttention(nn.Module):
    """Multi-head attention across joint dimension for each timestep."""

    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, D)

        Returns:
            (B, T, J, D)
        """
        B, T, J, D = x.shape
        x_reshaped = x.reshape(B * T, J, D)
        h = self.norm(x_reshaped)
        out = x_reshaped + self.attn(h, h, h, need_weights=False)[0]
        return out.reshape(B, T, J, D)


class MotionBERTBlock(nn.Module):
    """Transformer block with temporal + spatial attention + FFN."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.temporal_attn = TemporalAttention(dim, num_heads, drop)
        self.spatial_attn = SpatialAttention(dim, num_heads, drop)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_attn(x)
        x = self.spatial_attn(x)
        B, T, J, D = x.shape
        x_flat = x.reshape(B * T * J, D)
        x_flat = x_flat + self.mlp(self.norm(x_flat))
        return x_flat.reshape(B, T, J, D)


class MotionBERT(nn.Module):
    """MotionBERT: unified model for 3D pose lifting, mesh recovery, and action recognition.

    Processes 2D pose sequences through temporal-spatial transformer blocks
    to produce task-specific outputs.

    Args:
        num_joints: Number of body joints.
        joint_dim: Input joint dimension (2 for 2D, 3 for 3D).
        embed_dim: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        max_seq_len: Maximum temporal sequence length.
        num_classes: Action classes (0 disables action head).
        drop_rate: Dropout rate.
    """

    def __init__(
        self,
        num_joints: int = 17,
        joint_dim: int = 2,
        embed_dim: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        max_seq_len: int = 243,
        num_classes: int = 0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.motion_embed = MotionEmbedding(num_joints, joint_dim, embed_dim)

        self.temporal_pos = nn.Parameter(torch.zeros(1, max_seq_len, 1, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        self.blocks = nn.ModuleList([
            MotionBERTBlock(embed_dim, num_heads, drop=drop_rate) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.pose3d_head = nn.Linear(embed_dim, 3)

        self.mesh_head = nn.Sequential(
            nn.Linear(embed_dim * num_joints, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 85),
        ) if num_joints >= 17 else None

        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes),
        ) if num_classes > 0 else None

    def forward_encoder(self, pose_2d: torch.Tensor) -> torch.Tensor:
        """Encode 2D pose sequence.

        Args:
            pose_2d: (B, T, J, 2) input 2D pose sequences.

        Returns:
            (B, T, J, D) encoded features.
        """
        B, T, J, _ = pose_2d.shape
        tokens = self.motion_embed(pose_2d)
        tokens = tokens + self.temporal_pos[:, :T]

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens.reshape(-1, self.embed_dim)).reshape(B, T, J, self.embed_dim)

    def forward_pose3d(self, features: torch.Tensor) -> torch.Tensor:
        """Predict 3D pose from encoded features.

        Args:
            features: (B, T, J, D)

        Returns:
            (B, T, J, 3) predicted 3D poses.
        """
        return self.pose3d_head(features)

    def forward_action(self, features: torch.Tensor) -> torch.Tensor:
        """Predict action class from encoded features.

        Args:
            features: (B, T, J, D)

        Returns:
            (B, num_classes) action logits.
        """
        pooled = features.mean(dim=(1, 2))
        return self.action_head(pooled)

    def forward(
        self,
        pose_2d: torch.Tensor,
        task: str = "pose3d",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with task selection.

        Args:
            pose_2d: (B, T, J, 2) input 2D pose sequences.
            task: One of 'pose3d', 'action', 'mesh', 'all'.

        Returns:
            Task-specific outputs.
        """
        features = self.forward_encoder(pose_2d)
        output: Dict[str, torch.Tensor] = {"features": features.mean(dim=(1, 2))}

        if task in ("pose3d", "all"):
            output["pose_3d"] = self.forward_pose3d(features)

        if task in ("action", "all") and self.action_head is not None:
            output["action_logits"] = self.forward_action(features)

        if task in ("mesh", "all") and self.mesh_head is not None:
            B, T = features.shape[:2]
            mid_frame = features[:, T // 2]
            mesh_input = mid_frame.reshape(B, -1)
            mesh_params = self.mesh_head(mesh_input)
            output["mesh_params"] = mesh_params

        return output

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        gt_pose3d: Optional[torch.Tensor] = None,
        gt_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses.

        Args:
            pred: Model predictions.
            gt_pose3d: Ground-truth 3D poses (B, T, J, 3).
            gt_action: Ground-truth action labels (B,).

        Returns:
            Loss dictionary.
        """
        losses = {}

        if gt_pose3d is not None and "pose_3d" in pred:
            losses["pose3d"] = F.l1_loss(pred["pose_3d"], gt_pose3d)

        if gt_action is not None and "action_logits" in pred:
            losses["action"] = F.cross_entropy(pred["action_logits"], gt_action)

        losses["total"] = sum(losses.values()) if losses else torch.tensor(0.0)
        return losses
