"""ViTPose — Vision Transformer backbone for pose estimation.

Reference: "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation"
           (Xu et al., NeurIPS 2022)
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from flashpose.cfg.config import PoseConfig
from flashpose.registry import MODELS


class PatchEmbed(nn.Module):
    """Image to patch embedding via convolution."""

    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Vision Transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@MODELS.register("ViTPose")
class ViTPose(nn.Module):
    """Vision Transformer backbone for pose estimation.

    Implements a standard ViT encoder that outputs spatial feature maps
    suitable for heatmap-based keypoint detection.

    Variants:
        - vit_small: embed_dim=384, depth=12, heads=6
        - vit_base: embed_dim=768, depth=12, heads=12
        - vit_large: embed_dim=1024, depth=24, heads=16
    """

    CONFIGS = {
        "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
        "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    }

    def __init__(self, config: PoseConfig):
        super().__init__()
        backbone_name = config.backbone if config.backbone in self.CONFIGS else "vit_base"
        cfg = self.CONFIGS[backbone_name]

        self.embed_dim = cfg["embed_dim"]
        img_h, img_w = config.input_size
        self.patch_size = 16

        self.patch_embed_h = PatchEmbed(img_size=img_h, patch_size=self.patch_size, embed_dim=self.embed_dim)
        num_patches_h = img_h // self.patch_size
        num_patches_w = img_w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=cfg["num_heads"],
                mlp_ratio=4.0,
                drop=0.0,
                attn_drop=0.0,
            )
            for _ in range(cfg["depth"])
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        self.feat_h = num_patches_h
        self.feat_w = num_patches_w

        self._out_channels = self.embed_dim

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning spatial feature map.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Feature tensor of shape (B, embed_dim, H/16, W/16).
        """
        B = x.shape[0]

        x = self.patch_embed_h.proj(x)
        feat_h, feat_w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_embed_h.norm(x)

        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = self._interpolate_pos_embed(x.shape[1], feat_h, feat_w)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x = x.transpose(1, 2).reshape(B, self.embed_dim, feat_h, feat_w)
        return x

    def _interpolate_pos_embed(self, num_tokens: int, h: int, w: int) -> torch.Tensor:
        """Interpolate positional embeddings to handle variable input sizes."""
        pos = self.pos_embed.reshape(1, self.feat_h, self.feat_w, self.embed_dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        pos = pos.flatten(2).transpose(1, 2)
        return pos
