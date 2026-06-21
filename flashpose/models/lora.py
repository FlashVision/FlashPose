"""Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning of pose models."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer augmented with a low-rank adapter (LoRA).

    Freezes the original weight and adds a trainable low-rank decomposition:
        output = W_frozen @ x + (B @ A) @ x * (alpha / rank)
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the original linear layer for inference."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        merged.weight.data = self.original.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        return merged


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply LoRA adapters to Linear layers in a model.

    Freezes all base parameters and injects trainable low-rank adapters
    into the specified (or all) linear layers.

    Args:
        model: The model to adapt.
        rank: LoRA rank (low = fewer params, high = more capacity).
        alpha: LoRA scaling factor.
        dropout: Dropout rate for LoRA layers.
        target_modules: List of module name patterns to target.
            If None, applies to all nn.Linear layers.

    Returns:
        Modified model with LoRA adapters.
    """
    for param in model.parameters():
        param.requires_grad = False

    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is not None:
                if not any(t in name for t in target_modules):
                    continue

            lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            replacements.append((name, lora_layer))

    for name, lora_layer in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters back into base weights for deployment.

    After merging, the model has no LoRA overhead and can be exported normally.

    Args:
        model: Model with LoRA layers.

    Returns:
        Model with merged weights (no LoRA modules).
    """
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            merged = module.merge()
            replacements.append((name, merged))

    for name, merged_layer in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], merged_layer)

    for param in model.parameters():
        param.requires_grad = True

    return model
