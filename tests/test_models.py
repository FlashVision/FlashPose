"""Tests for FlashPose models and architectures."""

import pytest
import torch

from flashpose.cfg import get_config
from flashpose.models.flashpose_model import FlashPose
from flashpose.models.lora import apply_lora, merge_lora_weights, LoRALinear


class TestViTPose:
    def test_forward_body(self):
        config = get_config(model_name="ViTPose", task="body_2d", num_keypoints=17)
        model = FlashPose(config)
        model.eval()
        x = torch.randn(2, 3, 256, 192)
        with torch.no_grad():
            output = model(x)
        assert "heatmaps" in output
        assert output["heatmaps"].shape == (2, 17, 64, 48)

    def test_forward_hand(self):
        config = get_config(model_name="ViTPose", task="hand", num_keypoints=21, input_size=(256, 256))
        model = FlashPose(config)
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        assert "heatmaps" in output
        assert output["heatmaps"].shape[1] == 21


class TestHRNet:
    def test_forward(self):
        config = get_config(model_name="HRNet", task="body_2d", num_keypoints=17)
        model = FlashPose(config)
        model.eval()
        x = torch.randn(1, 3, 256, 192)
        with torch.no_grad():
            output = model(x)
        assert "heatmaps" in output
        assert output["heatmaps"].shape[1] == 17

    def test_w48_variant(self):
        config = get_config(model_name="HRNet", backbone="hrnet_w48", task="body_2d")
        model = FlashPose(config)
        assert model.num_parameters > 0


class TestRTMPose:
    def test_forward_simcc(self):
        config = get_config(model_name="RTMPose", task="body_2d", head="simcc")
        model = FlashPose(config)
        model.eval()
        x = torch.randn(1, 3, 256, 192)
        with torch.no_grad():
            output = model(x)
        assert "simcc_x" in output
        assert "simcc_y" in output

    def test_rtmpose_tiny(self):
        config = get_config(model_name="RTMPose", backbone="rtmpose_t", task="body_2d", head="simcc")
        model = FlashPose(config)
        model.eval()
        x = torch.randn(1, 3, 256, 192)
        with torch.no_grad():
            output = model(x)
        assert "simcc_x" in output


class TestLoRA:
    def test_apply_lora(self):
        config = get_config(model_name="ViTPose", task="body_2d")
        model = FlashPose(config)
        total_before = sum(p.numel() for p in model.parameters())

        model = apply_lora(model, rank=4, alpha=8.0)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < total_before
        assert trainable > 0

    def test_merge_lora(self):
        config = get_config(model_name="ViTPose", task="body_2d")
        model = FlashPose(config)
        model = apply_lora(model, rank=4)

        model.eval()
        x = torch.randn(1, 3, 256, 192)
        with torch.no_grad():
            out_before = model(x)

        model = merge_lora_weights(model)

        has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
        assert not has_lora

    def test_lora_linear(self):
        linear = torch.nn.Linear(64, 32)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 32)
