"""Tests for the FlashPose registry system."""

import pytest

from flashpose.registry import Registry, MODELS, HEADS, DATASETS, TASKS


class TestRegistry:
    def test_register_decorator(self):
        reg = Registry("test")

        @reg.register("MyClass")
        class MyClass:
            def __init__(self, value=1):
                self.value = value

        assert "MyClass" in reg
        assert len(reg) == 1

    def test_register_without_name(self):
        reg = Registry("test")

        @reg.register
        class AutoNamed:
            pass

        assert "AutoNamed" in reg

    def test_build(self):
        reg = Registry("test")

        @reg.register("Builder")
        class Builder:
            def __init__(self, x=10):
                self.x = x

        obj = reg.build("Builder", x=42)
        assert obj.x == 42

    def test_build_not_found(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.build("NonExistent")

    def test_duplicate_registration(self):
        reg = Registry("test")

        @reg.register("Dup")
        class Dup1:
            pass

        with pytest.raises(KeyError, match="already registered"):
            @reg.register("Dup")
            class Dup2:
                pass

    def test_list(self):
        reg = Registry("test")

        @reg.register("B")
        class B:
            pass

        @reg.register("A")
        class A:
            pass

        assert reg.list() == ["A", "B"]

    def test_get(self):
        reg = Registry("test")

        @reg.register("Getter")
        class Getter:
            pass

        assert reg.get("Getter") is Getter

    def test_repr(self):
        reg = Registry("models")
        assert "models" in repr(reg)


class TestGlobalRegistries:
    def test_models_registry(self):
        assert "ViTPose" in MODELS
        assert "HRNet" in MODELS
        assert "RTMPose" in MODELS
        assert "FlashPose" in MODELS

    def test_heads_registry(self):
        assert "HeatmapHead" in HEADS
        assert "RegressionHead" in HEADS

    def test_datasets_registry(self):
        assert "COCO" in DATASETS
        assert "MPII" in DATASETS
        assert "H36M" in DATASETS

    def test_tasks_registry(self):
        assert "body_2d" in TASKS
        assert "body_3d" in TASKS
        assert "hand" in TASKS
        assert "face" in TASKS
        assert "wholebody" in TASKS
        assert "action" in TASKS
