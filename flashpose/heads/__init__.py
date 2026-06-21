"""Prediction heads for FlashPose."""

from flashpose.heads.heatmap_head import HeatmapHead
from flashpose.heads.regression_head import RegressionHead
from flashpose.heads.simcc_head import SimCCHead

__all__ = ["HeatmapHead", "RegressionHead", "SimCCHead"]
