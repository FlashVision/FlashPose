"""Backbone architectures for FlashPose."""

from flashpose.models.architectures.vitpose import ViTPose
from flashpose.models.architectures.hrnet import HRNet
from flashpose.models.architectures.rtmpose import RTMPose
from flashpose.models.architectures.simcc import SimCCHead
from flashpose.models.architectures.dwpose import DWPose
from flashpose.models.architectures.densepose import DensePose

__all__ = ["ViTPose", "HRNet", "RTMPose", "SimCCHead", "DWPose", "DensePose"]
