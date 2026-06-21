"""Backbone architectures for FlashPose."""

from flashpose.models.architectures.vitpose import ViTPose
from flashpose.models.architectures.hrnet import HRNet
from flashpose.models.architectures.rtmpose import RTMPose
from flashpose.models.architectures.simcc import SimCCHead

__all__ = ["ViTPose", "HRNet", "RTMPose", "SimCCHead"]
