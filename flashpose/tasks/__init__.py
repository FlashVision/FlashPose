"""Task definitions for FlashPose."""

from flashpose.tasks.body_2d import Body2DTask
from flashpose.tasks.body_3d import Body3DTask
from flashpose.tasks.hand import HandTask
from flashpose.tasks.face import FaceTask
from flashpose.tasks.wholebody import WholeBodyTask
from flashpose.tasks.action import ActionTask

__all__ = ["Body2DTask", "Body3DTask", "HandTask", "FaceTask", "WholeBodyTask", "ActionTask"]
