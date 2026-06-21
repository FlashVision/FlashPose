"""FlashPose — Production-grade 2D/3D human pose estimation, hand/face keypoints, action & gesture recognition."""

__version__ = "1.0.0"

from flashpose.models.flashpose_model import FlashPose
from flashpose.engine.trainer import Trainer
from flashpose.engine.predictor import Predictor
from flashpose.engine.exporter import Exporter
from flashpose.solutions.pose_estimator import PoseEstimator
from flashpose.solutions.action_classifier import ActionClassifier
from flashpose.analytics.benchmark import Benchmark

__all__ = [
    "FlashPose",
    "Trainer",
    "Predictor",
    "Exporter",
    "PoseEstimator",
    "ActionClassifier",
    "Benchmark",
    "__version__",
]
