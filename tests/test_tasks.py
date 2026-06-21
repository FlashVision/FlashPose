"""Tests for FlashPose tasks."""

import numpy as np
import pytest

from flashpose.tasks.body_2d import Body2DTask
from flashpose.tasks.body_3d import Body3DTask
from flashpose.tasks.hand import HandTask
from flashpose.tasks.face import FaceTask
from flashpose.tasks.wholebody import WholeBodyTask
from flashpose.tasks.action import ActionTask
from flashpose.analytics.metrics import compute_pck, compute_ap, compute_mpjpe, compute_pa_mpjpe


class TestBody2DTask:
    def test_properties(self):
        task = Body2DTask()
        assert task.name == "body_2d"
        assert task.num_keypoints == 17

    def test_postprocess(self):
        task = Body2DTask()
        kps = np.random.randn(17, 2) * 50 + 128
        scores = np.random.rand(17)
        center = np.array([128.0, 128.0])
        scale = np.array([200.0, 200.0])

        result = task.postprocess(kps, scores, center, scale)
        assert "keypoints" in result
        assert "scores" in result
        assert "bbox" in result
        assert result["bbox"].shape == (4,)

    def test_evaluate(self):
        task = Body2DTask()
        preds = np.random.randn(10, 17, 2) * 100
        gts = preds + np.random.randn(10, 17, 2) * 5
        metrics = task.evaluate(preds, gts)
        assert "PCK@0.5" in metrics
        assert 0.0 <= metrics["PCK@0.5"] <= 1.0


class TestBody3DTask:
    def test_properties(self):
        task = Body3DTask()
        assert task.name == "body_3d"
        assert task.num_keypoints == 17

    def test_procrustes(self):
        task = Body3DTask()
        gt = np.random.randn(17, 3) * 100
        pred = gt + np.random.randn(17, 3) * 10
        aligned = task.procrustes_align(pred, gt)
        assert aligned.shape == (17, 3)

    def test_evaluate(self):
        task = Body3DTask()
        preds = np.random.randn(20, 17, 3) * 100
        gts = preds + np.random.randn(20, 17, 3) * 15
        metrics = task.evaluate(preds, gts)
        assert "MPJPE" in metrics
        assert "PA-MPJPE" in metrics
        assert metrics["PA-MPJPE"] <= metrics["MPJPE"]


class TestHandTask:
    def test_properties(self):
        task = HandTask()
        assert task.name == "hand"
        assert task.num_keypoints == 21

    def test_finger_count(self):
        task = HandTask()
        kps = np.random.randn(21, 2) * 50 + 128
        count = task.count_extended_fingers(kps)
        assert 0 <= count <= 5


class TestFaceTask:
    def test_properties(self):
        task = FaceTask()
        assert task.name == "face"
        assert task.num_keypoints == 68

    def test_ear(self):
        task = FaceTask()
        kps = np.random.randn(68, 2) * 20 + 128
        ear = task.compute_eye_aspect_ratio(kps)
        assert ear >= 0

    def test_evaluate(self):
        task = FaceTask()
        preds = np.random.randn(5, 68, 2) * 50 + 128
        gts = preds + np.random.randn(5, 68, 2) * 3
        metrics = task.evaluate(preds, gts)
        assert "NME_interocular" in metrics


class TestWholeBodyTask:
    def test_properties(self):
        task = WholeBodyTask()
        assert task.name == "wholebody"
        assert task.num_keypoints == 133

    def test_split(self):
        task = WholeBodyTask()
        kps = np.random.randn(133, 2)
        parts = task.split_predictions(kps)
        assert parts["body"].shape == (17, 2)
        assert parts["left_hand"].shape == (21, 2)
        assert parts["right_hand"].shape == (21, 2)


class TestActionTask:
    def test_properties(self):
        task = ActionTask()
        assert task.name == "action"
        assert task.num_classes == 60

    def test_preprocess(self):
        task = ActionTask()
        seq = np.random.randn(120, 17, 2).astype(np.float32)
        processed = task.preprocess_sequence(seq, target_length=64)
        assert processed.shape == (64, 17, 2)

    def test_preprocess_short(self):
        task = ActionTask()
        seq = np.random.randn(20, 17, 2).astype(np.float32)
        processed = task.preprocess_sequence(seq, target_length=64)
        assert processed.shape == (64, 17, 2)

    def test_evaluate(self):
        task = ActionTask()
        preds = np.random.randn(50, 60)
        gts = np.random.randint(0, 60, 50)
        metrics = task.evaluate(preds, gts)
        assert "accuracy" in metrics
        assert "top5_accuracy" in metrics


class TestMetrics:
    def test_pck(self):
        preds = np.random.randn(10, 17, 2) * 100
        gts = preds + np.random.randn(10, 17, 2) * 5
        pck = compute_pck(preds, gts, threshold=0.5)
        assert 0.0 <= pck <= 1.0

    def test_pck_perfect(self):
        preds = np.random.randn(10, 17, 2) * 100
        pck = compute_pck(preds, preds, threshold=0.1)
        assert pck == 1.0

    def test_ap(self):
        preds = np.random.randn(10, 17, 2) * 100
        gts = preds + np.random.randn(10, 17, 2) * 2
        ap = compute_ap(preds, gts)
        assert 0.0 <= ap <= 1.0

    def test_mpjpe(self):
        preds = np.random.randn(10, 17, 3) * 100
        gts = preds + np.random.randn(10, 17, 3) * 20
        mpjpe = compute_mpjpe(preds, gts)
        assert mpjpe > 0

    def test_pa_mpjpe(self):
        preds = np.random.randn(5, 17, 3) * 100
        gts = preds + np.random.randn(5, 17, 3) * 20
        pa = compute_pa_mpjpe(preds, gts)
        mpjpe = compute_mpjpe(preds, gts)
        assert pa <= mpjpe + 1e-6
