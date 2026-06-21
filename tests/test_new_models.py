"""Tests for new FlashPose architectures and tasks."""

import pytest
import numpy as np
import torch


class TestDWPose:
    def test_forward(self):
        from flashpose.cfg.config import get_config
        from flashpose.models.architectures.dwpose import DWPose

        config = get_config(model_name="DWPose", backbone="dwpose_t", task="wholebody", num_keypoints=133, input_size=(256, 192))
        model = DWPose(config)
        model.eval()
        x = torch.randn(1, 3, 256, 192)
        with torch.no_grad():
            out = model(x)
        assert "simcc_x" in out
        assert "simcc_y" in out
        assert out["simcc_x"].shape[1] == 133

    def test_distillation_loss(self):
        from flashpose.cfg.config import get_config
        from flashpose.models.architectures.dwpose import DWPose

        config = get_config(model_name="DWPose", backbone="dwpose_t", task="body_2d", num_keypoints=17, input_size=(256, 192))
        student = DWPose(config)
        teacher = DWPose(config)
        student.train()
        teacher.eval()

        x = torch.randn(2, 3, 256, 192)
        s_out = student(x)
        with torch.no_grad():
            t_out = teacher(x)

        loss = student.compute_distillation_loss(s_out, t_out)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestSMPL:
    def test_smpl_layer(self):
        from flashpose.models.body import SMPLLayer

        smpl = SMPLLayer(num_betas=10, num_joints=23, num_vertices=100)
        betas = torch.zeros(2, 10)
        pose = torch.zeros(2, 23 * 3)
        out = smpl(betas, pose)
        assert "vertices" in out
        assert "joints" in out
        assert out["vertices"].shape == (2, 100, 3)

    def test_mesh_recovery(self):
        from flashpose.models.body import SMPLMeshRecovery

        model = SMPLMeshRecovery(num_betas=10, num_joints=23, feature_dim=512)
        images = torch.randn(2, 3, 224, 224)
        out = model(images)
        assert "vertices" in out
        assert "joints" in out
        assert "betas" in out
        assert out["betas"].shape == (2, 10)


class TestAnimalPose:
    def test_task(self):
        from flashpose.tasks.animal_pose import AnimalPoseTask

        task = AnimalPoseTask(species="quadruped")
        assert task.num_keypoints == 20
        assert task.name == "animal_pose_quadruped"

    def test_model(self):
        from flashpose.tasks.animal_pose import AnimalPoseModel

        model = AnimalPoseModel(species="quadruped", base_channels=16)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert "heatmaps" in out
        assert out["heatmaps"].shape[1] == 20

    def test_evaluate(self):
        from flashpose.tasks.animal_pose import AnimalPoseTask

        task = AnimalPoseTask()
        preds = np.random.randn(10, 20, 2)
        gt = preds + np.random.randn(10, 20, 2) * 0.01
        metrics = task.evaluate(preds, gt)
        assert "PCK@0.2" in metrics


class TestDensePose:
    def test_forward(self):
        from flashpose.models.architectures.densepose import DensePose

        model = DensePose(base_channels=16, fpn_channels=32, num_parts=25)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "part_logits" in out
        assert "u" in out
        assert "v" in out
        assert out["part_logits"].shape[1] == 25


class TestMotionBERT:
    def test_pose3d(self):
        from flashpose.models.motion import MotionBERT

        model = MotionBERT(num_joints=17, joint_dim=2, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
        pose_2d = torch.randn(2, 16, 17, 2)
        out = model(pose_2d, task="pose3d")
        assert "pose_3d" in out
        assert out["pose_3d"].shape == (2, 16, 17, 3)

    def test_action(self):
        from flashpose.models.motion import MotionBERT

        model = MotionBERT(num_joints=17, joint_dim=2, embed_dim=64, depth=2, num_heads=4, max_seq_len=32, num_classes=60)
        pose_2d = torch.randn(2, 16, 17, 2)
        out = model(pose_2d, task="action")
        assert "action_logits" in out
        assert out["action_logits"].shape == (2, 60)


class TestHandMediaPipe:
    def test_task(self):
        from flashpose.tasks.hand_mediapipe import HandMediaPipeTask

        task = HandMediaPipeTask()
        assert task.num_keypoints == 21
        assert task.name == "hand_mediapipe"

    def test_pipeline(self):
        from flashpose.tasks.hand_mediapipe import HandMediaPipePipeline

        model = HandMediaPipePipeline(landmark_size=64)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert "landmarks" in out
        assert out["landmarks"].shape == (2, 21, 3)
        assert "confidence" in out

    def test_gesture_recognition(self):
        from flashpose.tasks.hand_mediapipe import HandMediaPipeTask

        task = HandMediaPipeTask()
        landmarks = np.random.randn(21, 3)
        gesture = task.recognize_gesture(landmarks)
        assert isinstance(gesture, str)


class TestRegistration:
    def test_models_registered(self):
        from flashpose.registry import MODELS
        from flashpose.models.architectures import DWPose, DensePose
        assert "DWPose" in MODELS
        assert "DensePose" in MODELS

    def test_tasks_registered(self):
        from flashpose.registry import TASKS
        from flashpose.tasks import AnimalPoseTask, HandMediaPipeTask
        assert "animal_pose" in TASKS
        assert "hand_mediapipe" in TASKS
