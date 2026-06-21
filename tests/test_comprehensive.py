"""Comprehensive test suite for FlashPose covering all architectures, heads,
body model, motion, tasks, metrics, visualization, and CLI."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from flashpose.cfg.config import PoseConfig, get_config
from flashpose.registry import MODELS, HEADS, TASKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return get_config(
        model_name="ViTPose",
        task="body_2d",
        num_keypoints=17,
        input_size=(32, 32),
    )


@pytest.fixture
def hrnet_config():
    return get_config(
        model_name="HRNet",
        task="body_2d",
        num_keypoints=17,
        input_size=(32, 32),
    )


@pytest.fixture
def rtmpose_config():
    return get_config(
        model_name="RTMPose",
        task="body_2d",
        num_keypoints=17,
        input_size=(32, 32),
        head="simcc",
    )


# ===================================================================
# Architecture: ViTPose
# ===================================================================


class TestViTPoseArchitecture:
    def test_forward_shape(self, small_config):
        from flashpose.models.architectures.vitpose import ViTPose

        model = ViTPose(small_config)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == model.embed_dim

    def test_out_channels(self, small_config):
        from flashpose.models.architectures.vitpose import ViTPose

        model = ViTPose(small_config)
        assert model.out_channels == model.embed_dim

    def test_vit_small_variant(self):
        cfg = get_config(model_name="ViTPose", backbone="vit_small", input_size=(32, 32))
        from flashpose.models.architectures.vitpose import ViTPose

        model = ViTPose(cfg)
        assert model.embed_dim == 384

    def test_vit_large_variant(self):
        cfg = get_config(model_name="ViTPose", backbone="vit_large", input_size=(32, 32))
        from flashpose.models.architectures.vitpose import ViTPose

        model = ViTPose(cfg)
        assert model.embed_dim == 1024

    def test_patch_embed(self):
        from flashpose.models.architectures.vitpose import PatchEmbed

        pe = PatchEmbed(img_size=32, patch_size=16, embed_dim=64)
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        assert out.shape == (2, 4, 64)

    def test_interpolate_pos_embed(self, small_config):
        from flashpose.models.architectures.vitpose import ViTPose

        model = ViTPose(small_config)
        model.eval()
        x = torch.randn(1, 3, 48, 48)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1


# ===================================================================
# Architecture: HRNet
# ===================================================================


class TestHRNetArchitecture:
    def test_forward_shape(self, hrnet_config):
        from flashpose.models.architectures.hrnet import HRNet

        model = HRNet(hrnet_config)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.dim() == 4

    def test_w32_variant(self):
        cfg = get_config(model_name="HRNet", backbone="hrnet_w32", input_size=(32, 32))
        from flashpose.models.architectures.hrnet import HRNet

        model = HRNet(cfg)
        assert model.out_channels == 32

    def test_w48_variant(self):
        cfg = get_config(model_name="HRNet", backbone="hrnet_w48", input_size=(32, 32))
        from flashpose.models.architectures.hrnet import HRNet

        model = HRNet(cfg)
        assert model.out_channels == 48

    def test_basic_block(self):
        from flashpose.models.architectures.hrnet import BasicBlock

        block = BasicBlock(32, 32)
        x = torch.randn(1, 32, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_basic_block_downsample(self):
        from flashpose.models.architectures.hrnet import BasicBlock

        block = BasicBlock(32, 64, stride=2)
        x = torch.randn(1, 32, 8, 8)
        out = block(x)
        assert out.shape == (1, 64, 4, 4)


# ===================================================================
# Architecture: RTMPose
# ===================================================================


class TestRTMPoseArchitecture:
    def test_forward_shape(self, rtmpose_config):
        from flashpose.models.architectures.rtmpose import RTMPose

        model = RTMPose(rtmpose_config)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.dim() == 4

    def test_rtmpose_tiny(self):
        cfg = get_config(model_name="RTMPose", backbone="rtmpose_t", input_size=(32, 32))
        from flashpose.models.architectures.rtmpose import RTMPose

        model = RTMPose(cfg)
        assert model.out_channels > 0

    def test_gated_attention_unit(self):
        from flashpose.models.architectures.rtmpose import GatedAttentionUnit

        gau = GatedAttentionUnit(64)
        x = torch.randn(2, 64, 4, 4)
        out = gau(x)
        assert out.shape == x.shape

    def test_csp_block(self):
        from flashpose.models.architectures.rtmpose import CSPBlock

        block = CSPBlock(64, 64, num_blocks=1)
        x = torch.randn(1, 64, 4, 4)
        out = block(x)
        assert out.shape == x.shape


# ===================================================================
# Architecture: DWPose
# ===================================================================


class TestDWPoseArchitecture:
    def test_forward_returns_simcc(self):
        from flashpose.models.architectures.dwpose import DWPose

        cfg = get_config(model_name="DWPose", backbone="dwpose_t", num_keypoints=17, input_size=(32, 32))
        model = DWPose(cfg)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "simcc_x" in out
        assert "simcc_y" in out
        assert out["simcc_x"].shape[1] == 17

    def test_head_aware_attention(self):
        from flashpose.models.architectures.dwpose import HeadAwareAttention

        attn = HeadAwareAttention(64, num_parts=3)
        x = torch.randn(2, 64, 4, 4)
        parts = attn(x)
        assert len(parts) == 3
        assert parts[0].shape == x.shape

    def test_distillation_loss(self):
        from flashpose.models.architectures.dwpose import DWPose

        cfg = get_config(model_name="DWPose", backbone="dwpose_t", num_keypoints=17, input_size=(32, 32))
        student = DWPose(cfg)
        teacher = DWPose(cfg)
        x = torch.randn(2, 3, 32, 32)
        s_out = student(x)
        with torch.no_grad():
            t_out = teacher(x)
        loss = student.compute_distillation_loss(s_out, t_out)
        assert loss.dim() == 0
        assert loss.item() >= 0


# ===================================================================
# Architecture: DensePose
# ===================================================================


class TestDensePoseArchitecture:
    def test_forward_outputs(self):
        from flashpose.models.architectures.densepose import DensePose

        model = DensePose(base_channels=16, fpn_channels=32, num_parts=25)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "part_logits" in out
        assert "u" in out
        assert "v" in out
        assert out["part_logits"].shape[1] == 25

    def test_get_dense_uv(self):
        from flashpose.models.architectures.densepose import DensePose

        model = DensePose(base_channels=16, fpn_channels=32, num_parts=5)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model.get_dense_uv(x)
        assert "part_labels" in out
        assert "u" in out
        assert "v" in out
        assert out["part_labels"].shape == (1, 32, 32)

    def test_compute_loss(self):
        from flashpose.models.architectures.densepose import DensePose

        model = DensePose(base_channels=16, fpn_channels=32, num_parts=5)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        h, w = out["part_logits"].shape[2:]
        gt_parts = torch.randint(0, 5, (2, h, w))
        gt_u = torch.rand(2, h, w)
        gt_v = torch.rand(2, h, w)
        losses = model.compute_loss(out, gt_parts, gt_u, gt_v)
        assert "total" in losses
        assert losses["total"].dim() == 0


# ===================================================================
# Heads: HeatmapHead, RegressionHead, SimCCHead
# ===================================================================


class TestHeatmapHead:
    def test_forward(self):
        from flashpose.heads.heatmap_head import HeatmapHead

        cfg = get_config(input_size=(32, 32), num_keypoints=17, heatmap_size=(8, 6))
        head = HeatmapHead(in_channels=64, num_keypoints=17, config=cfg, num_deconv_layers=1, num_deconv_filters=32)
        x = torch.randn(2, 64, 4, 4)
        out = head(x)
        assert "heatmaps" in out
        assert out["heatmaps"].shape == (2, 17, 8, 6)

    def test_decode_heatmaps(self):
        from flashpose.heads.heatmap_head import HeatmapHead

        heatmaps = torch.randn(2, 17, 8, 6)
        coords = HeatmapHead.decode_heatmaps(heatmaps, input_size=(32, 32))
        assert coords.shape == (2, 17, 3)
        assert (coords[:, :, 2] >= 0).all()


class TestRegressionHead:
    def test_forward(self):
        from flashpose.heads.regression_head import RegressionHead

        cfg = get_config(input_size=(32, 32), num_keypoints=17)
        head = RegressionHead(in_channels=64, num_keypoints=17, config=cfg)
        x = torch.randn(2, 64, 4, 4)
        out = head(x)
        assert "keypoints" in out
        assert out["keypoints"].shape == (2, 17, 2)
        assert (out["keypoints"] >= 0).all()
        assert (out["keypoints"] <= 1).all()

    def test_decode(self):
        from flashpose.heads.regression_head import RegressionHead

        cfg = get_config(input_size=(32, 32), num_keypoints=17)
        head = RegressionHead(in_channels=64, num_keypoints=17, config=cfg)
        kp = torch.rand(2, 17, 2)
        pixel = head.decode(kp)
        assert pixel.shape == (2, 17, 2)


class TestSimCCHead:
    def test_forward(self):
        from flashpose.models.architectures.simcc import SimCCHead

        cfg = get_config(input_size=(32, 32), num_keypoints=17)
        head = SimCCHead(in_channels=64, num_keypoints=17, config=cfg)
        x = torch.randn(2, 64, 4, 4)
        out = head(x)
        assert "simcc_x" in out
        assert "simcc_y" in out

    def test_decode(self):
        from flashpose.models.architectures.simcc import SimCCHead

        cfg = get_config(input_size=(32, 32), num_keypoints=17)
        head = SimCCHead(in_channels=64, num_keypoints=17, config=cfg)
        simcc_x = torch.randn(2, 17, head.x_size)
        simcc_y = torch.randn(2, 17, head.y_size)
        coords = head.decode(simcc_x, simcc_y, input_size=(32, 32))
        assert coords.shape == (2, 17, 2)


# ===================================================================
# SMPL body model
# ===================================================================


class TestSMPLBody:
    def test_rodrigues(self):
        from flashpose.models.body import SMPLLayer

        rot = torch.tensor([[0.1, 0.2, 0.3]])
        R = SMPLLayer.rodrigues(rot)
        assert R.shape == (1, 3, 3)
        det = torch.det(R[0])
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-4)

    def test_smpl_layer_forward(self):
        from flashpose.models.body import SMPLLayer

        smpl = SMPLLayer(num_betas=10, num_joints=23, num_vertices=100)
        betas = torch.zeros(2, 10)
        pose = torch.zeros(2, 23 * 3)
        out = smpl(betas, pose)
        assert "vertices" in out
        assert "joints" in out
        assert out["vertices"].shape == (2, 100, 3)

    def test_smpl_regressor(self):
        from flashpose.models.body import SMPLRegressor

        reg = SMPLRegressor(feature_dim=512, num_betas=10, num_joints=23)
        feat = torch.randn(2, 512)
        out = reg(feat)
        assert out["betas"].shape == (2, 10)
        assert out["pose"].shape == (2, 69)
        assert out["global_orient"].shape == (2, 3)

    def test_mesh_recovery_pipeline(self):
        from flashpose.models.body import SMPLMeshRecovery

        model = SMPLMeshRecovery(num_betas=10, num_joints=23, feature_dim=512)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "vertices" in out
        assert "joints" in out
        assert "betas" in out
        assert out["betas"].shape == (2, 10)


# ===================================================================
# MotionBERT
# ===================================================================


class TestMotionBERT:
    def test_pose3d_lifting(self):
        from flashpose.models.motion import MotionBERT

        model = MotionBERT(num_joints=17, joint_dim=2, embed_dim=32, depth=1, num_heads=4, max_seq_len=16)
        model.eval()
        pose_2d = torch.randn(2, 8, 17, 2)
        with torch.no_grad():
            out = model(pose_2d, task="pose3d")
        assert "pose_3d" in out
        assert out["pose_3d"].shape == (2, 8, 17, 3)

    def test_action_recognition(self):
        from flashpose.models.motion import MotionBERT

        model = MotionBERT(
            num_joints=17, joint_dim=2, embed_dim=32, depth=1, num_heads=4, max_seq_len=16, num_classes=60
        )
        model.eval()
        pose_2d = torch.randn(2, 8, 17, 2)
        with torch.no_grad():
            out = model(pose_2d, task="action")
        assert "action_logits" in out
        assert out["action_logits"].shape == (2, 60)

    def test_compute_loss(self):
        from flashpose.models.motion import MotionBERT

        model = MotionBERT(
            num_joints=17, joint_dim=2, embed_dim=32, depth=1, num_heads=4, max_seq_len=16, num_classes=10
        )
        pose_2d = torch.randn(1, 8, 17, 2)
        out = model(pose_2d, task="all")
        gt3d = torch.randn(1, 8, 17, 3)
        gt_action = torch.tensor([3])
        losses = model.compute_loss(out, gt_pose3d=gt3d, gt_action=gt_action)
        assert "pose3d" in losses
        assert "action" in losses
        assert losses["total"].dim() == 0


# ===================================================================
# Animal Pose
# ===================================================================


class TestAnimalPose:
    def test_task_quadruped(self):
        from flashpose.tasks.animal_pose import AnimalPoseTask

        task = AnimalPoseTask(species="quadruped")
        assert task.num_keypoints == 20
        assert task.name == "animal_pose_quadruped"

    def test_task_bird(self):
        from flashpose.tasks.animal_pose import AnimalPoseTask

        task = AnimalPoseTask(species="bird")
        assert task.num_keypoints == 14

    def test_model_forward(self):
        from flashpose.tasks.animal_pose import AnimalPoseModel

        model = AnimalPoseModel(species="quadruped", base_channels=8)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "heatmaps" in out
        assert out["heatmaps"].shape[1] == 20

    def test_decode_heatmaps(self):
        from flashpose.tasks.animal_pose import AnimalPoseModel

        model = AnimalPoseModel(species="quadruped", base_channels=8)
        heatmaps = torch.randn(2, 20, 8, 8)
        coords = model.decode_heatmaps(heatmaps)
        assert coords.shape == (2, 20, 2)

    def test_evaluate(self):
        from flashpose.tasks.animal_pose import AnimalPoseTask

        task = AnimalPoseTask()
        preds = np.random.randn(5, 20, 2)
        gt = preds + np.random.randn(5, 20, 2) * 0.01
        metrics = task.evaluate(preds, gt)
        assert "PCK@0.2" in metrics
        assert "mean_error" in metrics


# ===================================================================
# Hand MediaPipe
# ===================================================================


class TestHandMediaPipe:
    def test_palm_detector(self):
        from flashpose.tasks.hand_mediapipe import PalmDetector

        det = PalmDetector(in_channels=3, num_anchors=2)
        det.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = det(x)
        assert "cls_logits" in out
        assert "bbox" in out
        assert "handedness" in out

    def test_hand_landmark_model(self):
        from flashpose.tasks.hand_mediapipe import HandLandmarkModel

        model = HandLandmarkModel(in_channels=3, num_keypoints=21)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "landmarks" in out
        assert out["landmarks"].shape == (2, 21, 3)
        assert "confidence" in out

    def test_pipeline_forward(self):
        from flashpose.tasks.hand_mediapipe import HandMediaPipePipeline

        pipe = HandMediaPipePipeline(landmark_size=32)
        pipe.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = pipe(x)
        assert out["landmarks"].shape == (1, 21, 3)

    def test_gesture_recognition(self):
        from flashpose.tasks.hand_mediapipe import HandMediaPipeTask

        task = HandMediaPipeTask()
        landmarks = np.random.randn(21, 3)
        gesture = task.recognize_gesture(landmarks)
        assert isinstance(gesture, str)

    def test_count_fingers(self):
        from flashpose.tasks.hand_mediapipe import HandMediaPipeTask

        task = HandMediaPipeTask()
        landmarks = np.random.randn(21, 3)
        count = task.count_fingers(landmarks)
        assert 0 <= count <= 5

    def test_evaluate(self):
        from flashpose.tasks.hand_mediapipe import HandMediaPipeTask

        task = HandMediaPipeTask()
        preds = np.random.randn(5, 21, 3)
        gt = preds + np.random.randn(5, 21, 3) * 0.01
        metrics = task.evaluate(preds, gt)
        assert "mean_error" in metrics
        assert "PCK@0.1" in metrics


# ===================================================================
# Metrics: PCK, AP, OKS, MPJPE, PA-MPJPE
# ===================================================================


class TestPoseMetrics:
    def test_pck_perfect(self):
        from flashpose.analytics.metrics import compute_pck

        preds = np.random.randn(5, 17, 2) * 100
        pck = compute_pck(preds, preds, threshold=0.1)
        assert pck == 1.0

    def test_pck_range(self):
        from flashpose.analytics.metrics import compute_pck

        preds = np.random.randn(10, 17, 2) * 100
        gt = preds + np.random.randn(10, 17, 2) * 5
        pck = compute_pck(preds, gt, threshold=0.5)
        assert 0.0 <= pck <= 1.0

    def test_pck_empty(self):
        from flashpose.analytics.metrics import compute_pck

        assert compute_pck(np.array([]), np.array([]), threshold=0.5) == 0.0

    def test_ap(self):
        from flashpose.analytics.metrics import compute_ap

        preds = np.random.randn(10, 17, 2) * 100
        gt = preds + np.random.randn(10, 17, 2) * 2
        ap = compute_ap(preds, gt)
        assert 0.0 <= ap <= 1.0

    def test_mpjpe(self):
        from flashpose.analytics.metrics import compute_mpjpe

        preds = np.random.randn(10, 17, 3) * 100
        gt = preds + np.random.randn(10, 17, 3) * 20
        mpjpe = compute_mpjpe(preds, gt)
        assert mpjpe > 0

    def test_pa_mpjpe_leq_mpjpe(self):
        from flashpose.analytics.metrics import compute_pa_mpjpe, compute_mpjpe

        preds = np.random.randn(5, 17, 3) * 100
        gt = preds + np.random.randn(5, 17, 3) * 20
        pa = compute_pa_mpjpe(preds, gt)
        mpjpe = compute_mpjpe(preds, gt)
        assert pa <= mpjpe + 1e-4

    def test_pck_2d_input(self):
        from flashpose.analytics.metrics import compute_pck

        preds = np.random.randn(17, 2) * 50
        pck = compute_pck(preds, preds, threshold=0.5)
        assert pck == 1.0


# ===================================================================
# Skeleton Visualization (mocked cv2)
# ===================================================================


class TestSkeletonVisualization:
    def test_draw_skeleton(self):
        from flashpose.utils.visualize import draw_skeleton

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.random.rand(17, 2) * 60
        scores = np.ones(17)
        result = draw_skeleton(img, kps, scores, threshold=0.3, task="body_2d")
        assert result.shape == (64, 64, 3)

    def test_draw_hand(self):
        from flashpose.utils.visualize import draw_hand

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.random.rand(21, 2) * 60
        result = draw_hand(img, kps)
        assert result.shape == (64, 64, 3)

    def test_draw_face(self):
        from flashpose.utils.visualize import draw_face

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        kps = np.random.rand(68, 2) * 60
        result = draw_face(img, kps)
        assert result.shape == (64, 64, 3)


# ===================================================================
# CLI
# ===================================================================


class TestCLI:
    def test_main_no_command(self):
        from flashpose.cli import main

        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["flashpose"]):
                main()
        assert exc.value.code == 0

    def test_version_command(self, capsys):
        from flashpose.cli import main

        with patch("sys.argv", ["flashpose", "version"]):
            main()
        captured = capsys.readouterr()
        assert "FlashPose" in captured.out


# ===================================================================
# FlashPose unified model
# ===================================================================


class TestFlashPoseModel:
    def test_vitpose_heatmap(self):
        from flashpose.models.flashpose_model import FlashPose

        cfg = get_config(model_name="ViTPose", task="body_2d", num_keypoints=17, input_size=(32, 32))
        model = FlashPose(cfg)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "heatmaps" in out

    def test_hrnet_heatmap(self):
        from flashpose.models.flashpose_model import FlashPose

        cfg = get_config(model_name="HRNet", task="body_2d", num_keypoints=17, input_size=(32, 32))
        model = FlashPose(cfg)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "heatmaps" in out

    def test_rtmpose_simcc(self):
        from flashpose.models.flashpose_model import FlashPose

        cfg = get_config(model_name="RTMPose", task="body_2d", head="simcc", num_keypoints=17, input_size=(32, 32))
        model = FlashPose(cfg)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "simcc_x" in out

    def test_regression_head(self):
        from flashpose.models.flashpose_model import FlashPose

        cfg = get_config(model_name="ViTPose", task="body_2d", head="regression", num_keypoints=17, input_size=(32, 32))
        model = FlashPose(cfg)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "keypoints" in out

    def test_num_parameters(self):
        from flashpose.models.flashpose_model import FlashPose

        cfg = get_config(model_name="ViTPose", task="body_2d", input_size=(32, 32))
        model = FlashPose(cfg)
        assert model.num_parameters > 0

    def test_unknown_backbone_raises(self):
        cfg = get_config(model_name="UnknownModel", task="body_2d", input_size=(32, 32))
        from flashpose.models.flashpose_model import FlashPose

        with pytest.raises(ValueError, match="Unknown backbone"):
            FlashPose(cfg)


# ===================================================================
# Keypoint utilities
# ===================================================================


class TestKeypointUtils:
    def test_flip_keypoints(self):
        from flashpose.data.keypoint_utils import flip_keypoints, COCO_FLIP_PAIRS

        kps = np.random.rand(17, 2) * 128
        flipped = flip_keypoints(kps, image_width=128, flip_pairs=COCO_FLIP_PAIRS)
        assert flipped.shape == (17, 2)

    def test_keypoints_to_bbox(self):
        from flashpose.data.keypoint_utils import keypoints_to_bbox

        kps = np.array([[10, 10], [50, 50], [30, 30]], dtype=np.float64)
        bbox = keypoints_to_bbox(kps, expansion=1.0)
        assert len(bbox) == 4
        assert bbox[0] <= 10

    def test_oks_nms(self):
        from flashpose.data.keypoint_utils import oks_nms

        kps = [np.random.rand(17, 3) * 100 for _ in range(5)]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        keep = oks_nms(kps, scores, oks_threshold=0.9)
        assert len(keep) >= 1
        assert keep[0] == 0


# ===================================================================
# Config
# ===================================================================


class TestConfig:
    def test_get_config_defaults(self):
        cfg = get_config()
        assert cfg.model_name == "ViTPose"
        assert cfg.num_keypoints == 17

    def test_get_config_hand(self):
        cfg = get_config(task="hand")
        assert cfg.num_keypoints == 21
        assert cfg.input_size == (256, 256)

    def test_get_config_face(self):
        cfg = get_config(task="face")
        assert cfg.num_keypoints == 68

    def test_to_dict(self):
        cfg = get_config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "model_name" in d

    def test_from_dict(self):
        cfg = PoseConfig.from_dict({"model_name": "HRNet", "num_keypoints": 21})
        assert cfg.model_name == "HRNet"
        assert cfg.num_keypoints == 21


# ===================================================================
# Registry
# ===================================================================


class TestGlobalRegistries:
    def test_models_registered(self):
        assert "ViTPose" in MODELS
        assert "HRNet" in MODELS
        assert "RTMPose" in MODELS
        assert "DWPose" in MODELS
        assert "DensePose" in MODELS
        assert "FlashPose" in MODELS

    def test_heads_registered(self):
        assert "HeatmapHead" in HEADS
        assert "RegressionHead" in HEADS

    def test_tasks_registered(self):
        assert "animal_pose" in TASKS
        assert "hand_mediapipe" in TASKS
