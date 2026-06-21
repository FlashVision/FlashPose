"""SMPL/SMPL-X parametric body model for 3D mesh recovery.

Implements parametric human body mesh generation from shape and pose
parameters. Provides a differentiable layer that maps body model
parameters to 3D mesh vertices and joints.

Reference: "SMPL: A Skinned Multi-Person Linear Model" (Loper et al., SIGGRAPH Asia 2015)
           "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image" (SMPL-X)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMPLLayer(nn.Module):
    """Differentiable SMPL body model layer.

    Maps body shape (beta) and pose (theta) parameters to 3D mesh
    vertices and joints via linear blend skinning.

    Args:
        num_betas: Number of shape parameters.
        num_joints: Number of body joints (23 for SMPL, 21 for SMPL-X body).
        num_vertices: Number of mesh vertices (6890 for SMPL).
    """

    def __init__(self, num_betas: int = 10, num_joints: int = 23, num_vertices: int = 6890):
        super().__init__()
        self.num_betas = num_betas
        self.num_joints = num_joints
        self.num_vertices = num_vertices

        self.register_buffer("v_template", torch.zeros(num_vertices, 3))
        self.register_buffer("shapedirs", torch.zeros(num_vertices, 3, num_betas))
        self.register_buffer("posedirs", torch.zeros(num_vertices * 3, (num_joints) * 9))
        self.register_buffer("J_regressor", torch.zeros(num_joints + 1, num_vertices))
        self.register_buffer("lbs_weights", torch.zeros(num_vertices, num_joints + 1))
        self.register_buffer("kintree_table", torch.zeros(2, num_joints + 1, dtype=torch.long))

        self._init_default_params()

    def _init_default_params(self):
        nn.init.normal_(self.v_template, 0, 0.01)
        nn.init.normal_(self.shapedirs, 0, 0.001)
        nn.init.normal_(self.J_regressor, 0, 0.01)
        self.lbs_weights.fill_(1.0 / (self.num_joints + 1))

    @staticmethod
    def rodrigues(rot_vec: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle rotation to rotation matrix via Rodrigues formula."""
        angle = rot_vec.norm(dim=-1, keepdim=True).unsqueeze(-1)
        axis = rot_vec / (angle.squeeze(-1) + 1e-8)

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        K = torch.zeros(*rot_vec.shape[:-1], 3, 3, device=rot_vec.device, dtype=rot_vec.dtype)
        K[..., 0, 1] = -axis[..., 2]
        K[..., 0, 2] = axis[..., 1]
        K[..., 1, 0] = axis[..., 2]
        K[..., 1, 2] = -axis[..., 0]
        K[..., 2, 0] = -axis[..., 1]
        K[..., 2, 1] = axis[..., 0]

        eye = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype).expand_as(K)
        R = eye + sin * K + (1 - cos) * (K @ K)
        return R

    def forward(
        self,
        betas: torch.Tensor,
        pose: torch.Tensor,
        global_orient: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: parameters -> mesh vertices + joints.

        Args:
            betas: Shape parameters (B, num_betas).
            pose: Body pose in axis-angle (B, num_joints * 3).
            global_orient: Global rotation (B, 3). Defaults to zero.
            transl: Translation (B, 3). Defaults to zero.

        Returns:
            Dict with 'vertices' (B, V, 3), 'joints' (B, J, 3).
        """
        B = betas.shape[0]
        device = betas.device

        if global_orient is None:
            global_orient = torch.zeros(B, 3, device=device)
        if transl is None:
            transl = torch.zeros(B, 3, device=device)

        v_shaped = self.v_template.unsqueeze(0) + torch.einsum("ijk,bk->bij", self.shapedirs, betas)

        full_pose = torch.cat([global_orient, pose], dim=-1)
        num_rotations = full_pose.shape[-1] // 3
        rot_mats = self.rodrigues(full_pose.reshape(B, num_rotations, 3))

        pose_feature = (rot_mats[:, 1:] - torch.eye(3, device=device)).reshape(B, -1)
        min_dim = min(pose_feature.shape[-1], self.posedirs.shape[-1])
        v_posed = v_shaped + torch.einsum("ij,bj->bi", self.posedirs[:, :min_dim], pose_feature[:, :min_dim]).reshape(B, self.num_vertices, 3)

        joints = torch.einsum("jv,bvc->bjc", self.J_regressor, v_posed)

        vertices = self._lbs(v_posed, rot_mats, joints)
        vertices = vertices + transl.unsqueeze(1)
        joints = torch.einsum("jv,bvc->bjc", self.J_regressor, vertices)

        return {"vertices": vertices, "joints": joints}

    def _lbs(self, v_posed: torch.Tensor, rot_mats: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """Linear Blend Skinning."""
        B = v_posed.shape[0]
        num_joints = rot_mats.shape[1]

        transforms = torch.zeros(B, num_joints, 4, 4, device=v_posed.device, dtype=v_posed.dtype)
        transforms[:, :, :3, :3] = rot_mats
        transforms[:, :, :3, 3] = joints[:, :num_joints]
        transforms[:, :, 3, 3] = 1.0

        W = self.lbs_weights[:, :num_joints]
        T = torch.einsum("vj,bjmn->bvmn", W, transforms)

        v_homo = F.pad(v_posed, (0, 1), value=1.0)
        v_transformed = torch.einsum("bvmn,bvn->bvm", T, v_homo)
        return v_transformed[:, :, :3]


class SMPLRegressor(nn.Module):
    """Regresses SMPL parameters from image features.

    Predicts body shape (beta), pose (theta), and camera parameters
    from extracted image features.

    Args:
        feature_dim: Input feature dimension.
        num_betas: Shape parameter count.
        num_joints: Body joint count.
    """

    def __init__(self, feature_dim: int = 2048, num_betas: int = 10, num_joints: int = 23):
        super().__init__()
        self.num_betas = num_betas
        self.num_joints = num_joints

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.fc_betas = nn.Linear(1024, num_betas)
        self.fc_pose = nn.Linear(1024, num_joints * 3)
        self.fc_orient = nn.Linear(1024, 3)
        self.fc_cam = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.fc_betas.weight, gain=0.01)
        nn.init.xavier_uniform_(self.fc_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.fc_orient.weight, gain=0.01)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.fc_layers(features)
        betas = self.fc_betas(x)
        pose = self.fc_pose(x)
        orient = self.fc_orient(x)
        cam = self.fc_cam(x)
        return {"betas": betas, "pose": pose, "global_orient": orient, "cam": cam}


class ResNetBackbone(nn.Module):
    """Lightweight ResNet-like backbone for mesh recovery."""

    def __init__(self, out_dim: int = 2048):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            self._make_layer(64, 64, 3),
            self._make_layer(64, 128, 4, stride=2),
            self._make_layer(128, 256, 6, stride=2),
            self._make_layer(256, 512, 3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, out_dim) if out_dim != 512 else nn.Identity()
        self._out_dim = out_dim

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [self._block(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(self._block(out_ch, out_ch))
        return nn.Sequential(*layers)

    @staticmethod
    def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        class Block(nn.Module):
            def __init__(self, in_ch, out_ch, stride, downsample):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.downsample = downsample

            def forward(self, x):
                identity = x
                out = F.relu(self.bn1(self.conv1(x)), inplace=True)
                out = self.bn2(self.conv2(out))
                if self.downsample is not None:
                    identity = self.downsample(x)
                return F.relu(out + identity, inplace=True)

        return Block(in_ch, out_ch, stride, downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class SMPLMeshRecovery(nn.Module):
    """End-to-end SMPL mesh recovery from a single image.

    Pipeline: Image -> Backbone -> Regressor -> SMPL Layer -> Mesh

    Args:
        num_betas: Number of shape coefficients.
        num_joints: Number of body joints.
        feature_dim: Backbone output dimension.
    """

    def __init__(self, num_betas: int = 10, num_joints: int = 23, feature_dim: int = 2048):
        super().__init__()
        self.backbone = ResNetBackbone(feature_dim)
        self.regressor = SMPLRegressor(feature_dim, num_betas, num_joints)
        self.smpl = SMPLLayer(num_betas, num_joints)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Recover 3D mesh from images.

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Dict with 'vertices', 'joints', 'betas', 'pose', 'cam'.
        """
        features = self.backbone(images)
        params = self.regressor(features)
        mesh = self.smpl(
            betas=params["betas"],
            pose=params["pose"],
            global_orient=params["global_orient"],
        )
        return {
            "vertices": mesh["vertices"],
            "joints": mesh["joints"],
            "betas": params["betas"],
            "pose": params["pose"],
            "global_orient": params["global_orient"],
            "cam": params["cam"],
        }

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        gt_joints_3d: Optional[torch.Tensor] = None,
        gt_joints_2d: Optional[torch.Tensor] = None,
        gt_betas: Optional[torch.Tensor] = None,
        gt_pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses.

        Args:
            pred: Model predictions.
            gt_joints_3d: Ground-truth 3D joints (B, J, 3).
            gt_joints_2d: Ground-truth 2D joints (B, J, 2).
            gt_betas: Ground-truth shape params.
            gt_pose: Ground-truth pose params.

        Returns:
            Dict of loss components.
        """
        losses = {}

        if gt_joints_3d is not None:
            num_j = min(pred["joints"].shape[1], gt_joints_3d.shape[1])
            losses["joint3d"] = F.l1_loss(pred["joints"][:, :num_j], gt_joints_3d[:, :num_j])

        if gt_joints_2d is not None:
            cam = pred["cam"]
            proj = pred["joints"][:, :gt_joints_2d.shape[1], :2]
            scale = cam[:, 0:1].unsqueeze(1)
            transl_2d = cam[:, 1:3].unsqueeze(1)
            proj_2d = scale * proj + transl_2d
            losses["joint2d"] = F.l1_loss(proj_2d, gt_joints_2d)

        if gt_betas is not None:
            losses["betas"] = F.mse_loss(pred["betas"], gt_betas)

        if gt_pose is not None:
            min_dim = min(pred["pose"].shape[-1], gt_pose.shape[-1])
            losses["pose"] = F.mse_loss(pred["pose"][..., :min_dim], gt_pose[..., :min_dim])

        losses["total"] = sum(losses.values())
        return losses
