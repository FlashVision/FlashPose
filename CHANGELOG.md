# Changelog

All notable changes to FlashPose will be documented in this file.

## [1.0.0] - 2024-12-01

### Added
- Initial release of FlashPose
- ViTPose backbone (small, base, large variants)
- HRNet backbone (W32, W48 variants)
- RTMPose backbone (tiny, small, medium, large variants)
- Heatmap, Regression, and SimCC prediction heads
- 2D body pose estimation (COCO 17 keypoints)
- 3D body pose estimation with 2D-to-3D lifting
- Hand pose estimation (21 keypoints)
- Face landmark detection (68 keypoints)
- Whole-body pose estimation (133 keypoints)
- Skeleton-based action recognition (ST-GCN)
- Gesture recognition from hand keypoints
- LoRA fine-tuning support
- Mixed-precision training (AMP)
- ONNX export with graph simplification
- PCK, AP, MPJPE, PA-MPJPE evaluation metrics
- Skeleton visualization utilities
- CLI with train, predict, estimate, export, benchmark commands
- COCO, MPII, Human3.6M dataset support
- Docker support with GPU acceleration
- CI/CD pipeline with GitHub Actions
