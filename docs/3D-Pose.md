# 3D Pose Estimation

## Overview

FlashPose supports 3D body pose estimation using Human3.6M skeleton (17 joints). Two approaches:

1. **Direct 3D prediction** from images
2. **2D-to-3D lifting** using an MLP network

## Lifting Pipeline

```python
from flashpose.tasks.body_3d import Body3DTask

task = Body3DTask()
lifter = task.build_lifter(hidden_dim=1024, num_layers=4)

# Input: 2D keypoints (B, 17, 2)
# Output: 3D joints (B, 17, 3)
joints_3d = lifter(joints_2d.flatten(1))
```

## Procrustes Alignment

For fair evaluation, predictions are aligned to ground truth:

```python
aligned = task.procrustes_align(predicted_3d, ground_truth_3d)
```

## Metrics

- **MPJPE**: Mean Per Joint Position Error (in mm)
- **PA-MPJPE**: Procrustes-Aligned MPJPE (shape-only error)

## Training

```bash
flashpose train --config configs/flashpose_3d.yaml
```
