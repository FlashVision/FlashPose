# 2D Body Pose Estimation

## Overview

FlashPose supports 2D body pose estimation with 17 COCO keypoints using ViTPose, HRNet, or RTMPose backbones.

## Keypoints

| ID | Name | ID | Name |
|---|---|---|---|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

## Usage

```python
from flashpose import FlashPose, PoseEstimator
from flashpose.cfg import get_config

# Quick inference
estimator = PoseEstimator(model_path="best.pth", task="body_2d")
results = estimator.run("image.jpg", visualize=True)

# Custom model
config = get_config(model_name="ViTPose", task="body_2d", num_keypoints=17)
model = FlashPose(config)
```

## Training

```bash
flashpose train --config configs/flashpose_body_17.yaml
```

## Metrics

- **PCK@0.5**: Percentage of Correct Keypoints (threshold = 0.5 × bbox size)
- **AP**: Average Precision using Object Keypoint Similarity (OKS)
