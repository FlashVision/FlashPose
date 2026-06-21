# Hand Pose Estimation

## Overview

FlashPose detects 21 hand keypoints: wrist + 4 joints per finger (MCP, PIP, DIP, TIP).

## Keypoints

- 0: wrist
- 1-4: thumb (CMC, MCP, IP, TIP)
- 5-8: index finger (MCP, PIP, DIP, TIP)
- 9-12: middle finger
- 13-16: ring finger
- 17-20: pinky finger

## Usage

```python
from flashpose import FlashPose, PoseEstimator
from flashpose.cfg import get_config
from flashpose.solutions import GestureRecognizer

# Hand pose estimation
estimator = PoseEstimator(model_path="hand_model.pth", task="hand")
results = estimator.run("hand_image.jpg")

# Gesture recognition
recognizer = GestureRecognizer()
gesture = recognizer.recognize(results[0]['keypoints'])
print(f"Gesture: {gesture['gesture']}")
```

## Supported Gestures

- open_palm, fist, peace, thumbs_up, pointing, ok, rock

## Training

```bash
flashpose train --config configs/flashpose_hand.yaml
```
