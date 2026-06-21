# Action Recognition

## Overview

FlashPose includes skeleton-based action recognition using a Spatial-Temporal Graph Convolutional Network (ST-GCN). It classifies human actions from sequences of body keypoints.

## Pipeline

1. Detect 2D pose per frame
2. Collect keypoint sequence (T frames)
3. Normalize and preprocess
4. Classify action with ST-GCN

## Usage

```python
from flashpose.solutions import ActionClassifier

classifier = ActionClassifier(model_path="action.pth", sequence_length=64)

# Batch classification
result = classifier.classify(skeleton_sequence)  # (T, 17, 2)
print(f"Action: {result['action']} (conf: {result['confidence']:.2f})")

# Real-time streaming
for frame_keypoints in stream:
    result = classifier.update(frame_keypoints)
    if result:
        print(f"Detected: {result['action']}")
```

## Supported Actions

60 action classes from NTU RGB+D: drink_water, eat_meal, clapping, hand_waving, kicking, etc.

## Model

- **ST-GCN**: Spatial graph convolution over body joints + temporal 1D convolution
- Input: (B, T, J, C) — batch, time, joints, coordinates
- Output: (B, num_classes) — action logits
