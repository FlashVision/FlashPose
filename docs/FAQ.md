# Frequently Asked Questions

## General

**Q: What GPU do I need?**
A: Training ViTPose-Base requires ~8GB VRAM. RTMPose-S can train on 4GB. Inference works on CPU too.

**Q: Can I use FlashPose without a GPU?**
A: Yes. Set `--device cpu`. RTMPose achieves real-time on modern CPUs.

**Q: What input formats are supported?**
A: Images (jpg, png, bmp), videos (mp4, avi, mov), directories, and webcam streams.

## Models

**Q: Which model should I use?**
A: RTMPose-S for real-time applications, ViTPose-B for best accuracy, HRNet-W32 for balanced performance.

**Q: Can I use custom keypoint definitions?**
A: Yes. Set `num_keypoints` in the config and provide matching annotations.

**Q: Does FlashPose support multi-person pose?**
A: FlashPose uses a top-down approach. Pair it with a person detector (e.g., FlashDet) for multi-person.

## Training

**Q: How do I fine-tune on my own data?**
A: Prepare COCO-format annotations, update the config YAML paths, and run `flashpose train --config your_config.yaml`.

**Q: What is LoRA fine-tuning?**
A: Low-Rank Adaptation freezes base weights and adds tiny trainable adapters. Use `--lora` to enable. Reduces trainable params by 95%+.

**Q: How long does training take?**
A: ~24h for ViTPose-B on COCO with 1x A100. RTMPose-S trains in ~8h.

## Deployment

**Q: How do I deploy to production?**
A: Export to ONNX (`flashpose export`), then use ONNX Runtime or TensorRT for inference.

**Q: Can I quantize the model?**
A: Export to ONNX first, then use onnxruntime's quantization tools.
