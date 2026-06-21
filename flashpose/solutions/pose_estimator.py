"""High-level pose estimation solution with end-to-end pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from flashpose.engine.predictor import Predictor
from flashpose.utils.visualize import draw_skeleton


class PoseEstimator:
    """End-to-end pose estimation solution.

    Provides a high-level API for detecting and visualizing human poses
    from images, videos, or webcam streams with minimal code.

    Example:
        estimator = PoseEstimator(model_path="best.pth")
        results = estimator.run("photo.jpg", visualize=True)
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        task: str = "body_2d",
        conf_threshold: float = 0.3,
        draw_bbox: bool = True,
        draw_scores: bool = False,
    ):
        self.predictor = Predictor(
            model_path=model_path,
            device=device,
            task=task,
            conf_threshold=conf_threshold,
        )
        self.task = task
        self.conf_threshold = conf_threshold
        self.draw_bbox = draw_bbox
        self.draw_scores = draw_scores

    def run(
        self,
        source: str,
        output_dir: Optional[str] = None,
        visualize: bool = True,
        return_images: bool = False,
    ) -> List[Dict]:
        """Run pose estimation pipeline.

        Args:
            source: Path to image, video, or directory.
            output_dir: Output directory for visualized results.
            visualize: Whether to generate visualizations.
            return_images: Whether to include annotated images in results.

        Returns:
            List of result dicts with keypoints, scores, and optionally images.
        """
        results = self.predictor.predict(source, output_dir=output_dir, visualize=visualize)

        if return_images:
            for result in results:
                if "image_path" in result:
                    img = cv2.imread(result["image_path"])
                    if img is not None and "keypoints" in result:
                        vis = draw_skeleton(
                            img, result["keypoints"], result["scores"],
                            threshold=self.conf_threshold, task=self.task,
                        )
                        result["annotated_image"] = vis

        return results

    def run_webcam(
        self,
        camera_id: int = 0,
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Run real-time pose estimation from webcam.

        Args:
            camera_id: Camera device ID.
            output_path: Path to save output video.
            show: Whether to display the window.
        """
        cap = cv2.VideoCapture(camera_id)
        writer = None

        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print("Press 'q' to quit webcam...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = self.predictor._infer(frame)
            vis_frame = draw_skeleton(
                frame, keypoints, scores,
                threshold=self.conf_threshold, task=self.task,
            )

            if writer:
                writer.write(vis_frame)

            if show:
                cv2.imshow("FlashPose", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    def estimate_single(self, image: np.ndarray) -> Dict:
        """Estimate pose for a single BGR image (numpy array).

        Args:
            image: BGR image as numpy array.

        Returns:
            Dict with 'keypoints' (K, 2) and 'scores' (K,).
        """
        keypoints, scores = self.predictor._infer(image)
        return {"keypoints": keypoints, "scores": scores}
