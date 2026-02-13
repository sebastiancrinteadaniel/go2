"""YOLO Object Detection Processor"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, Any
from src.unified_app.processor import BaseProcessor
from src.unified_app.config import YoloConfig

logger = logging.getLogger(__name__)


class YoloProcessor(BaseProcessor):
    """
    YOLO object detection processor.
    Runs YOLOv8 inference on frames and overlays detections.
    """

    def __init__(self, config: YoloConfig):
        super().__init__("yolo", config)
        self.model = None
        self.device = config.device
        self.confidence = config.confidence
        self.model_path = config.model_path

    async def start(self, camera, input_queue, output_queue, model_pool):
        """Start processor and load model"""
        await super().start(camera, input_queue, output_queue, model_pool)

        # Load YOLO model
        if self.model_path:
            self.model = await self.model_pool.get_yolo(self.model_path, self.device)
            self.logger.info(f"YOLO model loaded from {self.model_path}")
        else:
            self.logger.warning("No YOLO model path configured")

    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect objects in frame using YOLO and draw bounding boxes.

        Args:
            frame: Input frame in BGR format

        Returns:
            Frame with detection boxes overlaid
        """
        if self.model is None:
            return frame

        try:
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.model(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                ),
            )

            # Get the result
            result = results[0]

            # Draw bounding boxes on frame
            annotated_frame = result.plot()

            return annotated_frame

        except Exception as e:
            self.logger.error(f"Error in YOLO inference: {e}", exc_info=True)
            return frame
