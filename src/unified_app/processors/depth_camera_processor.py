"""Depth Camera Processor - RealSense Integration with YOLO"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, Any
from src.unified_app.processor import BaseProcessor
from src.unified_app.config import DepthCameraConfig

logger = logging.getLogger(__name__)


class DepthCameraProcessor(BaseProcessor):
    """
    Depth camera processor using Intel RealSense.
    Combines RGB stream with depth data and runs YOLO detection on RGB.
    """

    def __init__(self, config: DepthCameraConfig):
        super().__init__("depth_camera", config)
        self.pipeline = None
        self.yolo_model = None
        self.confidence = 0.5

    async def start(self, camera, input_queue, output_queue, model_pool):
        """Start processor and initialize RealSense"""
        await super().start(camera, input_queue, output_queue, model_pool)

        # Initialize RealSense pipeline
        try:
            import pyrealsense2 as rs

            self.pipeline = rs.pipeline()
            rs_config = rs.config()

            rs_config.enable_stream(
                rs.stream.depth, self.config.depth_width, self.config.depth_height, rs.format.z16, self.config.fps
            )
            rs_config.enable_stream(
                rs.stream.color, self.config.color_width, self.config.color_height, rs.format.bgr8, self.config.fps
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.pipeline.start(rs_config))

            self.logger.info("RealSense pipeline initialized")

        except ImportError:
            self.logger.warning(
                "pyrealsense2 not installed - depth camera disabled. "
                "Install with: pip install pyrealsense2"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize RealSense: {e}")

    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Capture frames from RealSense and fuse depth with RGB.

        Args:
            frame: Ignored (uses RealSense internal pipeline)

        Returns:
            RGB frame with optional depth visualization
        """
        if self.pipeline is None:
            return frame

        try:
            # Get frames from RealSense
            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(
                None, lambda: self.pipeline.wait_for_frames()
            )

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return frame

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Normalize depth for visualization (0-255)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Run YOLO on color frame if available
            if self.yolo_model is not None:
                results = await loop.run_in_executor(
                    None,
                    lambda: self.yolo_model(color_image, conf=self.confidence, verbose=False),
                )
                annotated_color = results[0].plot()
            else:
                annotated_color = color_image

            # Stack RGB and Depth side by side
            output = np.hstack([annotated_color, depth_colormap])

            return output

        except Exception as e:
            self.logger.error(f"Error in depth camera processing: {e}", exc_info=True)
            return frame
