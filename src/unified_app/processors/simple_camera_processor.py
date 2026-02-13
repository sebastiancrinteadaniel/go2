"""Simple Camera Processor - Minimal overhead, raw frame pass-through"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any
from src.unified_app.processor import BaseProcessor
from src.unified_app.config import SimpleCameraConfig
from src.unified_app.common.fps import draw_fps_on_frame

logger = logging.getLogger(__name__)


class SimpleCameraProcessor(BaseProcessor):
    """
    Simple camera processor for low-latency raw frame streaming.
    Minimal processing - just passes frames through with minimal modifications.
    """

    def __init__(self, config: SimpleCameraConfig):
        super().__init__("simple_camera", config)

    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Minimal processing - just return the frame as-is with FPS overlay.

        Args:
            frame: Input frame in BGR format

        Returns:
            Same frame with FPS overlay
        """
        try:
            # Get current FPS
            fps = self.fps_calc.get()

            # Draw FPS on frame
            frame = draw_fps_on_frame(
                frame,
                fps,
                text_color=(0, 255, 0),  # Green
                bg_color=(0, 0, 0),  # Black
            )

            return frame
        except Exception as e:
            self.logger.error(f"Error in simple camera processing: {e}")
            return frame
