"""Base Processor - Abstract base class for all vision processors"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import cv2
import numpy as np
from src.unified_app.model_pool import ModelPool
from src.unified_app.common.cameras import BaseCamera
from src.unified_app.common.fps import FPSCalculator, draw_fps_on_frame

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for vision processors.
    All processors (YOLO, Hand Detection, Depth Camera, etc.) inherit from this.
    """

    def __init__(self, name: str, config: Union[Dict[str, Any], Any]):
        """
        Initialize processor.

        Args:
            name: Processor identifier (e.g., "yolo", "hand_detection")
            config: Configuration dictionary for this processor
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"Processor.{name}")

        # State management
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.frame_skip_counter = 0

        # Camera
        self.camera: Optional[BaseCamera] = None

        # Frame queues for streaming
        # Input queue: frames from camera (shared with other processors)
        # Output queue: processed frames for streaming
        self.input_queue: Optional[asyncio.Queue] = None
        self.output_queue: Optional[asyncio.Queue] = None

        # Model pool
        self.model_pool: Optional[ModelPool] = None

        # FPS Calculator
        self.fps_calc = FPSCalculator(buffer_len=5)

        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0

        self.logger.info(f"Processor initialized with config: {config}")

    async def start(
        self,
        camera: BaseCamera,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        model_pool: ModelPool,
    ):
        """
        Start the processor.

        Args:
            camera: Camera instance to use
            input_queue: Queue to read input frames from
            output_queue: Queue to write processed frames to
            model_pool: Shared model pool for models
        """
        self.camera = camera
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_pool = model_pool
        self.running = True

        # Create and start the processing task
        self.task = asyncio.create_task(self._run())
        self.logger.info(f"Processor started")

    async def stop(self):
        """Stop the processor gracefully"""
        self.running = False

        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"Processor stopped")

    async def _run(self):
        """
        Main processing loop - calls process_frame in a loop.
        """
        try:
            while self.running:
                try:
                    # Get frame from input queue (with timeout to allow cancellation)
                    try:
                        frame = await asyncio.wait_for(
                            self.input_queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        # Timeout is normal, just continue waiting
                        continue

                    # Skip frames if configured
                    if self._should_skip_frame():
                        self.dropped_frames += 1
                        continue

                    # Process the frame
                    processed_frame = await self.process_frame(frame)

                    # Put processed frame in output queue (non-blocking, drop if full)
                    if not self.output_queue.full():
                        await self.output_queue.put(processed_frame)
                    else:
                        self.dropped_frames += 1

                    self.frame_count += 1

                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}", exc_info=True)
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self.logger.info(f"Processing loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in processor: {e}", exc_info=True)
        finally:
            self.running = False
            self.logger.info(
                f"Processor stopped. Processed {self.frame_count} frames, "
                f"dropped {self.dropped_frames}"
            )

    def _should_skip_frame(self) -> bool:
        """Check if current frame should be skipped"""
        skip_frames = self.config.get("skip_frames", 0)
        if skip_frames <= 0:
            return False

        self.frame_skip_counter += 1
        if self.frame_skip_counter >= skip_frames + 1:
            self.frame_skip_counter = 0
            return False
        return True

    @abstractmethod
    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame and return the processed frame.
        Subclasses must implement this.

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            Processed frame as numpy array (BGR format)
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary with processor stats
        """
        return {
            "name": self.name,
            "running": self.running,
            "frame_count": self.frame_count,
            "dropped_frames": self.dropped_frames,
            "fps": self.fps_calc.get(),
        }
