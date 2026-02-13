"""Frame Router - Orchestrates frame capture and distribution to processors"""

import asyncio
import logging
from typing import Dict, List, Optional
import numpy as np
from src.unified_app.common.cameras import create_camera, BaseCamera
from src.unified_app.config import CameraConfig

logger = logging.getLogger(__name__)


class FrameRouter:
    """
    Central orchestrator for frame capture and distribution.
    - Manages a single camera instance
    - Distributes frames to all active processors via async queues
    - Handles backpressure when queues fill up
    """

    def __init__(self, camera_config: CameraConfig):
        self.camera_config = camera_config
        self.camera: Optional[BaseCamera] = None
        self.running = False
        self.task: Optional[asyncio.Task] = None

        # Queue for each processor: {processor_name: asyncio.Queue}
        self.processor_queues: Dict[str, asyncio.Queue] = {}
        self.queue_size = 3  # Bounded queue size to prevent memory buildup

        # Statistics
        self.frame_count = 0
        self.dropped_frames_total = 0
        self.logger = logging.getLogger("FrameRouter")

    async def initialize(self):
        """Initialize the camera"""
        try:
            self.camera = await asyncio.get_event_loop().run_in_executor(
                None, lambda: create_camera(self.camera_config.source, self.camera_config.interface)
            )
            self.logger.info(f"Camera initialized: {self.camera_config.source} on {self.camera_config.interface}")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise

    async def register_processor(self, processor_name: str) -> asyncio.Queue:
        """
        Register a processor and get its input queue.

        Args:
            processor_name: Unique identifier for the processor

        Returns:
            asyncio.Queue to read frames from
        """
        if processor_name not in self.processor_queues:
            queue = asyncio.Queue(maxsize=self.queue_size)
            self.processor_queues[processor_name] = queue
            self.logger.info(
                f"Processor registered: {processor_name}. "
                f"Total processors: {len(self.processor_queues)}"
            )
            return queue
        return self.processor_queues[processor_name]

    async def unregister_processor(self, processor_name: str):
        """Unregister a processor"""
        if processor_name in self.processor_queues:
            del self.processor_queues[processor_name]
            self.logger.info(
                f"Processor unregistered: {processor_name}. "
                f"Total processors: {len(self.processor_queues)}"
            )

    async def start(self):
        """Start the frame capture loop"""
        if self.running:
            self.logger.warning("Frame router already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._capture_loop())
        self.logger.info("Frame router started")

    async def stop(self):
        """Stop the frame capture loop"""
        self.running = False

        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        self.logger.info(
            f"Frame router stopped. "
            f"Captured {self.frame_count} frames, "
            f"dropped {self.dropped_frames_total} total"
        )

    async def _capture_loop(self):
        """
        Main capture loop - continuously reads frames and distributes to processors.
        """
        if self.camera is None:
            await self.initialize()

        try:
            while self.running:
                try:
                    # Capture frame from camera in thread pool
                    loop = asyncio.get_event_loop()
                    ret, frame = await loop.run_in_executor(
                        None, self.camera.read
                    )

                    if not ret or frame is None:
                        self.logger.warning("Failed to read frame from camera")
                        await asyncio.sleep(0.1)
                        continue

                    # Distribute frame to all registered processors
                    await self._distribute_frame(frame)

                    self.frame_count += 1

                except Exception as e:
                    self.logger.error(f"Error in capture loop: {e}", exc_info=True)
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self.logger.info("Capture loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in frame router: {e}", exc_info=True)
        finally:
            self.running = False

    async def _distribute_frame(self, frame: np.ndarray):
        """
        Distribute frame to all registered processors.
        Uses non-blocking put - drops frame if queue is full.

        Args:
            frame: Frame to distribute
        """
        dropped = 0

        for processor_name, queue in self.processor_queues.items():
            try:
                # Non-blocking put - frame is dropped if queue is full
                if not queue.full():
                    queue.put_nowait(frame)
                else:
                    dropped += 1
            except Exception as e:
                self.logger.error(
                    f"Error distributing frame to {processor_name}: {e}"
                )

        if dropped > 0:
            self.dropped_frames_total += dropped

    def get_stats(self) -> Dict[str, any]:
        """Get frame router statistics"""
        return {
            "running": self.running,
            "frame_count": self.frame_count,
            "dropped_frames": self.dropped_frames_total,
            "processors": len(self.processor_queues),
            "processor_names": list(self.processor_queues.keys()),
        }
