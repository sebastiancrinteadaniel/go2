"""Video Streaming - MJPEG frame encoding and HTTP streaming"""

import asyncio
import logging
import cv2
import numpy as np
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


class MJPEGStreamer:
    """
    MJPEG (Motion JPEG) streamer for HTTP video streaming.
    Encodes frames to JPEG and sends as multipart/x-mixed-replace stream.
    """

    def __init__(self, quality: int = 80):
        """
        Initialize MJPEG streamer.

        Args:
            quality: JPEG compression quality (0-100)
        """
        self.quality = quality
        self.logger = logging.getLogger("MJPEGStreamer")

    async def encode_frame(self, frame: np.ndarray) -> bytes:
        """
        Encode frame to JPEG bytes asynchronously.

        Args:
            frame: Frame in BGR format

        Returns:
            JPEG-encoded bytes
        """
        try:
            loop = asyncio.get_event_loop()

            # Encode in thread pool to avoid blocking
            def encode():
                ret, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality],
                )
                if ret:
                    return jpeg.tobytes()
                return None

            jpeg_bytes = await loop.run_in_executor(None, encode)
            return jpeg_bytes

        except Exception as e:
            self.logger.error(f"Error encoding frame: {e}")
            return None

    async def stream_generator(
        self, frame_queue: asyncio.Queue, timeout: float = 5.0
    ) -> AsyncGenerator[bytes, None]:
        """
        Async generator for MJPEG stream.
        Yields multipart frames for HTTP streaming.

        Args:
            frame_queue: asyncio.Queue containing frames
            timeout: Timeout for getting frames

        Yields:
            Multipart JPEG data for HTTP streaming
        """
        try:
            while True:
                try:
                    # Get frame from queue with timeout
                    frame = await asyncio.wait_for(
                        frame_queue.get(), timeout=timeout
                    )

                    # Encode frame
                    jpeg_data = await self.encode_frame(frame)

                    if jpeg_data is None:
                        continue

                    # Yield multipart JPEG frame
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_data)).encode() + b"\r\n\r\n"
                        + jpeg_data
                        + b"\r\n"
                    )

                except asyncio.TimeoutError:
                    # Timeout while waiting for frame - yield empty marker
                    self.logger.debug("Frame queue timeout")
                    continue

        except asyncio.CancelledError:
            self.logger.info("Stream generator cancelled")
        except Exception as e:
            self.logger.error(f"Error in stream generator: {e}", exc_info=True)


async def mjpeg_handler(
    frame_queue: asyncio.Queue, quality: int = 80
) -> AsyncGenerator[bytes, None]:
    """
    FastAPI-compatible MJPEG stream handler.

    Args:
        frame_queue: Queue containing frames
        quality: JPEG quality

    Yields:
        MJPEG frame data
    """
    streamer = MJPEGStreamer(quality=quality)

    # Send stream start boundary
    yield b"--frame\r\n"

    # Stream frames
    async for frame_data in streamer.stream_generator(frame_queue):
        yield frame_data
