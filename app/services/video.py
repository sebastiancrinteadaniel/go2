import asyncio
import cv2
import logging
import time

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from app.core.config import settings

from app.services.yolo_processor import YOLOProcessor

logger = logging.getLogger(__name__)


class CameraStreamTrack(VideoStreamTrack):
    """
    A video stream track that reads frames from a local web camera.
    """

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)

        self.yolo_processor = YOLOProcessor()
        self.latest_detections = []

        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)

        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        frame, self.latest_detections = await self.yolo_processor.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame


class Go2CameraStreamTrack(VideoStreamTrack):
    """
    A video stream track that reads frames from the Go2 robot's camera via the Unitree SDK.
    """

    def __init__(self):
        super().__init__()
        try:
            from unitree_sdk2py.go2.video.video_client import VideoClient

            self.client = VideoClient()
            self.client.SetTimeout(3.0)
            self.client.Init()
            self.connected = True
            logger.info("Unitree SDK VideoClient initialized successfully.")
        except ImportError:
            self.connected = False
            logger.warning(
                "'unitree_sdk2py' not found. Ensure it is installed for the Go2 camera stream to work."
            )
        except Exception as e:
            self.connected = False
            logger.error(f"Error initializing Go2 VideoClient: {e}")

        self.yolo_processor = YOLOProcessor()
        self.latest_detections = []

        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = None
        if self.connected:
            loop = asyncio.get_event_loop()
            try:
                code, data = await loop.run_in_executor(None, self.client.GetImageSample)
                if code == 0:
                    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                else:
                    logger.warning(f"Get image sample error. code: {code}")
            except Exception as e:
                logger.error(f"Error getting Go2 image: {e}")

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "GO2 CAMERA UNAVAILABLE",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        frame, self.latest_detections = await self.yolo_processor.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame
