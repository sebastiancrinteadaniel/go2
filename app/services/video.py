import asyncio
import cv2
import logging
import threading
import time
from collections import deque

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from app.core.config import settings

from app.services.yolo_processor import YOLOProcessor

logger = logging.getLogger(__name__)


class CvFpsCalc:
    def __init__(self, buffer_len: int = 10):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self) -> float:
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick
        self._difftimes.append(different_time)
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        return round(fps, 2)


class CameraStreamTrack(VideoStreamTrack):
    """
    A video stream track that reads frames from a local web camera.
    Camera capture runs in a background thread (mirrors the example capture_loop
    pattern) so recv() never blocks on I/O and can run at the WebRTC target FPS.
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

        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.current_fps = 0.0

        # Single-slot frame buffer — always holds the freshest camera frame
        self._frame_lock = threading.Lock()
        self._latest_frame = np.zeros(
            (settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH, 3), dtype=np.uint8
        )
        self._reader_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._reader_thread.start()

    def _capture_loop(self):
        """Continuously reads camera frames and caches the latest one."""
        while True:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.005)

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        with self._frame_lock:
            frame = self._latest_frame.copy()

        self.current_fps = self.fps_calc.get()

        self.yolo_processor.submit(frame)
        frame, self.latest_detections = self.yolo_processor.get_latest(frame)

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

        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.current_fps = 0.0

        # Single-slot frame buffer — always holds the freshest robot camera frame
        self._frame_lock = threading.Lock()
        self._offline_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            self._offline_frame,
            "GO2 CAMERA UNAVAILABLE",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        self._latest_frame = self._offline_frame.copy()
        self._reader_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._reader_thread.start()

    def _capture_loop(self):
        """Continuously reads Go2 camera frames and caches the latest one."""
        while True:
            if not self.connected:
                time.sleep(0.05)
                continue
            try:
                code, data = self.client.GetImageSample()
                if code == 0:
                    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._frame_lock:
                            self._latest_frame = frame
                else:
                    logger.warning(f"Get image sample error. code: {code}")
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"Error getting Go2 image: {e}")
                time.sleep(0.05)

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        with self._frame_lock:
            frame = self._latest_frame.copy()

        self.current_fps = self.fps_calc.get()

        self.yolo_processor.submit(frame)
        frame, self.latest_detections = self.yolo_processor.get_latest(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame
