import asyncio
import cv2
import logging
import queue
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


def _safe_put(q: queue.Queue, item):
    """Non-blocking put — drops the oldest item if full so queue always holds the newest."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class CameraStreamTrack(VideoStreamTrack):
    """
    Exact port of example/go2/src/yolo/run.py multithreaded pipeline for WebRTC:
      _capture_loop  (Thread) -> _frame_queue
      _inference_worker (Thread) -> _result_queue
      recv()  (aiortc, mirrors display loop) -> polls _result_queue, returns last_vis
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

        # Two-queue pipeline (mirrors example capture_loop -> inference_worker)
        self._frame_queue = queue.Queue(maxsize=2)
        self._result_queue = queue.Queue(maxsize=2)
        self._last_vis = None

        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True, name="webcam-capture")
        self._inf_thread = threading.Thread(target=self._inference_worker, daemon=True, name="webcam-inference")
        self._cap_thread.start()
        self._inf_thread.start()

    def _capture_loop(self):
        """Mirrors capture_loop from example: reads camera, drops stale frames."""
        while True:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                _safe_put(self._frame_queue, frame)
            else:
                time.sleep(0.005)

    def _inference_worker(self):
        """Mirrors inference_worker from example: reads frame_queue, runs YOLO (or passthrough), puts to result_queue."""
        while True:
            try:
                img = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            vis, detections = self.yolo_processor.process(img)
            _safe_put(self._result_queue, (vis, detections))

    async def recv(self):
        """Mirrors display loop from example: polls result_queue, returns last_vis."""
        pts, time_base = await self.next_timestamp()

        # Non-blocking poll — update last_vis when a new result is ready
        try:
            vis, detections = self._result_queue.get_nowait()
            self._last_vis = vis
            self.latest_detections = detections
        except queue.Empty:
            pass

        if self._last_vis is None:
            frame = np.zeros((settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH, 3), dtype=np.uint8)
        else:
            frame = self._last_vis

        self.current_fps = self.fps_calc.get()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame


class Go2CameraStreamTrack(VideoStreamTrack):
    """
    Same two-queue pipeline for the Go2 robot camera.
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

        # Offline placeholder shown before robot connects
        self._offline_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            self._offline_frame, "GO2 CAMERA UNAVAILABLE",
            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA,
        )

        self._frame_queue = queue.Queue(maxsize=2)
        self._result_queue = queue.Queue(maxsize=2)
        self._last_vis = self._offline_frame.copy()

        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True, name="go2-capture")
        self._inf_thread = threading.Thread(target=self._inference_worker, daemon=True, name="go2-inference")
        self._cap_thread.start()
        self._inf_thread.start()

    def _capture_loop(self):
        """Mirrors capture_loop: reads Go2 camera, drops stale frames."""
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
                        _safe_put(self._frame_queue, frame)
                else:
                    logger.warning(f"Get image sample error. code: {code}")
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"Error getting Go2 image: {e}")
                time.sleep(0.05)

    def _inference_worker(self):
        """Mirrors inference_worker: reads frame_queue, runs YOLO (or passthrough), puts to result_queue."""
        while True:
            try:
                img = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            vis, detections = self.yolo_processor.process(img)
            _safe_put(self._result_queue, (vis, detections))

    async def recv(self):
        """Mirrors display loop: polls result_queue, returns last_vis."""
        pts, time_base = await self.next_timestamp()

        try:
            vis, detections = self._result_queue.get_nowait()
            self._last_vis = vis
            self.latest_detections = detections
        except queue.Empty:
            pass

        frame = self._last_vis
        self.current_fps = self.fps_calc.get()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame
