import queue
import threading
import cv2
import logging
from collections import deque

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from app.core.config import settings

from app.services.yolo_processor import YOLOProcessor
from app.services.gesture_processor import GestureProcessor
from app.services.gesture_dispatcher import GestureDispatcher

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


def _safe_put(q: queue.Queue, item) -> None:
    """Drop the oldest item when the queue is full, then enqueue the new one."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put_nowait(item)


class CameraStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)

        self.yolo_processor = YOLOProcessor()
        self.gesture_processor = GestureProcessor()
        self.latest_detections = []
        self.session_detections: dict = {}
        self.latest_gestures = []
        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.current_fps = 0.0

        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._last_frame: np.ndarray = np.zeros(
            (settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH, 3), dtype=np.uint8
        )
        self._last_detections: list = []
        self._last_gestures: list = []

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            _safe_put(self._frame_queue, frame)

    def _inference_loop(self) -> None:
        self.yolo_processor.warmup()
        self.gesture_processor.warmup()
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.yolo_processor.enabled:
                annotated, detections = self.yolo_processor.process(frame)
                for det in detections:
                    cls = det.get("class", "")
                    if cls and (cls not in self.session_detections or det.get("conf", 0) > self.session_detections[cls].get("conf", 0)):
                        self.session_detections[cls] = det
            else:
                annotated, detections = frame, []
            if self.gesture_processor.enabled:
                annotated, gestures = self.gesture_processor.process(annotated)
            else:
                gestures = []

            _safe_put(self._result_queue, (annotated, detections, gestures))

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        try:
            frame, detections, gestures = self._result_queue.get_nowait()
            self._last_frame = frame
            self._last_detections = detections
            self._last_gestures = gestures
        except queue.Empty:
            frame = self._last_frame
            detections = self._last_detections
            gestures = self._last_gestures

        self.latest_detections = detections
        self.latest_gestures = gestures
        self.current_fps = self.fps_calc.get()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    def stop(self):
        self._stop_event.set()
        self.gesture_processor.stop()
        self.cap.release()
        super().stop()


class Go2CameraStreamTrack(VideoStreamTrack):
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
        self.gesture_processor = GestureProcessor()
        self.session_detections: dict = {}
        self.gesture_dispatcher = GestureDispatcher(
            enabled=True,
            cooldown_seconds=settings.GESTURE_DISPATCH_COOLDOWN,
            global_cooldown_seconds=settings.GESTURE_DISPATCH_GLOBAL_COOLDOWN,
            min_confidence=settings.GESTURE_DISPATCH_MIN_CONFIDENCE,
            min_stable_frames=settings.GESTURE_DISPATCH_MIN_STABLE_FRAMES,
        )
        self.latest_detections = []
        self.latest_gestures = []
        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.current_fps = 0.0

        self._offline_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._connecting_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self._initializing = True
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self._result_queue: queue.Queue = queue.Queue(maxsize=3)
        self._last_frame: np.ndarray = self._connecting_frame.copy() if self.connected else self._offline_frame.copy()
        self._last_detections: list = []
        self._last_gestures: list = []

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.connected or self.client is None:
                self._stop_event.wait(0.05)
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
            except Exception as e:
                logger.error(f"Error getting Go2 image: {e}")

    def _inference_loop(self) -> None:
        self.yolo_processor.warmup()
        self.gesture_processor.warmup()
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.yolo_processor.enabled:
                annotated, detections = self.yolo_processor.process(frame)
                for det in detections:
                    cls = det.get("class", "")
                    if cls and (cls not in self.session_detections or det.get("conf", 0) > self.session_detections[cls].get("conf", 0)):
                        self.session_detections[cls] = det
            else:
                annotated, detections = frame, []
            if self.gesture_processor.enabled:
                annotated, gestures = self.gesture_processor.process(annotated)
            else:
                gestures = []

            self.gesture_dispatcher.process(gestures)

            self._initializing = False
            _safe_put(self._result_queue, (annotated, detections, gestures))

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        try:
            frame, detections, gestures = self._result_queue.get_nowait()
            self._last_frame = frame
            self._last_detections = detections
            self._last_gestures = gestures
        except queue.Empty:
            if self._initializing:
                frame = self._connecting_frame if self.connected else self._offline_frame
            else:
                frame = self._last_frame
            detections = self._last_detections
            gestures = self._last_gestures

        self.latest_detections = detections
        self.latest_gestures = gestures
        self.current_fps = self.fps_calc.get()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    def stop(self):
        self._stop_event.set()
        self.gesture_processor.stop()
        self.client = None  # stop capture loop from calling GetImageSample()
        super().stop()
