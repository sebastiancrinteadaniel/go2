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
from app.services.industrial_processor import IndustrialProcessor
from app.services.gesture_processor import GestureProcessor
from app.services.gesture_dispatcher import GestureDispatcher
from app.services.industrial_depth_processor import IndustrialDepthProcessor

logger = logging.getLogger(__name__)


def _overlay_depth_pip(frame: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
    """Overlay a colourised depth minimap in the bottom-right corner of *frame*.

    Depth values are clipped to 0–5 000 mm and mapped with COLORMAP_TURBO
    (blue ≈ close, red ≈ far).
    """
    h, w = frame.shape[:2]
    pip_w, pip_h = w // 4, h // 4

    clipped = np.clip(depth_raw, 0, 5000).astype(np.float32)
    normalized = (clipped / 5000 * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_resized = cv2.resize(depth_color, (pip_w, pip_h))

    # Thin border + label
    cv2.rectangle(depth_resized, (0, 0), (pip_w - 1, pip_h - 1), (180, 180, 180), 1)
    cv2.putText(
        depth_resized, "DEPTH (0-5m)", (4, pip_h - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
    )

    x_off, y_off = w - pip_w - 4, h - pip_h - 4
    frame[y_off : y_off + pip_h, x_off : x_off + pip_w] = depth_resized
    return frame


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


class CameraSource:
    """Webcam capture + inference pipeline. Shared across all viewer connections."""

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)

        self.yolo_processor = YOLOProcessor()
        self.industrial_processor = IndustrialProcessor()
        self.gesture_processor = GestureProcessor()
        self.latest_detections = []
        self.latest_industrial_detections = []
        self.session_detections: dict = {}
        self.latest_gestures = []
        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.current_fps = 0.0

        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._last_frame: np.ndarray = np.zeros(
            (settings.CAMERA_HEIGHT, settings.CAMERA_WIDTH, 3), dtype=np.uint8
        )

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            _safe_put(self._frame_queue, frame)

    def _inference_loop(self) -> None:
        self.yolo_processor.warmup()
        self.industrial_processor.warmup()
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
            if self.industrial_processor.enabled:
                annotated, industrial_detections = self.industrial_processor.process(annotated)
            else:
                industrial_detections = []
            if self.gesture_processor.enabled:
                annotated, gestures = self.gesture_processor.process(annotated)
            else:
                gestures = []

            self._last_frame = annotated
            self.latest_detections = detections
            self.latest_industrial_detections = industrial_detections
            self.latest_gestures = gestures

    def get_latest_frame(self) -> np.ndarray:
        return self._last_frame

    def stop(self):
        self._stop_event.set()
        self.gesture_processor.stop()
        self.cap.release()


class Go2CameraSource:
    """Go2 SDK camera + inference pipeline. Shared across all viewer connections."""

    def __init__(self):
        try:
            from unitree_sdk2py.go2.video.video_client import VideoClient

            self.client = VideoClient()
            self.client.SetTimeout(3.0)
            self.client.Init()
            self.connected = True
            logger.info("Unitree SDK VideoClient initialized successfully.")
        except ImportError:
            self.client = None
            self.connected = False
            logger.warning(
                "'unitree_sdk2py' not found. Ensure it is installed for the Go2 camera stream to work."
            )
        except Exception as e:
            self.client = None
            self.connected = False
            logger.error(f"Error initializing Go2 VideoClient: {e}")

        self.yolo_processor = YOLOProcessor()
        self.industrial_processor = IndustrialProcessor()
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
        self.latest_industrial_detections = []
        self.latest_gestures = []
        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.current_fps = 0.0

        self._offline_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._connecting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._initializing = True
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self._last_frame: np.ndarray = (
            self._connecting_frame.copy() if self.connected else self._offline_frame.copy()
        )

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
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
        self.industrial_processor.warmup()
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
            if self.industrial_processor.enabled:
                annotated, industrial_detections = self.industrial_processor.process(annotated)
            else:
                industrial_detections = []
            if self.gesture_processor.enabled:
                annotated, gestures = self.gesture_processor.process(annotated)
            else:
                gestures = []

            self.gesture_dispatcher.process(gestures)

            self._last_frame = annotated
            self.latest_detections = detections
            self.latest_industrial_detections = industrial_detections
            self.latest_gestures = gestures
            self._initializing = False

    def get_latest_frame(self) -> np.ndarray:
        if self._initializing:
            return self._connecting_frame if self.connected else self._offline_frame
        return self._last_frame

    def stop(self):
        self._stop_event.set()
        self.gesture_processor.stop()
        self.client = None


class DepthCameraSource:
    """OAK-D S2 RGB + depth camera with sensor fusion composite output.

    Streams the full RGB feed (with AI inference) and composites a colourised
    depth minimap (picture-in-picture) in the bottom-right corner.
    """

    def __init__(self):
        self.camera_error: str = ""
        try:
            import depthai as dai

            self._dai = dai
            self._pipeline = self._build_pipeline(dai)
            self.connected = True
            logger.info("DepthAI pipeline ready.")
        except ImportError:
            self._dai = None
            self._pipeline = None
            self.connected = False
            self.camera_error = "depthai not installed"
            logger.warning("'depthai' not installed. Sensor fusion mode unavailable.")
        except Exception as e:
            self._dai = None
            self._pipeline = None
            self.connected = False
            self.camera_error = str(e)
            logger.error(f"Error building DepthAI pipeline: {e}")

        self.yolo_processor = YOLOProcessor()
        self.industrial_processor = IndustrialDepthProcessor()
        self.gesture_processor = GestureProcessor()
        self.session_detections: dict = {}
        # Gesture dispatch works whenever the Unitree SDK is importable —
        # ChannelFactoryInitialize is already called at app startup in main.py.
        self.gesture_dispatcher = GestureDispatcher(
            enabled=True,
            cooldown_seconds=settings.GESTURE_DISPATCH_COOLDOWN,
            global_cooldown_seconds=settings.GESTURE_DISPATCH_GLOBAL_COOLDOWN,
            min_confidence=settings.GESTURE_DISPATCH_MIN_CONFIDENCE,
            min_stable_frames=settings.GESTURE_DISPATCH_MIN_STABLE_FRAMES,
        )
        self.latest_detections = []
        self.latest_industrial_detections = []
        self.latest_gestures = []
        self.fps_calc = CvFpsCalc(buffer_len=10)
        self.current_fps = 0.0

        self._offline_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._connecting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._initializing = True
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._last_frame: np.ndarray = (
            self._connecting_frame.copy() if self.connected else self._offline_frame.copy()
        )

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    @staticmethod
    def _build_pipeline(dai):
        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)
        cam_rgb.initialControl.setManualFocus(130)

        cam_left = pipeline.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        cam_right = pipeline.create(dai.node.MonoCamera)
        cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # warp depth to RGB perspective
        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        cam_rgb.video.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)

        return pipeline

    def _capture_loop(self) -> None:
        if not self.connected or self._pipeline is None:
            return

        # USB cleanup from a previous session can lag a few seconds; retry the
        # device open instead of failing hard on the first "device busy" error.
        max_attempts = 5
        last_error = None  # type: Exception | None  (kept as comment for Python 3.8 compatibility)
        for attempt in range(1, max_attempts + 1):
            if self._stop_event.is_set():
                return
            try:
                with self._dai.Device(self._pipeline) as device:
                    logger.info(
                        f"OAK-D connected: {device.getDeviceName()}  USB: {device.getUsbSpeed().name}"
                    )
                    rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
                    depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)
                    latest_depth = None
                    while not self._stop_event.is_set():
                        rgb_msg = rgb_queue.tryGet()
                        depth_msg = depth_queue.tryGet()
                        if depth_msg is not None:
                            latest_depth = depth_msg.getFrame()
                        if rgb_msg is not None and latest_depth is not None:
                            # Camera is physically mounted upside down — rotate 180°
                            rgb_frame = cv2.flip(rgb_msg.getCvFrame(), -1)
                            depth_frame = cv2.flip(latest_depth.copy(), -1)
                            _safe_put(self._frame_queue, (rgb_frame, depth_frame))
                        else:
                            self._stop_event.wait(0.002)
                return  # clean exit via stop_event
            except Exception as e:
                last_error = e
                logger.warning(
                    f"OAK-D open attempt {attempt}/{max_attempts} failed: {e}"
                )
                if self._stop_event.wait(1.5):
                    return

        logger.error(f"OAK-D capture error after {max_attempts} attempts: {last_error}")
        self.connected = False
        self.camera_error = str(last_error) if last_error else "unknown"

    def _inference_loop(self) -> None:
        self.yolo_processor.warmup()
        self.industrial_processor.warmup()
        self.gesture_processor.warmup()
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            frame, depth_raw = item

            if self.yolo_processor.enabled:
                annotated, detections = self.yolo_processor.process(frame)
                for det in detections:
                    cls = det.get("class", "")
                    if cls and (
                        cls not in self.session_detections
                        or det.get("conf", 0) > self.session_detections[cls].get("conf", 0)
                    ):
                        self.session_detections[cls] = det
            else:
                annotated, detections = frame, []

            if self.industrial_processor.enabled:
                annotated, industrial_detections = self.industrial_processor.process(annotated, depth_raw)
            else:
                industrial_detections = []

            if self.gesture_processor.enabled:
                annotated, gestures = self.gesture_processor.process(annotated)
            else:
                gestures = []

            self.gesture_dispatcher.process(gestures)

            # Composite depth PiP onto the annotated RGB frame
            annotated = _overlay_depth_pip(annotated, depth_raw)

            self._last_frame = annotated
            self.latest_detections = detections
            self.latest_industrial_detections = industrial_detections
            self.latest_gestures = gestures
            self._initializing = False

    def get_latest_frame(self) -> np.ndarray:
        if self._initializing:
            return self._connecting_frame if self.connected else self._offline_frame
        return self._last_frame

    def stop(self):
        self._stop_event.set()
        self.gesture_processor.stop()
        # Wait for the capture thread to exit its `with dai.Device(...)` block
        # so the OAK-D USB handle is fully released before a new source opens.
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        if self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.0)


class ViewerTrack(VideoStreamTrack):
    """One WebRTC video track per viewer. Reads frames from a shared camera source."""

    def __init__(self, source: "CameraSource | Go2CameraSource | DepthCameraSource"):
        super().__init__()
        self._source = source

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self._source.get_latest_frame()
        self._source.current_fps = self._source.fps_calc.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vf = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        vf.pts = pts
        vf.time_base = time_base
        return vf
