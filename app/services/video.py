import cv2
import logging
import time
import threading

from aiortc import VideoStreamTrack
from av import VideoFrame
from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO

    model = YOLO("app/models/yolov8n.pt")
except ImportError:
    model = None
    logger.warning(
        "'ultralytics' package not found. Object detection disabled. Please install it later to enable YOLO."
    )


class YOLOProcessor:
    """
    Handles YOLO object detection in a background thread so it doesn't block the video stream.
    """

    def __init__(self):
        self.latest_detections = []
        self.frame_to_process = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        if model is not None:
            self.thread.start()

    def update_frame(self, frame):
        if frame is None:
            return

        if self.lock.acquire(blocking=False):
            self.frame_to_process = frame.copy()
            self.lock.release()

    def get_detections(self):
        with self.lock:
            return list(self.latest_detections)

    def _worker(self):
        while self.running:
            frame = None
            with self.lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process
                    self.frame_to_process = None

            if frame is not None:
                results = model(frame, verbose=False)

                detections = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        {
                            "class": cls_name,
                            "conf": conf,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        }
                    )

                with self.lock:
                    self.latest_detections = detections
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False


class CameraStreamTrack(VideoStreamTrack):
    """
    A video stream track that reads frames from a local web camera (or a dummy frames generator if missing).
    """

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)

        self.yolo_processor = YOLOProcessor()

        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

    @property
    def latest_detections(self):
        return self.yolo_processor.get_detections()

    def _draw_detections(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cls_name = d["class"]
            label = f"{cls_name} {d['conf']:.2f}"

            h = hash(cls_name)
            color = ((h * 31 % 200 + 55), (h * 73 % 200 + 55), (h * 127 % 200 + 55))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
        return frame

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()

        if not ret:
            if frame is None:
                import numpy as np

                frame = np.zeros((480, 640, 3), dtype=np.uint8)

        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        self.yolo_processor.update_frame(frame)

        frame = self._draw_detections(frame, self.latest_detections)

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

        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

    @property
    def latest_detections(self):
        return self.yolo_processor.get_detections()

    def _draw_detections(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cls_name = d["class"]
            label = f"{cls_name} {d['conf']:.2f}"

            h = hash(cls_name)
            color = ((h * 31 % 200 + 55), (h * 73 % 200 + 55), (h * 127 % 200 + 55))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
        return frame

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = None
        if self.connected:
            import numpy as np

            try:
                code, data = self.client.GetImageSample()
                if code == 0:
                    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                else:
                    logger.warning(f"Get image sample error. code: {code}")
            except Exception as e:
                logger.error(f"Error getting Go2 image: {e}")

        if frame is None:
            import numpy as np

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

        self.yolo_processor.update_frame(frame)

        frame = self._draw_detections(frame, self.latest_detections)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame
