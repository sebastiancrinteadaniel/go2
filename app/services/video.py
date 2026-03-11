import cv2
import logging
import time

from aiortc import VideoStreamTrack
from av import VideoFrame
from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    model = YOLO("app/models/yolov8n.pt")
except ImportError:
    model = None
    logger.warning("'ultralytics' package not found. Object detection disabled. Please install it later to enable YOLO.")



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
        self.latest_detections = []
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

    def _apply_yolo_inference(self, frame):
        """
        Applies YOLO object detection to the frame.
        """
        self.latest_detections = []
        if model is not None:
            results = model(frame, verbose=False)
            
            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                detections.append({"class": cls_name, "conf": conf})
            self.latest_detections = detections

            return results[0].plot()
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

        frame = self._apply_yolo_inference(frame)

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
            logger.warning("'unitree_sdk2py' not found. Ensure it is installed for the Go2 camera stream to work.")
        except Exception as e:
            self.connected = False
            logger.error(f"Error initializing Go2 VideoClient: {e}")

        self.latest_detections = []
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

    def _apply_yolo_inference(self, frame):
        """
        Applies YOLO object detection to the frame.
        """
        self.latest_detections = []
        if model is not None:
            results = model(frame, verbose=False)
            
            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                detections.append({"class": cls_name, "conf": conf})
            self.latest_detections = detections

            return results[0].plot()
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
                    logger.info(f"Go2 Camera Resolution: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    logger.warning(f"Get image sample error. code: {code}")
            except Exception as e:
                logger.error(f"Error getting Go2 image: {e}")

        if frame is None:
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "GO2 CAMERA UNAVAILABLE", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        frame = self._apply_yolo_inference(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame
