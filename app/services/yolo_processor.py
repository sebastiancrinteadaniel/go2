import threading
import time
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    model = YOLO("app/models/yolov8n.pt")
except ImportError:
    model = None
    logger.warning("'ultralytics' package not found. Object detection disabled. Please install it later to enable YOLO.")


class YOLOProcessor:
    """
    Handles YOLO object detection in a background thread so it doesn't block the video stream.
    """
    def __init__(self):
        self.enabled = False
        self.latest_detections = []
        self.frame_to_process = None
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        if model is not None:
            self.thread.start()

    def update_frame(self, frame):
        if frame is None or not self.enabled:
            return
        
        if self.lock.acquire(blocking=False):
            self.frame_to_process = frame.copy()
            self.lock.release()

    def get_detections(self):
        if not self.enabled:
            return []
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
                    detections.append({
                        "class": cls_name, 
                        "conf": conf,
                        "bbox": (int(x1), int(y1), int(x2), int(y2))
                    })
                
                with self.lock:
                    self.latest_detections = detections
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False
