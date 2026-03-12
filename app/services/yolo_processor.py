import logging
import threading
import time
import cv2

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _model = YOLO("app/models/yolov8n.pt")
except ImportError:
    _model = None
    logger.warning("'ultralytics' package not found. Object detection disabled. Please install it later to enable YOLO.")


class YOLOProcessor:
    """
    Runs YOLO inference in a dedicated background thread with a single-slot
    frame buffer. recv() submits each camera frame non-blocking and draws the
    most recent detections immediately — camera FPS is never blocked by YOLO.
    """

    def __init__(self):
        self.enabled = False
        self._lock = threading.Lock()
        self._pending_frame = None
        self._detections = []
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Submit a frame for background detection. Drops the previous pending frame."""
        if not self.enabled or _model is None:
            return
        with self._lock:
            self._pending_frame = frame.copy()

    def get_detections(self):
        """Return the latest detection results (non-blocking)."""
        with self._lock:
            return list(self._detections)

    def draw(self, frame):
        """Draw the latest detections onto the given frame and return it."""
        detections = self.get_detections()
        if not detections:
            return frame
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['conf']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out

    def _worker(self):
        while True:
            frame = None
            with self._lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None

            if frame is not None:
                try:
                    results = _model(frame, verbose=False)
                    result = results[0]
                    detections = []
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = _model.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            "class": cls_name,
                            "conf": conf,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        })
                    with self._lock:
                        self._detections = detections
                except Exception as e:
                    logger.error(f"YOLO worker error: {e}")
            else:
                time.sleep(0.001)

    def stop(self):
        pass
