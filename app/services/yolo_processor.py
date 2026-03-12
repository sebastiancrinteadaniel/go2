import logging
import threading
import time

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _model = YOLO("app/models/yolov8n.pt")
except ImportError:
    _model = None
    logger.warning("'ultralytics' package not found. Object detection disabled. Please install it later to enable YOLO.")


class YOLOProcessor:
    """
    Runs YOLO inference in a dedicated background thread (mirrors the example
    yolo/run.py pattern). The worker calls result.plot() to produce a fully
    annotated frame and caches it. recv() picks up that pre-drawn frame with
    zero per-frame drawing overhead on the main loop.
    """

    def __init__(self):
        self.enabled = False
        self._lock = threading.Lock()
        self._pending_frame = None
        self._latest_annotated = None
        self._detections = []
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Submit a frame for background detection. Always keeps only the newest."""
        if not self.enabled or _model is None:
            return
        with self._lock:
            self._pending_frame = frame.copy()

    def get_latest(self, raw_frame):
        """
        Return (annotated_frame, detections).
        If a YOLO result is ready, returns the pre-drawn annotated frame.
        Otherwise returns raw_frame unmodified.
        """
        with self._lock:
            annotated = self._latest_annotated
            detections = list(self._detections)
        if annotated is not None:
            return annotated, detections
        return raw_frame, detections

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
                    # Use YOLO's native renderer — same as example inference_worker
                    try:
                        annotated = result.plot()
                    except Exception:
                        annotated = frame
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
                        self._latest_annotated = annotated
                        self._detections = detections
                except Exception as e:
                    logger.error(f"YOLO worker error: {e}")
            else:
                time.sleep(0.001)

    def stop(self):
        pass
