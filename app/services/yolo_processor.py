import cv2
import logging
import queue
import threading

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _model = YOLO("app/models/yolov8n.pt")
except ImportError:
    _model = None
    logger.warning("'ultralytics' package not found. Object detection disabled. Please install it later to enable YOLO.")


def _safe_put(q: queue.Queue, item):
    """Non-blocking put — drops the old item if full (always keep newest frame)."""
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


class YOLOProcessor:
    """
    Two-thread pipeline mirroring example/go2/src/yolo/run.py:
      capture (video.py) -> frame_queue -> _inference_worker

    Worker calls result.plot() so boxes are baked into the frame they were
    detected on (always visually in sync). draw() returns that pre-annotated
    frame — same as the example's `last_vis` pattern — so boxes never drift
    against a moving subject.
    """

    def __init__(self):
        self.enabled = False
        # maxsize=1: always process the freshest frame, drop stale ones
        self._frame_queue = queue.Queue(maxsize=1)
        # Latest pre-annotated frame from result.plot() + its detections
        self._latest_annotated = None
        self._latest_detections = []
        self._result_lock = threading.Lock()
        self._thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Push a frame for inference. Drops the previous pending frame if not yet consumed."""
        if not self.enabled or _model is None:
            return
        _safe_put(self._frame_queue, frame)

    def _inference_worker(self):
        """Mirrors inference_worker from example/go2/src/yolo/run.py."""
        while True:
            try:
                img = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                results = _model(img, verbose=False)
                result = results[0]
                # Bake boxes into the frame — same as example's result.plot()
                try:
                    annotated = result.plot()
                except Exception:
                    annotated = img
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
                with self._result_lock:
                    self._latest_annotated = annotated
                    self._latest_detections = detections
            except Exception as e:
                logger.error(f"YOLO worker error: {e}")

    def draw(self, raw_frame):
        """
        Return (frame_to_send, detections).
        If a YOLO result exists, returns the pre-annotated frame where boxes are
        baked in — boxes are always aligned to the image they were detected on.
        Falls back to raw_frame before the first inference completes.
        """
        with self._result_lock:
            annotated = self._latest_annotated
            detections = list(self._latest_detections)
        if annotated is not None:
            return annotated, detections
        return raw_frame, []

    def stop(self):
        pass
