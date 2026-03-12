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
    recv() draws the last-known detections onto the CURRENT camera frame so
    the WebRTC stream always flows at camera FPS, never frozen between YOLO runs.
    """

    def __init__(self):
        self.enabled = False
        # maxsize=1: inference_worker always processes the freshest frame
        self._frame_queue = queue.Queue(maxsize=1)
        self._detections = []
        self._det_lock = threading.Lock()
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
                with self._det_lock:
                    self._detections = detections
            except Exception as e:
                logger.error(f"YOLO worker error: {e}")

    def draw(self, frame):
        """
        Draw last-known detections onto the CURRENT camera frame.
        Always returns a live frame — never a stale frozen one.
        """
        with self._det_lock:
            detections = self._detections
        if not detections:
            return frame, []
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['conf']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out, list(detections)

    def stop(self):
        pass
