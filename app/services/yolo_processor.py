import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _model = YOLO("app/models/yolov8n.pt")
except ImportError:
    _model = None
    logger.warning("'ultralytics' package not found. Object detection disabled. Please install it later to enable YOLO.")


class YOLOProcessor:
    """
    Pure synchronous YOLO inference.
    Threading / queuing is handled by the caller (video.py) exactly as in
    example/go2/src/yolo/run.py — capture_loop -> frame_queue ->
    inference_worker -> result_queue -> recv().
    """

    def __init__(self):
        self.enabled = False

    def process(self, frame):
        """
        Run YOLO on frame synchronously.
        Returns (annotated_frame, detections) where annotated_frame has boxes
        baked in via result.plot() — same as the example inference_worker.
        If disabled or model unavailable returns (frame, []) immediately.
        """
        if not self.enabled or _model is None:
            return frame, []

        results = _model(frame, verbose=False)
        result = results[0]
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
        return annotated, detections

    def stop(self):
        pass
