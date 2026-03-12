import asyncio
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
    Handles YOLO object detection by running inference in a thread-pool executor
    so it never blocks the asyncio event loop. Inference and annotation happen on
    the same frame that is about to be sent, eliminating the stale-detection lag.
    """

    def __init__(self):
        self.enabled = False

    async def process(self, frame):
        """
        Run YOLO inference on the frame if enabled.

        Returns:
            (annotated_frame, detections) where detections is a list of dicts
            with keys 'class', 'conf', 'bbox'.
            If disabled or model unavailable, returns (original_frame, []).
        """
        if not self.enabled or _model is None:
            return frame, []

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: _model(frame, verbose=False)
        )

        result = results[0]
        annotated_frame = result.plot()

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

        return annotated_frame, detections

    def stop(self):
        pass
