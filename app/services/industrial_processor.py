import logging

import numpy as np

_IMGSZ = 640
_CONF = 0.45

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO

    _model = YOLO("app/models/industrial.pt")
except ImportError:
    _model = None
    logger.warning("'ultralytics' package not found. Industrial detection disabled.")


class IndustrialProcessor:
    def __init__(self):
        self.enabled = False

    def process(self, frame):
        if not self.enabled or _model is None:
            return frame, []

        results = _model(frame, imgsz=_IMGSZ, conf=_CONF, verbose=False)
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
            detections.append(
                {
                    "class": cls_name,
                    "conf": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )
        return annotated, detections

    def warmup(self) -> None:
        if _model is None:
            return
        dummy = np.zeros((_IMGSZ, _IMGSZ, 3), dtype=np.uint8)
        _model(dummy, imgsz=_IMGSZ, conf=_CONF, verbose=False)
        logger.info("Industrial model warmed up.")

    def stop(self):
        pass
