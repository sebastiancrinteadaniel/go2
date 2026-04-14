import colorsys
import logging

import cv2
import numpy as np

_IMGSZ = 640
_CONF = 0.75

logger = logging.getLogger(__name__)

# Classes whose bounding boxes are drawn by a downstream processor (PersonRoleTracker).
# We still include them in the detections list — just skip drawing them here.
_SKIP_DRAW_CLASSES = {"person"}


def _class_color(cls_id: int) -> tuple:
    """Deterministic BGR colour from class ID (golden-ratio hue spacing)."""
    hue = (cls_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
    return (int(b * 255), int(g * 255), int(r * 255))

_model = None
try:
    from ultralytics import YOLO

    try:
        _model = YOLO("app/models/yolov8n.engine")
        logger.info("YOLO: loaded engine model.")
    except Exception as _e:
        logger.warning(f"YOLO engine load failed ({_e}), falling back to .pt")
        try:
            _model = YOLO("app/models/yolov8n.pt")
            logger.info("YOLO: loaded .pt model.")
        except Exception as _e2:
            logger.error(f"YOLO .pt load also failed ({_e2}). Object detection disabled.")
except ImportError:
    logger.warning("'ultralytics' package not found. Object detection disabled.")


class YOLOProcessor:
    def __init__(self):
        self.enabled = False

    def process(self, frame):
        if not self.enabled or _model is None:
            return frame, []

        results = _model(frame, imgsz=_IMGSZ, conf=_CONF, verbose=False)
        result = results[0]

        annotated = frame.copy()
        detections = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = _model.names[cls_id]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

            detections.append(
                {"class": cls_name, "conf": conf, "bbox": (x1, y1, x2, y2)}
            )

            if cls_name in _SKIP_DRAW_CLASSES:
                continue  # PersonRoleTracker owns the visual for these

            color = _class_color(cls_id)
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return annotated, detections

    def warmup(self) -> None:
        """Run one dummy inference to JIT-compile/warm up CUDA kernels."""
        if _model is None:
            return
        dummy = np.zeros((_IMGSZ, _IMGSZ, 3), dtype=np.uint8)
        _model(dummy, imgsz=_IMGSZ, conf=_CONF, verbose=False)
        logger.info("YOLO model warmed up.")

    def stop(self):
        pass
