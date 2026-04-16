import logging

import cv2
import numpy as np

_IMGSZ = 640
_CONF = 0.15
_DEPTH_MAX_MM = 5000.0

logger = logging.getLogger(__name__)

_model = None
try:
    from ultralytics import YOLO

    try:
        _model = YOLO("app/models/industrial_depth.engine")
        logger.info("IndustrialDepth: loaded engine model.")
    except Exception as _e:
        logger.warning(f"IndustrialDepth engine load failed ({_e}), falling back to .pt")
        try:
            _model = YOLO("app/models/industrial_depth.pt")
            logger.info("IndustrialDepth: loaded .pt model.")
        except Exception as _e2:
            logger.error(f"IndustrialDepth .pt load also failed ({_e2}). Disabled.")
except ImportError:
    logger.warning("'ultralytics' not found. IndustrialDepth detection disabled.")


def _depth_to_colormap(depth_raw: np.ndarray, target_size: tuple) -> np.ndarray:
    """Convert uint16 depth (mm) to a 3-channel Turbo colormap image."""
    resized = cv2.resize(depth_raw, target_size, interpolation=cv2.INTER_NEAREST)
    clipped = np.clip(resized, 0, _DEPTH_MAX_MM).astype(np.float32)
    depth_u8 = (clipped / _DEPTH_MAX_MM * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)  # (H, W, 3)


class IndustrialDepthProcessor:
    def __init__(self):
        self.enabled = False

    def process(self, frame: np.ndarray, depth_raw: np.ndarray):
        if not self.enabled or _model is None:
            return frame, []

        h, w = frame.shape[:2]
        depth_color = _depth_to_colormap(depth_raw, (w, h))

        results = _model(depth_color, imgsz=_IMGSZ, conf=_CONF, verbose=False)
        result = results[0]

        # Draw detections onto the RGB frame for display
        annotated = frame.copy()
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = _model.names[cls_id]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append({"class": cls_name, "conf": conf,
                                "bbox": (x1, y1, x2, y2)})
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 255), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return annotated, detections

    def warmup(self) -> None:
        if _model is None:
            return
        dummy = np.zeros((_IMGSZ, _IMGSZ), dtype=np.uint16)
        _model(_depth_to_colormap(dummy, (_IMGSZ, _IMGSZ)), imgsz=_IMGSZ, conf=_CONF, verbose=False)
        logger.info("IndustrialDepth model warmed up.")

    def stop(self):
        pass
