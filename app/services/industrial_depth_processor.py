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


def _build_rgbd(rgb: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
    """Stack normalised depth as a 4th channel onto the BGR frame."""
    depth_resized = cv2.resize(depth_raw, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    clipped = np.clip(depth_resized, 0, _DEPTH_MAX_MM).astype(np.float32)
    depth_u8 = (clipped / _DEPTH_MAX_MM * 255).astype(np.uint8)
    return np.dstack([rgb, depth_u8])  # (H, W, 4)


class IndustrialDepthProcessor:
    def __init__(self):
        self.enabled = False

    def process(self, frame: np.ndarray, depth_raw: np.ndarray):
        if not self.enabled or _model is None:
            return frame, []

        rgbd = _build_rgbd(frame, depth_raw)
        results = _model(rgbd, imgsz=_IMGSZ, conf=_CONF, verbose=False)
        result = results[0]
        try:
            annotated = result.plot()[:, :, :3]  # drop alpha if present
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

    def warmup(self) -> None:
        if _model is None:
            return
        dummy_rgb = np.zeros((_IMGSZ, _IMGSZ, 3), dtype=np.uint8)
        dummy_depth = np.zeros((_IMGSZ, _IMGSZ), dtype=np.uint16)
        _model(_build_rgbd(dummy_rgb, dummy_depth), imgsz=_IMGSZ, conf=_CONF, verbose=False)
        logger.info("IndustrialDepth model warmed up.")

    def stop(self):
        pass
