"""
YOLO-specific configuration. Camera/display come from common.config.

Usage:
    from yolo.config import CONFIG
    model_name = CONFIG["yolo"]["model"]
"""

CONFIG = {
    "yolo": {
        "model": "src/yolo/model/yolov8n.pt",
        "conf": 0.25,
        "imgsz": 640,
        # Threading options
        # When enabled, capture/inference/display run in separate threads
        "enable_threads": True,
        # Number of inference worker threads. If > 1, each worker may load
        # its own model instance for better parallelism depending on backend.
        "workers": 1,
        # Max items allowed in queues between stages (match example.py)
        "queue_size": 3,
        # When queues are full, drop newest frame instead of blocking
        "drop_if_full": True,
    },
}
