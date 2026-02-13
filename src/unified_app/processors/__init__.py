"""Vision Processors - Individual implementations of BaseProcessor"""

from .yolo_processor import YoloProcessor
from .hand_detection_processor import HandDetectionProcessor
from .depth_camera_processor import DepthCameraProcessor
from .simple_camera_processor import SimpleCameraProcessor

__all__ = [
    "YoloProcessor",
    "HandDetectionProcessor",
    "DepthCameraProcessor",
    "SimpleCameraProcessor",
]
