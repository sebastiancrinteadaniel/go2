from ..common.config import CONFIG as COMMON_CONFIG

"""
Depth Camera specific configuration.
"""

CONFIG = {
    # Reuse shared display/camera
    "display": COMMON_CONFIG["display"],
    "camera": COMMON_CONFIG["camera"],

    "depth_camera": {
        # RealSense stream settings
        "width": 320,
        "height": 240,
        "fps": 30,
        
        # Analysis settings
        "model_path": "src/depth_camera/model/yolov8n.pt",
        "depth_threshold_m": 0.5, # Threshold for depth segmentation (meters)
        "min_contour_area": 1000, # Minimum area for contour detection
    }
}
