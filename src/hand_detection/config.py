"""
Centralized configuration for hand detection demo.

Switch camera sources and tune inference without command-line args.

Usage:
    from config import CONFIG
    width = CONFIG["display"]["width"]
"""

CONFIG = {
    "display": {
        # Output window size (also used as desired capture size when possible)
        "width": 960,
        "height": 540,
        # Toggle drawing overlays to gain FPS when False
        "draw": True,
        # Window title
        "window_name": "Hand Gesture Recognition",
    },
    "camera": {
        # One of: "opencv" (webcam), "video" (file), "go2" (Unitree)
        "source": "opencv",
        # For OpenCV webcam
        "device": 0,
        # For video file playback
        "video_path": "debug_video.mp4",
        # For Unitree GO2 camera
        "go2": {
            "timeout_sec": 3.0,
            # Optionally initialize the ChannelFactory (recommended)
            "init_channel": True,
        },
    },
    "mediapipe": {
        # Hands model settings
        "use_static_image_mode": False,
        # Note: the original script enforced a minimum of 4 hands.
        # We'll keep that behavior in run.py for parity.
        "max_num_hands": 1,
        "model_complexity": 0,  # 0 or 1
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
    },
    "processing": {
        # Scale factor for the image used for inference (0.3 ~ 1.0). Smaller is faster
        "process_scale": 0.7,
        # Run hand inference every N frames (>=1)
        "infer_every_n": 1,
        # Number of TFLite threads for the classifiers
        "tflite_threads": 2,
    },
    "gestures": {
        # Enable dispatching gestures to a Unitree SportClient (GO2)
        "enable_dispatch": False,
        # Cooldown seconds between repeated dispatches of the same gesture
        "cooldown": 2.0,
    },
}
