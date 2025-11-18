from ..common.config import CONFIG as COMMON_CONFIG


CONFIG = {
    # Reuse shared display/camera
    "display": COMMON_CONFIG["display"],
    "camera": COMMON_CONFIG["camera"],

    "mediapipe": {
        # Hands model settings
        "use_static_image_mode": False,
        # Note: the original script enforced a minimum of 4 hands.
        # We'll keep that behavior in run.py for parity.
        "max_num_hands": 10,
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
