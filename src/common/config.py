# Hand-detection-like config for common use across modules
CONFIG = {
    "display": {
        "width": 1280,
        "height": 720,
            "draw": True,
            # Separate flag to control FPS overlay independently of other drawings
            "draw_fps": True,
        "window_name": "Computer Vision",
    },
    "camera": {
        "source": "opencv",  # opencv | video | go2
        "device": 0,
        "video_path": "",
        "go2": {
            "timeout_sec": 3.0,
            "init_channel": True,
        },
    },
}
