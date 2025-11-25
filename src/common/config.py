# Hand-detection-like config for common use across modules
CONFIG = {
    "display": {
        "width": 640,
        "height": 480,
            "draw": True,
            "draw_fps": True,
        "window_name": "Computer Vision",
    },
    "camera": {
        "source": "opencv",  # opencv | video | go2
        "device": 0,
        "video_path": "",
        # Request capture resolution (only applied when >0). If either is 0 the
        # native camera dimension is kept. Lower resolutions can improve FPS.
        "width": 640,
        "height": 480,
        "go2": {
            "timeout_sec": 3.0,
            "init_channel": True,
        },
    },
}
