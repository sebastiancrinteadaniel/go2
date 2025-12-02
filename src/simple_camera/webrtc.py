import logging
import threading
import time
import sys
import cv2 as cv

from ..common.cameras import create_camera
from ..common.config import CONFIG as COMMON_CONFIG
from ..common.web_server import WebStreamer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize Web Streamer
    streamer = WebStreamer(port=8084)
    streamer.start()

    # Camera Config
    cam_cfg = COMMON_CONFIG["camera"]
    source = cam_cfg["source"]
    do_flip = COMMON_CONFIG["display"].get("flip", False)
    
    # Initialize Camera
    if len(sys.argv) > 1:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        try:
            ChannelFactoryInitialize(0, sys.argv[1])
        except Exception:
            pass

    camera = create_camera(
        source=source,
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        device=cam_cfg["device"],
        video_path=cam_cfg["video_path"],
        go2_timeout=cam_cfg["go2"]["timeout_sec"],
        go2_init_channel=cam_cfg["go2"]["init_channel"],
    )

    logger.info("Simple Camera WebRTC started.")
    
    stop_event = threading.Event()
    
    try:
        prev_time = time.time()
        while not stop_event.is_set():
            frame = camera.read()
            if not frame.ok or frame.image is None:
                time.sleep(0.005)
                continue
                
            img = frame.image
            if do_flip:
                img = cv.flip(img, 1)
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            
            # Push to web streamer
            streamer.put_frame(img)
            
            # Update Stats
            h, w = img.shape[:2]
            streamer.update_stats({
                "fps": fps,
                "width": w,
                "height": h,
                "info": "Raw Feed"
            })
            
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        streamer.stop()
        camera.close()

if __name__ == "__main__":
    main()
