import logging
import threading
import queue
import time
import sys

from ..common.cameras import create_camera
from ..common.config import CONFIG as COMMON_CONFIG
from .config import CONFIG
from ..common.web_server import WebStreamer

import cv2 as cv
from ultralytics import YOLO

# Re-use logic from run.py where possible, but we need to override the display part
from .run import YoloCtx, safe_put, inference_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def capture_loop(ctx: YoloCtx):
    # Similar to run.py but we don't need to handle "video" source break logic strictly if we want continuous stream
    frame_id = 0
    prev_time = time.time()
    while not ctx.stop_event.is_set():
        frame = ctx.camera.read()
        if not frame.ok or frame.image is None:
            time.sleep(0.005)
            continue
            
        img = frame.image
        if ctx.do_flip:
            img = cv.flip(img, 1)
            
        now = time.time()
        with ctx.fps_lock:
            ctx.camera_fps_val = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now
        
        safe_put(ctx.frame_queue, (frame_id, img), ctx.drop_if_full)
        frame_id += 1
    
    safe_put(ctx.frame_queue, (None, None), False)

def main():
    # Initialize Web Streamer
    streamer = WebStreamer(port=8081) # Use 8081 for YOLO
    streamer.start()

    # Config
    yolo_cfg = CONFIG["yolo"]
    model_path = yolo_cfg["model"]
    conf_thres = yolo_cfg["conf"]
    imgsz = yolo_cfg["imgsz"]
    
    # Camera
    cam_cfg = COMMON_CONFIG["camera"]
    source = cam_cfg["source"]
    
    # Initialize Camera
    # If running on Jetson/DDS, ensure we pass the interface if provided
    if len(sys.argv) > 1:
        # Hack: create_camera might use sys.argv or we need to set it manually if it uses ChannelFactoryInitialize
        # The create_camera function in common/cameras.py handles ChannelFactoryInitialize internally if needed
        # We just need to make sure it uses the right interface.
        # Assuming create_camera handles it or we initialize before.
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

    # Context
    ctx = YoloCtx(
        camera=camera,
        source=source,
        do_flip=COMMON_CONFIG["display"].get("flip", False),
        imgsz=imgsz,
        conf=conf_thres,
        frame_queue=queue.Queue(maxsize=yolo_cfg["queue_size"]),
        result_queue=queue.Queue(maxsize=yolo_cfg["queue_size"]),
        drop_if_full=yolo_cfg["drop_if_full"],
        stop_event=threading.Event(),
        fps_lock=threading.Lock()
    )

    # Threads
    cap_thread = threading.Thread(target=capture_loop, args=(ctx,), daemon=True)
    cap_thread.start()

    # Load Model
    logger.info(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Inference Thread
    inf_thread = threading.Thread(target=inference_worker, args=(ctx, model), daemon=True)
    inf_thread.start()

    logger.info("YOLO WebRTC started.")

    try:
        while True:
            try:
                fid, vis, results = ctx.result_queue.get(timeout=1.0)
                # Push to web streamer
                streamer.put_frame(vis)
                
                # Update Stats
                with ctx.fps_lock:
                    h, w = vis.shape[:2]
                    num_objects = len(results[0].boxes) if results else 0
                    streamer.update_stats({
                        "fps": ctx.camera_fps_val,
                        "width": w,
                        "height": h,
                        "info": f"Objects: {num_objects}"
                    })
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        ctx.stop_event.set()
        streamer.stop()
        cap_thread.join()
        inf_thread.join()
        camera.close()

if __name__ == "__main__":
    main()
