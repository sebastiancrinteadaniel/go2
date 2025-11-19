import sys
import threading
import queue
import time
from dataclasses import dataclass
from typing import Any


from ..common.cameras import create_camera
from ..common.config import CONFIG as COMMON_CONFIG
from .config import CONFIG

import cv2 as cv

from ..common.fps import CvFpsCalc, draw_info_fps


from ultralytics import YOLO


@dataclass
class YoloCtx:
    camera: Any
    source: str
    do_flip: bool
    imgsz: int
    conf: float
    frame_queue: queue.Queue
    result_queue: queue.Queue
    drop_if_full: bool
    stop_event: threading.Event
    fps_lock: threading.Lock

    camera_fps_val: float = 0.0
    yolo_fps_val: float = 0.0


def safe_put(q: queue.Queue, item: Any, drop_if_full: bool):
    if not drop_if_full:
        q.put(item)
        return
    try:
        q.put(item, block=False)
    except queue.Full:
        pass


def capture_loop(ctx: YoloCtx):
    frame_id = 0
    prev_time = time.time()
    while not ctx.stop_event.is_set():
        frame = ctx.camera.read()
        if not frame.ok or frame.image is None:
            if ctx.source == "video":
                break
            else:
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
    # Signal end-of-stream to workers
    safe_put(ctx.frame_queue, (None, None), False)


def inference_worker(ctx: YoloCtx, mdl: Any):
    prev_time = time.time()
    while not ctx.stop_event.is_set():
        try:
            fid, img = ctx.frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if fid is None or img is None:
            # Propagate sentinel for other workers and exit
            safe_put(ctx.frame_queue, (None, None), False)
            break
        try:
            results = mdl(img, imgsz=ctx.imgsz, conf=ctx.conf, verbose=False)
            try:
                vis = results[0].plot()
            except Exception:
                vis = img
            # Block when full to keep pace with display (example parity)
            try:
                ctx.result_queue.put((fid, vis, results), block=True)
            except Exception:
                pass
            now = time.time()
            with ctx.fps_lock:
                ctx.yolo_fps_val = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
        except Exception as e:
            print(f"Worker inference error: {e}")
        finally:
            ctx.frame_queue.task_done()


def main():
    disp = COMMON_CONFIG.get("display", {})
    cam_cfg = COMMON_CONFIG.get("camera", {})
    yolo_cfg = CONFIG.get("yolo", {})

    # Read values purely from config dicts
    source = str(cam_cfg.get("source", "opencv"))
    device = int(cam_cfg.get("device", 0))
    video_path = str(cam_cfg.get("video_path", ""))
    go2 = cam_cfg.get("go2", {})
    go2_timeout = float(go2.get("timeout_sec", 3.0))
    go2_init_channel = bool(go2.get("init_channel", True))

    # width/height are only used for display resizing; keep disabled to maximize FPS
    width = int(disp.get("width", 640))
    height = int(disp.get("height", 480))
    window = str(disp.get("window_name", "YOLO"))
    do_flip = bool(disp.get("flip", True))
    draw_fps_enabled = bool(disp.get("draw_fps", True))

    model_path = str(yolo_cfg.get("model", "yolov8n.pt"))
    conf = float(yolo_cfg.get("conf", 0.25))
    imgsz = int(yolo_cfg.get("imgsz", 640))
    enable_threads = bool(yolo_cfg.get("enable_threads", True))
    workers = max(1, int(yolo_cfg.get("workers", 1)))
    qsize = max(1, int(yolo_cfg.get("queue_size", 2)))
    drop_if_full = bool(yolo_cfg.get("drop_if_full", True))

    # Create camera without forcing resolution to better match example behavior
    camera = create_camera(
        source=source,
        width=0,
        height=0,
        device=device,
        video_path=video_path,
        go2_timeout=go2_timeout,
        go2_init_channel=go2_init_channel,
    )

    # Load model (one eager instance; more may be created for extra workers)
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load YOLO model '{model_path}': {e}")
        sys.exit(1)

    # FPS counter (shared implementation)
    try:
        cvFpsCalc = CvFpsCalc(buffer_len=10)
    except Exception:
        cvFpsCalc = None
    fps = 0.0

    if not enable_threads:
        # Single-threaded fallback (original behavior)
        while True:
            key = cv.waitKey(1)
            if key == 27:
                break

            frame = camera.read()
            if not frame.ok or frame.image is None:
                if source == "video":
                    break
                else:
                    continue

            img = frame.image
            if do_flip:
                img = cv.flip(img, 1)

            try:
                results = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            try:
                vis = results[0].plot()
            except Exception:
                vis = img

            if cvFpsCalc is not None:
                fps = cvFpsCalc.get()

            if draw_fps_enabled:
                vis = draw_info_fps(
                    vis, f"display {fps:.1f}", font_scale=0.6, thickness=1, margin=6
                )

            cv.imshow(window, vis)

        camera.release()
        cv.destroyAllWindows()
        return

    # Multithreaded pipeline: capture -> inference worker(s) -> display
    frame_queue = queue.Queue(maxsize=qsize)
    result_queue = queue.Queue(maxsize=qsize)
    stop_event = threading.Event()
    fps_lock = threading.Lock()

    ctx = YoloCtx(
        camera=camera,
        source=source,
        do_flip=do_flip,
        imgsz=imgsz,
        conf=conf,
        frame_queue=frame_queue,
        result_queue=result_queue,
        drop_if_full=drop_if_full,
        stop_event=stop_event,
        fps_lock=fps_lock,
    )

    # Start capture thread
    cap_thread = threading.Thread(
        target=capture_loop, args=(ctx,), name="capture", daemon=True
    )
    cap_thread.start()

    # Start worker threads
    worker_threads = []
    if workers <= 1:
        worker_threads.append(
            threading.Thread(
                target=inference_worker, args=(ctx, model), name="worker-0", daemon=True
            )
        )
        worker_threads[0].start()
    else:
        # Create additional model instances per worker for safety/perf
        for wi in range(workers):
            try:
                mdl = YOLO(model_path) if wi > 0 else model
            except Exception as e:
                print(f"Worker {wi} failed to init model: {e}")
                mdl = model
            t = threading.Thread(
                target=inference_worker,
                args=(ctx, mdl),
                name=f"worker-{wi}",
                daemon=True,
            )
            t.start()
            worker_threads.append(t)

    last_vis = None
    try:
        while True:
            key = cv.waitKey(1)
            if key == 27:
                break
            if not result_queue.empty():
                try:
                    fid, vis, _ = result_queue.get(block=False)
                    last_vis = vis
                except queue.Empty:
                    pass

            if last_vis is None:
                continue

            if cvFpsCalc is not None:
                fps = cvFpsCalc.get()
            with ctx.fps_lock:
                cam_fps = ctx.camera_fps_val
                det_fps = ctx.yolo_fps_val

            # Draw unified black top bar with white text (single row, smaller)
            if draw_fps_enabled:
                try:
                    info_line = (
                        f"display {fps:.1f}   camera {cam_fps:.1f}   yolo {det_fps:.1f}"
                    )
                    last_vis = draw_info_fps(
                        last_vis, info_line, font_scale=0.6, thickness=1, margin=6
                    )
                except Exception:
                    pass

            cv.imshow(window, last_vis)
    finally:
        stop_event.set()
        # Let queues drain briefly
        time.sleep(0.05)
        try:
            for _ in range(workers):
                safe_put(frame_queue, (None, None))
        except Exception:
            pass
        # Join threads
        cap_thread.join(timeout=1.0)
        for t in worker_threads:
            t.join(timeout=1.0)
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
