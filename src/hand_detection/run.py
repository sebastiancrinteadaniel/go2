#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import threading
import queue
import time
from dataclasses import dataclass
from typing import Any

import cv2 as cv
import mediapipe as mp



from .utils import (
    draw_info_text,
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    draw_bounding_rect,
    draw_landmarks,
    GestureDispatcher,
    build_gesture_actions,
)
from .model import KeyPointClassifier

from .config import CONFIG
from ..common.cameras import create_camera
from ..common.fps import CvFpsCalc


@dataclass
class HandCtx:
    camera: Any
    source: str
    do_flip: bool
    process_scale: float
    infer_every_n: int
    tflite_threads: int
    # Mediapipe params
    use_static_image_mode: bool
    max_num_hands: int
    model_complexity: int
    min_detection_confidence: float
    min_tracking_confidence: float
    draw_enabled: bool
    use_brect: bool
    window_name: str
    enable_dispatch: bool
    dispatch_cooldown: float
    frame_queue: "queue.Queue[tuple[int, Any] | tuple[None, None]]"
    result_queue: "queue.Queue[tuple[int, Any]]"
    stop_event: threading.Event
    fps_lock: threading.Lock
    camera_fps: float = 0.0
    infer_fps: float = 0.0


def _safe_put(q: queue.Queue, item: Any, drop_if_full: bool = True):
    if not drop_if_full:
        q.put(item)
        return
    try:
        q.put(item, block=False)
    except queue.Full:
        pass


def _capture_thread(ctx: HandCtx):
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

        now = time.time()
        with ctx.fps_lock:
            ctx.camera_fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now

        _safe_put(ctx.frame_queue, (frame_id, img), drop_if_full=True)
        frame_id += 1

    # Signal end-of-stream
    _safe_put(ctx.frame_queue, (None, None), drop_if_full=False)


def _inference_thread(ctx: HandCtx):
    # Initialize per-thread resources for safety
    mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
    hands = mp_hands.Hands(
        static_image_mode=ctx.use_static_image_mode,
        max_num_hands=ctx.max_num_hands,
        model_complexity=ctx.model_complexity,
        min_detection_confidence=ctx.min_detection_confidence,
        min_tracking_confidence=ctx.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier(num_threads=ctx.tflite_threads)
    labels = _load_labels()
    dispatcher = _build_dispatcher_if_enabled(ctx.enable_dispatch, ctx.dispatch_cooldown)

    last_results = None
    last_hand_sign_id = None
    frame_idx = 0
    prev_time = time.time()

    while not ctx.stop_event.is_set():
        try:
            fid, image = ctx.frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if fid is None or image is None:
            # propagate sentinel and exit
            _safe_put(ctx.frame_queue, (None, None), drop_if_full=False)
            break

        # Mirror display (selfie view) done outside capture to keep camera lightweight
        if ctx.do_flip:
            image = cv.flip(image, 1)
        debug_image = image.copy()

        # Downscale for faster inference if requested
        if ctx.process_scale < 0.999:
            ih, iw = image.shape[:2]
            proc_w = max(64, int(iw * ctx.process_scale))
            proc_h = max(64, int(ih * ctx.process_scale))
            proc_frame = cv.resize(image, (proc_w, proc_h), interpolation=cv.INTER_AREA)
        else:
            proc_frame = image

        # Run inference every N frames, reuse last results otherwise
        do_infer = (frame_idx % ctx.infer_every_n) == 0
        if do_infer:
            rgb = cv.cvtColor(proc_frame, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            last_results = results
        else:
            results = last_results
        frame_idx += 1

        if results and results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # Preprocess
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Throttle printing to only when the sign changes
                if hand_sign_id != last_hand_sign_id:
                    try:
                        print(labels[hand_sign_id], hand_sign_id)
                    except Exception:
                        print(f"Gesture id: {hand_sign_id}")
                    last_hand_sign_id = hand_sign_id

                # Dispatch gesture (optional)
                if dispatcher is not None:
                    dispatcher.process(hand_sign_id)

                # Drawing
                if ctx.draw_enabled:
                    debug_image = draw_bounding_rect(ctx.use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        labels[hand_sign_id] if 0 <= hand_sign_id < len(labels) else "",
                    )

        # Update inference FPS
        now = time.time()
        with ctx.fps_lock:
            ctx.infer_fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now

        # Enqueue annotated frame (block to keep pace with display)
        try:
            ctx.result_queue.put((fid, debug_image), block=True)
        except Exception:
            pass

        ctx.frame_queue.task_done()



def _load_labels():
    with open(
        "src/hand_detection/model/keypoint_classifier/keypoint_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        keypoint_labels = [row[0] for row in csv.reader(f)]
    return keypoint_labels


def _build_dispatcher_if_enabled(enable: bool, cooldown: float):
    if not enable:
        return None
    # Lazy import to avoid initializing when not needed
    try:
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        sport_client = SportClient()
        sport_client.SetTimeout(10.0)
        sport_client.Init()
        return GestureDispatcher(build_gesture_actions(sport_client), cooldown=cooldown)
    except Exception as e:
        print(f"Gesture dispatcher disabled (initialization failed): {e}")
        return None


def main():
    # Read config ------------------------------------------------------------------
    disp = CONFIG["display"]
    cam_cfg = CONFIG["camera"]
    mp_cfg = CONFIG["mediapipe"]
    proc_cfg = CONFIG["processing"]
    gest_cfg = CONFIG.get("gestures", {})

    # Optional display sizing (not applied to capture to preserve FPS)
    draw_enabled = bool(disp.get("draw", True))
    window_name = disp.get("window_name", "Hand Gesture Recognition")

    use_static_image_mode = bool(mp_cfg.get("use_static_image_mode", False))
    # Keep original behavior: min 4 hands
    max_num_hands = max(4, int(mp_cfg.get("max_num_hands", 1)))
    model_complexity = int(mp_cfg.get("model_complexity", 0))
    min_detection_confidence = float(mp_cfg.get("min_detection_confidence", 0.7))
    min_tracking_confidence = float(mp_cfg.get("min_tracking_confidence", 0.5))

    process_scale = max(0.3, min(1.0, float(proc_cfg.get("process_scale", 0.7))))
    infer_every_n = max(1, int(proc_cfg.get("infer_every_n", 1)))
    tflite_threads = max(1, int(proc_cfg.get("tflite_threads", 2)))

    width = int(disp.get("width", 640))
    height = int(disp.get("height", 480))

    use_brect = True

    # Camera preparation -----------------------------------------------------------
    source = cam_cfg.get("source", "opencv")
    device = int(cam_cfg.get("device", 0))
    video_path = cam_cfg.get("video_path", "debug_video.mp4")
    go2 = cam_cfg.get("go2", {})
    # Open camera at native resolution to maximize capture FPS (resize later if needed)
    camera = create_camera(
        source=source,
        width=0,
        height=0,
        device=device,
        video_path=video_path,
        go2_timeout=float(go2.get("timeout_sec", 3.0)),
        go2_init_channel=bool(go2.get("init_channel", True)),
    )

    # Enable OpenCV optimizations ---------------------------------------------------
    try:
        cv.setUseOptimized(True)
        cv.setNumThreads(max(cv.getNumThreads(), 4))
    except Exception:
        pass

    # FPS Measurement ---------------------------------------------------------------
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    # Threaded pipeline ------------------------------------------------------------
    frame_queue = queue.Queue(maxsize=3)
    result_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    fps_lock = threading.Lock()

    ctx = HandCtx(
        camera=camera,
        source=source,
        do_flip=True,
        process_scale=process_scale,
        infer_every_n=infer_every_n,
        tflite_threads=tflite_threads,
        use_static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        draw_enabled=draw_enabled,
        use_brect=use_brect,
        window_name=window_name,
        enable_dispatch=bool(gest_cfg.get("enable_dispatch", False)),
        dispatch_cooldown=float(gest_cfg.get("cooldown", 2.0)),
        frame_queue=frame_queue,
        result_queue=result_queue,
        stop_event=stop_event,
        fps_lock=fps_lock,
    )

    t_cam = threading.Thread(target=_capture_thread, args=(ctx,), name="hand-cam", daemon=True)
    t_inf = threading.Thread(target=_inference_thread, args=(ctx,), name="hand-inf", daemon=True)
    t_cam.start()
    t_inf.start()

    last_vis = None
    try:
        while True:
            disp_fps = cvFpsCalc.get()
            if cv.waitKey(1) & 0xFF == 27:
                break

            if not result_queue.empty():
                try:
                    _, vis = result_queue.get(block=False)
                    last_vis = vis
                except queue.Empty:
                    pass

            if last_vis is None:
                continue


            shown = last_vis
            # if width and height:
            #     shown = cv.resize(shown, (width, height), interpolation=cv.INTER_AREA)
               
            # cv.imshow(window, shown)

            # Overlay FPS metrics
            with ctx.fps_lock:
                cam_fps = ctx.camera_fps
                inf_fps = ctx.infer_fps
            cv.putText(last_vis, f"Camera FPS: {cam_fps:.2f}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv.putText(last_vis, f"Infer FPS: {inf_fps:.2f}", (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv.putText(last_vis, f"Display FPS: {disp_fps:.2f}", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv.imshow(window_name, shown)
    finally:
        stop_event.set()
        time.sleep(0.05)
        _safe_put(frame_queue, (None, None), drop_if_full=False)
        t_cam.join(timeout=1.0)
        t_inf.join(timeout=1.0)
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
