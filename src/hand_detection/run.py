#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv


import cv2 as cv
import mediapipe as mp

from utils import (
    CvFpsCalc,
    draw_info,
    draw_info_text,
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    draw_bounding_rect,
    draw_landmarks,
    GestureDispatcher,
    build_gesture_actions,
)
from model import KeyPointClassifier

from config import CONFIG
from cameras import create_camera


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

    cap_width = int(disp.get("width", 960))
    cap_height = int(disp.get("height", 540))
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

    use_brect = True

    # Camera preparation -----------------------------------------------------------
    source = cam_cfg.get("source", "opencv")
    device = int(cam_cfg.get("device", 0))
    video_path = cam_cfg.get("video_path", "debug_video.mp4")
    go2 = cam_cfg.get("go2", {})
    camera = create_camera(
        source=source,
        width=cap_width,
        height=cap_height,
        device=device,
        video_path=video_path,
        go2_timeout=float(go2.get("timeout_sec", 3.0)),
        go2_init_channel=bool(go2.get("init_channel", True)),
    )

    # Model load -------------------------------------------------------------------
    mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier(num_threads=tflite_threads)

    # Read labels ------------------------------------------------------------------
    keypoint_classifier_labels = _load_labels()

    # Enable OpenCV optimizations ---------------------------------------------------
    try:
        cv.setUseOptimized(True)
        cv.setNumThreads(max(cv.getNumThreads(), 4))
    except Exception:
        pass

    # FPS Measurement ---------------------------------------------------------------
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Modes removed; only default visualization is kept

    # Optional gesture dispatcher ---------------------------------------------------
    dispatcher = _build_dispatcher_if_enabled(
        enable=bool(gest_cfg.get("enable_dispatch", False)),
        cooldown=float(gest_cfg.get("cooldown", 2.0)),
    )

    frame_idx = 0
    last_results = None
    last_hand_sign_id = None

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) -------------------------------------------------
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # Read frame --------------------------------------------------------------
        frame = camera.read()
        if not frame.ok or frame.image is None:
            # For video files, reaching the end is expected
            if source == "video":
                break
            else:
                continue

        image = frame.image
        # Mirror display (selfie mode) for consistent handedness view
        image = cv.flip(image, 1)
        debug_image = image.copy()

        # HAND DETECTION & PROCESSING --------------------------------------------
        if image is not None:
            # Downscale for faster inference if requested
            if process_scale < 0.999:
                ih, iw = image.shape[:2]
                proc_w = max(64, int(iw * process_scale))
                proc_h = max(64, int(ih * process_scale))
                proc_frame = cv.resize(
                    image, (proc_w, proc_h), interpolation=cv.INTER_AREA
                )
            else:
                proc_frame = image

            # Run inference every N frames, reuse last results otherwise
            do_infer = (frame_idx % infer_every_n) == 0
            if do_infer:
                rgb = cv.cvtColor(proc_frame, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True
                last_results = results
            else:
                results = last_results
            frame_idx += 1
        else:
            continue

        if results and results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
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
                        print(keypoint_classifier_labels[hand_sign_id], hand_sign_id)
                    except Exception:
                        print(f"Gesture id: {hand_sign_id}")
                    last_hand_sign_id = hand_sign_id

                # Dispatch gesture (optional)
                if dispatcher is not None:
                    dispatcher.process(hand_sign_id)

                # Drawing
                if draw_enabled:
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                    )
        else:
            pass

        # Drawing info -------------------------------------------------------------
        if debug_image is not None:
            if draw_enabled:
                debug_image = draw_info(debug_image, fps)

            cv.imshow(window_name, debug_image)

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
