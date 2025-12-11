import logging
import sys
import time

import cv2 as cv
import mediapipe as mp

from ..common.cameras import create_camera
from ..common.config import CONFIG as COMMON_CONFIG
from ..common.web_server import WebStreamer
from .config import CONFIG

from .utils import (
    draw_info_text,
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    draw_bounding_rect,
    draw_landmarks,
)
from .model import KeyPointClassifier
from .run import _load_labels, _build_dispatcher_if_enabled

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    # Initialize Web Streamer
    streamer = WebStreamer(port=8082) # Use 8082 for Hand Detection
    streamer.start()

    # Config
    cam_cfg = CONFIG["camera"]
    mp_cfg = CONFIG["mediapipe"]
    gest_cfg = CONFIG.get("gestures", {})

    use_static_image_mode = bool(mp_cfg.get("use_static_image_mode", False))
    max_num_hands = max(4, int(mp_cfg.get("max_num_hands", 1)))
    min_detection_confidence = float(mp_cfg.get("min_detection_confidence", 0.7))
    min_tracking_confidence = float(mp_cfg.get("min_tracking_confidence", 0.5))

    # Camera Init
    if len(sys.argv) > 1:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        try:
            ChannelFactoryInitialize(0, sys.argv[1])
        except Exception:
            pass

    source = cam_cfg.get("source", "opencv")
    camera = create_camera(
        source=source,
        width=int(cam_cfg.get("width", 0)),
        height=int(cam_cfg.get("height", 0)),
        device=int(cam_cfg.get("device", 0)),
        video_path=cam_cfg.get("video_path", "debug_video.mp4"),
        go2_timeout=float(cam_cfg.get("go2", {}).get("timeout_sec", 3.0)),
        go2_init_channel=bool(cam_cfg.get("go2", {}).get("init_channel", True)),
    )

    # Models
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    keypoint_classifier = KeyPointClassifier()
    keypoint_classifier_labels = _load_labels()

    # Dispatcher
    dispatcher = _build_dispatcher_if_enabled(
        gest_cfg.get("enable_dispatch", False),
        gest_cfg.get("cooldown", 2.0)
    )

    logger.info("Hand Detection WebRTC started.")

    fps_val = 0.0
    prev_time = time.time()

    try:
        while True:
            frame_data = camera.read()
            if not frame_data.ok or frame_data.image is None:
                time.sleep(0.01)
                continue

            now = time.time()
            fps_val = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            streamer.update_fps(fps_val)

            image = frame_data.image
            if COMMON_CONFIG["display"].get("flip", False):
                image = cv.flip(image, 1)
            
            debug_image = image.copy()

            # Process
            image.flags.writeable = False
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    
                    info_text = handedness.classification[0].label[0:]
                    if keypoint_classifier_labels[hand_sign_id] != "":
                        info_text = info_text + ":" + keypoint_classifier_labels[hand_sign_id]
                    debug_image = draw_info_text(debug_image, brect, handedness, info_text)

                    # Dispatch
                    if dispatcher:
                        dispatcher.process(hand_sign_id)

            # Update Stats
            h, w = debug_image.shape[:2]
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            streamer.update_stats({
                "fps": fps_val,
                "width": w,
                "height": h,
                "info": f"Hands: {num_hands}"
            })

            # Stream
            streamer.put_frame(debug_image)

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        camera.release()

if __name__ == "__main__":
    main()
