import itertools
from typing import List, Sequence

import cv2 as cv
import numpy as np
 


def calc_bounding_rect(image, landmarks):
    """Return bounding rectangle [x1,y1,x2,y2] for a Mediapipe hand landmark object.

    Uses a list accumulation instead of repeated np.append to avoid quadratic
    memory copies. Falls back to a minimal rect if landmarks list is empty.
    """
    image_width, image_height = image.shape[1], image.shape[0]

    points = [
        (
            min(int(lm.x * image_width), image_width - 1),
            min(int(lm.y * image_height), image_height - 1),
        )
        for lm in landmarks.landmark
    ]
    if not points:
        return [0, 0, 0, 0]
    landmark_array = np.asarray(points, dtype=np.int32)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """Convert Mediapipe landmarks to a list of [x,y] pixel coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    return [
        [
            min(int(lm.x * image_width), image_width - 1),
            min(int(lm.y * image_height), image_height - 1),
        ]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(landmark_list):
    """Return flattened, normalized landmark list with coordinates relative to first point.

    Avoids deepcopy + per-element mutation; uses list comprehension.
    """
    if not landmark_list:
        return []
    base_x, base_y = landmark_list[0]
    rel = [(x - base_x, y - base_y) for x, y in landmark_list]
    flat = list(itertools.chain.from_iterable(rel))
    max_value = max((abs(v) for v in flat), default=1)
    if max_value == 0:
        return [0 for _ in flat]
    return [v / max_value for v in flat]


 


 


def draw_landmarks(image, landmark_point: Sequence[Sequence[int]]):
    """Draw hand landmarks & connections with outlined style.

    Performance improvements:
    * Reduce duplicate drawing calls by looping through connection definitions.
    * Use polylines for each finger path (two draws for outline + inner style).
    * Minimal branching inside loops.
    """
    if len(landmark_point) < 2:
        return image

    # Define finger paths (ordered indices). These replicate the original visuals.
    finger_paths: List[List[int]] = [
        [2, 3, 4],  # Thumb
        [5, 6, 7, 8],  # Index
        [9, 10, 11, 12],  # Middle
        [13, 14, 15, 16],  # Ring
        [17, 18, 19, 20],  # Little
    ]
    palm_connections = [0, 1, 2, 5, 9, 13, 17, 0]

    def _as_pts(idxs):
        return np.array([landmark_point[i] for i in idxs], dtype=np.int32).reshape(
            -1, 1, 2
        )

    outline_color = (0, 0, 0)
    inner_color = (255, 255, 255)
    outline_thick = 6
    inner_thick = 2

    # Draw palm loop
    palm_pts = _as_pts(palm_connections)
    cv.polylines(image, [palm_pts], False, outline_color, outline_thick, cv.LINE_AA)
    cv.polylines(image, [palm_pts], False, inner_color, inner_thick, cv.LINE_AA)

    # Draw each finger path
    for path in finger_paths:
        pts = _as_pts(path)
        cv.polylines(image, [pts], False, outline_color, outline_thick, cv.LINE_AA)
        cv.polylines(image, [pts], False, inner_color, inner_thick, cv.LINE_AA)

    # Draw circles (landmarks)
    tip_indices = {4, 8, 12, 16, 20}
    for idx, (x, y) in enumerate(landmark_point):
        if x == 0 and y == 0:
            continue  # skip invalid points
        radius = 8 if idx in tip_indices else 5
        cv.circle(image, (x, y), radius, inner_color, -1, cv.LINE_AA)
        cv.circle(image, (x, y), radius, outline_color, 1, cv.LINE_AA)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )


    return image


 


def draw_info(image, fps):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return image
