from collections import deque
import cv2 as cv


class CvFpsCalc:
    def __init__(self, buffer_len: int = 1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self) -> float:
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        return round(fps, 2)


def draw_info_fps(
    image,
    lines,
    font=cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    thickness: int = 2,
    margin: int = 8,
    line_gap: int = 6,
):
    """Draw a black bar at the top with white info text.

    Parameters
    - image: np.ndarray BGR frame (modified in place and returned)
    - lines: Iterable of strings to render in order (e.g., FPS values)
    - font, font_scale, thickness: OpenCV text parameters
    - margin: inner padding around text (px)
    - line_gap: vertical spacing between lines (px)

    Returns
    - The image with the top bar and text rendered.
    """
    if image is None:
        return image

    try:
        h, w = image.shape[:2]
    except Exception:
        return image

    # Normalize lines to list of strings
    if lines is None:
        lines = []
    elif isinstance(lines, (str, bytes)):
        lines = [str(lines)]
    else:
        lines = [str(x) for x in lines]

    if not lines:
        return image

    # Compute bar height based on text sizes
    text_heights = []
    baseline = 0
    for t in lines:
        (tw, th), bl = cv.getTextSize(t, font, font_scale, thickness)
        text_heights.append(max(th, 1))
        baseline = max(baseline, bl)

    total_text_height = sum(text_heights) + line_gap * (len(lines) - 1)
    bar_height = 2 * margin + total_text_height + baseline
    bar_height = max(bar_height, margin * 2 + 16)
    bar_height = min(bar_height, h)  # don't exceed frame height

    # Draw black bar
    cv.rectangle(image, (0, 0), (w, bar_height), (0, 0, 0), thickness=-1)

    # Draw each line in white
    y = margin + text_heights[0]
    for idx, t in enumerate(lines):
        cv.putText(
            image,
            t,
            (margin, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv.LINE_AA,
        )
        if idx + 1 < len(lines):
            y += text_heights[idx + 1] + line_gap

    return image
