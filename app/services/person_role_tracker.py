from __future__ import annotations

import logging
import cv2

logger = logging.getLogger(__name__)

_PRISONER_COLOR = (30, 30, 220)   # red


class PersonRoleTracker:
    """Labels every detected person as "prisoner"."""

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def clear_active(self) -> None:
        pass

    def process(self, frame, detections: list[dict]) -> tuple:
        updated = []
        for det in detections:
            if det.get("class") != "person":
                updated.append(det)
                continue

            x1, y1, x2, y2 = det["bbox"]
            new_det = dict(det)
            new_det["class"] = "prisoner"
            updated.append(new_det)

            label_text = f"PRISONER {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), _PRISONER_COLOR, 2)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), _PRISONER_COLOR, -1)
            cv2.putText(
                frame, label_text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return frame, updated

    def is_weapon_authorized(self, weapon_bbox: tuple, margin: int = 60) -> bool:
        """No guards exist, so no weapon is ever authorized."""
        return False
