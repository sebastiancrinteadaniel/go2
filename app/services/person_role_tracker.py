from __future__ import annotations

import logging
import cv2

logger = logging.getLogger(__name__)

# BGR colours for video overlay
_GUARD_COLOR = (50, 205, 50)      # lime green
_PRISONER_COLOR = (30, 30, 220)   # red

# Max centroid distance (pixels) to match the same person across frames
_TRACKING_DISTANCE = 120


class PersonRoleTracker:
    """
    Assigns hardcoded roles to detected persons based on first-seen order:
      - 1st unique person in the session  →  "guard"
      - Every subsequent unique person    →  "prisoner"

    Tracking is centroid-proximity based: if a new detection's centre is
    within _TRACKING_DISTANCE pixels of an already-tracked person it is
    treated as the same individual (role unchanged).

    The tracker relabels the "class" field in each person detection so the
    existing threat-list UI (threats.json already contains "guard" /
    "prisoner" entries) picks them up automatically.

    It also draws its own coloured bounding boxes + labels on the frame,
    overriding YOLO's generic "person" annotation.
    """

    def __init__(self) -> None:
        # list of {"role": str, "centroid": (cx, cy), "bbox": (x1,y1,x2,y2)}
        self._tracked: list[dict] = []
        # guard bboxes seen in the *current* frame — cleared each process() call
        self._active_guard_bboxes: list[tuple] = []

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._tracked = []
        self._active_guard_bboxes = []
        logger.info("PersonRoleTracker: roles reset.")

    def clear_active(self) -> None:
        """Call when YOLO is disabled so no stale guard bboxes remain."""
        self._active_guard_bboxes = []

    # ------------------------------------------------------------------
    def process(self, frame, detections: list[dict]) -> tuple:
        """
        Process a detection list for one frame.

        Returns (annotated_frame, updated_detections) where every "person"
        entry has its class replaced with "guard" or "prisoner" and a
        coloured bounding box is drawn on the frame.
        """
        # Rebuild from scratch each frame — only guards actually in this frame count
        self._active_guard_bboxes = []

        updated = []
        for det in detections:
            if det.get("class") != "person":
                updated.append(det)
                continue

            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            role = self._match_or_create(cx, cy, (x1, y1, x2, y2))

            if role == "guard":
                self._active_guard_bboxes.append((x1, y1, x2, y2))

            new_det = dict(det)
            new_det["class"] = role
            updated.append(new_det)

            # Draw coloured box + label
            color = _GUARD_COLOR if role == "guard" else _PRISONER_COLOR
            label_text = f"{'GUARD' if role == 'guard' else 'PRISONER'} {det['conf']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 1
            )
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame,
                label_text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame, updated

    # ------------------------------------------------------------------
    def is_weapon_authorized(self, weapon_bbox: tuple, margin: int = 60) -> bool:
        """
        Returns True if the weapon's centroid falls within any guard bbox that
        was seen in the *current* frame (expanded by `margin` px for arm reach).
        Returns False immediately when YOLO is off (active list is empty).
        """
        wx1, wy1, wx2, wy2 = weapon_bbox
        wcx = (wx1 + wx2) // 2
        wcy = (wy1 + wy2) // 2
        for gx1, gy1, gx2, gy2 in self._active_guard_bboxes:
            if (
                (gx1 - margin) <= wcx <= (gx2 + margin)
                and (gy1 - margin) <= wcy <= (gy2 + margin)
            ):
                return True
        return False

    # ------------------------------------------------------------------
    def _match_or_create(self, cx: int, cy: int, bbox: tuple) -> str:
        best_match = None
        best_dist = float("inf")
        for tracked in self._tracked:
            tx, ty = tracked["centroid"]
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = tracked

        if best_match is not None and best_dist < _TRACKING_DISTANCE:
            best_match["centroid"] = (cx, cy)
            best_match["bbox"] = bbox
            return best_match["role"]

        # New unique person
        role = "guard" if not self._tracked else "prisoner"
        self._tracked.append({"role": role, "centroid": (cx, cy), "bbox": bbox})
        logger.info(
            "PersonRoleTracker: new person assigned '%s' (total tracked: %d)",
            role,
            len(self._tracked),
        )
        return role
