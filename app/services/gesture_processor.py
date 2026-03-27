import csv
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from scipy.special import softmax  # type: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp  # type: ignore[reportMissingImports]
except ImportError:
    mp = None
    logger.warning("'mediapipe' package not found. Hand gesture inference disabled.")

try:
    import onnxruntime as ort  # type: ignore[reportMissingImports]
    _HAS_ORT = True
except ImportError:
    ort = None
    _HAS_ORT = False
    logger.warning("'onnxruntime' package not found. ONNX gesture classifier disabled.")

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "keypoint_classifier"
_LABELS_PATH = _MODEL_DIR / "keypoint_classifier_label.csv"
_ONNX_PATH = _MODEL_DIR / "keypoint_classifier.onnx"


class GestureProcessor:
    _POINT_COLOR = (80, 22, 10)
    _LINE_COLOR = (80, 44, 121)
    _BOX_COLOR = (80, 44, 121)
    _LABEL_COLOR = (0, 255, 0)
    _HAND_COLOR = (255, 255, 255)

    _INFER_W: int = 640
    _INFER_H: int = 360

    def __init__(self):
        self.enabled = False
        self._hands = None
        self._mp_hands = None
        self._mp_draw = None
        self._point_style = None
        self._line_style = None
        self._ort_session = None
        self._input_name: str = ""
        self._labels: List[str] = []
        self._norm_buf = np.zeros(42, dtype=np.float32)

        self._setup_mediapipe()
        self._load_model_assets()

    _HEART_CONF_THRESHOLD = 0.6

    def _normalize_landmarks(self, landmarks, is_left: bool = False) -> np.ndarray:
        base_x, base_y = landmarks[0].x, landmarks[0].y
        for i, lm in enumerate(landmarks):
            x = lm.x - base_x
            if is_left:
                x = -x  # mirror left hand so it looks like right hand to the model
            self._norm_buf[i * 2] = x
            self._norm_buf[i * 2 + 1] = lm.y - base_y
        max_value = float(np.max(np.abs(self._norm_buf)))
        if max_value > 0:
            self._norm_buf /= max_value
        else:
            self._norm_buf[:] = 0.0
        return self._norm_buf

    def _setup_mediapipe(self) -> None:
        if mp is None:
            return
        if not hasattr(mp, "solutions"):
            logger.warning("mediapipe.solutions is unavailable in this mediapipe build. Gesture inference disabled.")
            return

        try:
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
        except Exception as e:
            logger.warning(f"MediaPipe Hands API unavailable: {e}")
            return

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5,
        )
        self._point_style = self._mp_draw.DrawingSpec(
            color=self._POINT_COLOR, thickness=2, circle_radius=4
        )
        self._line_style = self._mp_draw.DrawingSpec(
            color=self._LINE_COLOR, thickness=2, circle_radius=2
        )
        logger.info("MediaPipe Hands initialised.")

    def _load_model_assets(self) -> None:
        if _LABELS_PATH.exists():
            try:
                with _LABELS_PATH.open(encoding="utf-8-sig") as f:
                    self._labels = [row[0] for row in csv.reader(f) if row]
                logger.info(f"Loaded {len(self._labels)} gesture labels: {self._labels}")
            except Exception as e:
                logger.warning(f"Failed to load gesture labels: {e}")
        else:
            logger.warning(f"Gesture labels not found at {_LABELS_PATH}")

        if not _HAS_ORT:
            logger.info("onnxruntime not available — landmark-only drawing mode.")
            return
        if not _ONNX_PATH.exists():
            logger.warning(f"ONNX model not found at {_ONNX_PATH} — landmark-only drawing mode.")
            return

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ort_session = ort.InferenceSession(str(_ONNX_PATH), providers=providers)
            self._input_name = self._ort_session.get_inputs()[0].name
            logger.info(f"ONNX gesture model loaded. Providers: {self._ort_session.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load ONNX gesture model: {e}")

    def warmup(self) -> None:
        if self._ort_session is None:
            return
        try:
            dummy = np.zeros((1, 42), dtype=np.float32)
            self._ort_session.run(None, {self._input_name: dummy})
            logger.info("ONNX gesture model warmed up.")
        except Exception as e:
            logger.warning(f"Gesture warmup failed: {e}")

    def _classify(self, landmarks, is_left: bool = False) -> Tuple[str, float]:
        if self._ort_session is None or not self._labels:
            return "hand", 0.5
        try:
            self._normalize_landmarks(landmarks, is_left)
            input_data = self._norm_buf[np.newaxis, :]  # shape (1, 42), no copy
            probs = softmax(self._ort_session.run(None, {self._input_name: input_data})[0][0])
            gesture_id = int(np.argmax(probs))
            label = self._labels[gesture_id] if gesture_id < len(self._labels) else "unknown"
            return label, float(probs[gesture_id])
        except Exception as e:
            logger.debug(f"Classification error: {e}")
            return "hand", 0.5

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        if not self.enabled or self._hands is None:
            return frame, []

        frame_out = frame.copy()
        small = cv2.resize(frame, (self._INFER_W, self._INFER_H))
        results = self._hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        if not results or not results.multi_hand_landmarks:
            return frame_out, []

        gestures = []
        h, w, _ = frame.shape

        hand_labels: List[str] = []
        hand_confs: List[float] = []
        hand_centers: List[Tuple[int, int]] = []
        hand_top_ys: List[int] = []

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            self._mp_draw.draw_landmarks(
                frame_out,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._point_style,
                self._line_style,
            )

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            x1 = max(0, int(min(xs) * w) - 20)
            y1 = max(0, int(min(ys) * h) - 20)
            x2 = min(w, int(max(xs) * w) + 20)
            y2 = min(h, int(max(ys) * h) + 20)

            cv2.rectangle(frame_out, (x1, y1), (x2, y2), self._BOX_COLOR, 2)

            is_left = False
            handedness_str = ""
            if results.multi_handedness and i < len(results.multi_handedness):
                try:
                    raw = results.multi_handedness[i].classification[0].label
                    is_left = (raw == "Left")
                    handedness_str = "Right" if raw == "Left" else "Left"
                except Exception:
                    pass

            label, conf = self._classify(hand_landmarks.landmark, is_left)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            hand_labels.append(label)
            hand_confs.append(conf)
            hand_centers.append((cx, cy))
            hand_top_ys.append(y1 - 60)

            cv2.putText(
                frame_out, f"{label} ({conf * 100:.1f}%)",
                (x1, max(20, y1 - 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, self._LABEL_COLOR, 2, cv2.LINE_AA,
            )

            if handedness_str:
                cv2.putText(
                    frame_out, handedness_str,
                    (x1, max(10, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self._HAND_COLOR, 2, cv2.LINE_AA,
                )

            gestures.append({"class": f"gesture:{label}", "conf": conf, "bbox": (x1, y1, x2, y2)})

        if (
            len(hand_labels) == 2
            and "HeartHalf" in self._labels
            and all(lbl == "HeartHalf" for lbl in hand_labels)
            and all(c >= self._HEART_CONF_THRESHOLD for c in hand_confs)
        ):
            anchor_x = (hand_centers[0][0] + hand_centers[1][0]) // 2
            anchor_y = min(hand_top_ys) - 20
            text = "Heart <3"
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
            (tw, _th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.putText(
                frame_out, text,
                (anchor_x - tw // 2, max(30, anchor_y)),
                font, scale, (0, 0, 255), thickness, cv2.LINE_AA,
            )
            heart_conf = min(hand_confs)
            gestures.append({"class": "gesture:heart", "conf": heart_conf, "bbox": None})

        return frame_out, gestures

    def stop(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
