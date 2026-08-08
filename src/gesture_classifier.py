"""Gesture classification – rule‑based fallback and ML model."""

from abc import ABC, abstractmethod
import logging
import numpy as np
import joblib
from typing import Tuple, List

logger = logging.getLogger(__name__)


class BaseGestureClassifier(ABC):
    """Abstract interface for gesture classifiers."""

    @abstractmethod
    def predict(self, landmarks: List[float]) -> Tuple[str, float]:
        """
        Predict gesture from normalized landmarks.
        Returns (gesture_name, confidence).
        """
        pass


class RuleBasedClassifier(BaseGestureClassifier):
    """
    Simple heuristic classifier using hand landmark geometry.
    Used as a fallback or for quick demos without a trained model.
    """

    def __init__(self) -> None:
        # Landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        self.THUMB_IP = 3
        self.INDEX_PIP = 6
        self.MIDDLE_PIP = 10
        self.RING_PIP = 14
        self.PINKY_PIP = 18

    def _get_landmark(self, landmarks: List[float], idx: int) -> np.ndarray:
        """Return (x, y, z) for landmark idx."""
        start = idx * 3
        return np.array(landmarks[start:start+3])

    def _finger_extended(self, tip_idx: int, pip_idx: int, landmarks: List[float]) -> bool:
        """Check if finger is extended (tip y < pip y, assuming origin at top-left)."""
        tip = self._get_landmark(landmarks, tip_idx)
        pip = self._get_landmark(landmarks, pip_idx)
        return tip[1] < pip[1]  # y coordinate decreases upward

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return np.linalg.norm(p1 - p2)

    def predict(self, landmarks: List[float]) -> Tuple[str, float]:
        # Extract points
        thumb_tip = self._get_landmark(landmarks, self.THUMB_TIP)
        index_tip = self._get_landmark(landmarks, self.INDEX_TIP)
        middle_tip = self._get_landmark(landmarks, self.MIDDLE_TIP)
        ring_tip = self._get_landmark(landmarks, self.RING_TIP)
        pinky_tip = self._get_landmark(landmarks, self.PINKY_TIP)
        thumb_ip = self._get_landmark(landmarks, self.THUMB_IP)
        index_pip = self._get_landmark(landmarks, self.INDEX_PIP)
        middle_pip = self._get_landmark(landmarks, self.MIDDLE_PIP)
        ring_pip = self._get_landmark(landmarks, self.RING_PIP)
        pinky_pip = self._get_landmark(landmarks, self.PINKY_PIP)

        fingers_extended = {
            "thumb": thumb_tip[1] < thumb_ip[1],
            "index": self._finger_extended(self.INDEX_TIP, self.INDEX_PIP, landmarks),
            "middle": self._finger_extended(self.MIDDLE_TIP, self.MIDDLE_PIP, landmarks),
            "ring": self._finger_extended(self.RING_TIP, self.RING_PIP, landmarks),
            "pinky": self._finger_extended(self.PINKY_TIP, self.PINKY_PIP, landmarks),
        }

        # Thumbs up: thumb up, all other fingers down
        if fingers_extended["thumb"] and not any([
            fingers_extended["index"], fingers_extended["middle"],
            fingers_extended["ring"], fingers_extended["pinky"]
        ]):
            return "thumbs_up", 0.9

        # Peace: index and middle up, others down
        if fingers_extended["index"] and fingers_extended["middle"] and not fingers_extended["ring"] and not fingers_extended["pinky"]:
            return "peace", 0.9

        # OK: thumb and index tips close, others extended? Heuristic: distance between thumb tip and index tip is small.
        dist_thumb_index = self._distance(thumb_tip, index_tip)
        if dist_thumb_index < 40 and fingers_extended["middle"] and fingers_extended["ring"] and fingers_extended["pinky"]:
            return "ok", 0.8

        # Rock: index and pinky up, others down (thumb can be out). Approx.
        if fingers_extended["index"] and fingers_extended["pinky"] and not fingers_extended["middle"] and not fingers_extended["ring"]:
            return "rock", 0.9

        # Open palm: all five fingers extended
        if all(fingers_extended.values()):
            return "open_palm", 0.9

        # Fist: all fingers down
        if not any(fingers_extended.values()):
            return "fist", 0.9

        # Point left/right: index extended, others down; hand direction based on index tip x vs wrist x? We'll use index tip's x relative to wrist.
        if fingers_extended["index"] and not fingers_extended["middle"] and not fingers_extended["ring"] and not fingers_extended["pinky"]:
            wrist = self._get_landmark(landmarks, self.WRIST)
            # Determine pointing direction: if index tip x < wrist x → pointing left, else right
            if index_tip[0] < wrist[0]:
                return "point_left", 0.85
            else:
                return "point_right", 0.85

        # Finger gun: index and thumb extended, middle/ring/pinky down
        if fingers_extended["index"] and fingers_extended["thumb"] and not fingers_extended["middle"] and not fingers_extended["ring"] and not fingers_extended["pinky"]:
            return "finger_gun", 0.9

        # Fallback
        return "unknown", 0.0


class MLGestureClassifier(BaseGestureClassifier):
    """Gesture classifier using a pre-trained scikit-learn / XGBoost model."""

    def __init__(self, model_path: str, label_map: List[str]) -> None:
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict) and "model" in loaded:
            # New-style artifact from train_classifier.py: the label ordering the
            # model was actually trained on travels with the model, so it's the
            # source of truth. Fall back to the config-supplied label_map only if
            # the artifact didn't include one (defensive, shouldn't normally happen).
            self.model = loaded["model"]
            self.label_map = loaded.get("classes") or label_map
        else:
            # Legacy artifact: a bare estimator saved directly with joblib.dump().
            self.model = loaded
            self.label_map = label_map
        logger.info(f"Loaded ML model from {model_path}")

    def predict(self, landmarks: List[float]) -> Tuple[str, float]:
        if landmarks is None or len(landmarks) != 63:
            return "unknown", 0.0
        X = np.array(landmarks).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            confidence = float(proba[idx])
            gesture = self._resolve_label(idx)
        else:
            raw = self.model.predict(X)[0]
            confidence = 1.0  # models without predict_proba
            gesture = self._resolve_label(raw)
        return gesture, confidence

    def _resolve_label(self, prediction) -> str:
        """Map a raw model prediction to a gesture name.

        Handles both conventions transparently: some estimators (e.g. RandomForest/
        SVM fit directly on strings) return the gesture name itself from predict(),
        while others (e.g. models trained on LabelEncoder output, or the argmax of
        predict_proba) return an integer index into self.label_map. Falls back to
        "unknown" instead of raising if the model and label_map are out of sync.
        """
        if isinstance(prediction, (str, np.str_)):
            return str(prediction)
        try:
            idx = int(prediction)
        except (TypeError, ValueError):
            logger.warning(f"Unrecognized prediction type from model: {prediction!r}")
            return "unknown"
        if 0 <= idx < len(self.label_map):
            return self.label_map[idx]
        logger.warning(f"Predicted index {idx} out of range for label_map of size {len(self.label_map)}")
        return "unknown"
