"""Hand landmark detection using MediaPipe Hands."""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class HandDetector:
    """Wrapper around MediaPipe Hands for landmark extraction."""

    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    def process(self, frame: np.ndarray) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Detect hands and return flattened, normalized landmarks for the first hand,
        and handedness label ("Left" or "Right").

        Returns (landmarks, handedness) or (None, None) if no hand detected.
        Landmarks are a list of 63 floats (x,y,z for each of 21 points) normalized
        relative to the wrist (landmark 0). Coordinates are in image pixel space after
        scaling, but x,y are absolute pixel coords; z is relative to wrist depth.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None

        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label  # "Left" / "Right"

        # Convert landmarks to list of (x, y, z)
        h, w, _ = frame.shape
        points = []
        for lm in hand_landmarks.landmark:
            points.append((lm.x * w, lm.y * h, lm.z * w))  # scale z similarly

        # Normalize relative to wrist (landmark 0)
        wrist = np.array(points[0])
        normalized = []
        for p in points:
            normalized.extend((p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]))

        return normalized, handedness

    def draw(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks and connections on the frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_spec, self.drawing_spec
                )
        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()