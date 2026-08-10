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

    def _extract_hand_vector(self, hand_landmarks, frame_shape) -> List[float]:
        """
        Convert one MediaPipe hand_landmarks object into a flattened, wrist-relative
        63-float vector (x,y,z for each of the 21 points). This is the same
        normalization process() has always used, factored out so it can be reused
        for two-hand extraction without duplicating (and risking drift from) the logic.
        """
        h, w, _ = frame_shape
        points = []
        for lm in hand_landmarks.landmark:
            points.append((lm.x * w, lm.y * h, lm.z * w))  # scale z similarly

        # Normalize relative to wrist (landmark 0)
        wrist = np.array(points[0])
        normalized = []
        for p in points:
            normalized.extend((p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]))
        return normalized

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
        normalized = self._extract_hand_vector(hand_landmarks, frame.shape)
        return normalized, handedness

    def process_two_hands(self, frame: np.ndarray) -> dict:
        """
        Detect up to two hands and return each one's normalized 63-float vector in a
        dict keyed by MediaPipe-reported handedness, plus a ready-to-save 126-float
        combined vector.

        NOTE on handedness: MediaPipe's Left/Right label assumes the input image is
        already mirrored (selfie-view) - which matches how this project calls
        detector methods (app.py and the collectors flip the frame with cv2.flip(...,1)
        before processing), so "Left"/"Right" here means the user's actual left/right
        hand, not the raw camera image's left/right.

        Returns a dict:
            {
                "left": [63 floats] or None,   # None if the left hand wasn't detected
                "right": [63 floats] or None,  # None if the right hand wasn't detected
                "num_hands": int,               # 0, 1, or 2
                "combined": [126 floats],        # left (or 63 zeros) + right (or 63 zeros)
            }

        A missing hand is represented as 63 zeros in "combined" rather than being
        omitted, so every sample has a fixed-length, deterministic 126-feature shape
        regardless of how many hands were actually visible.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hands_out = {"left": None, "right": None}
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness_info.classification[0].label  # "Left" / "Right"
                vector = self._extract_hand_vector(hand_landmarks, frame.shape)
                # If MediaPipe ever reports the same side twice (shouldn't normally
                # happen with max_num_hands=2), the first detection for that side wins
                # rather than being silently overwritten by the second.
                if hands_out[label.lower()] is None:
                    hands_out[label.lower()] = vector

        num_hands = sum(1 for v in hands_out.values() if v is not None)
        zeros = [0.0] * 63
        combined = (hands_out["left"] or zeros) + (hands_out["right"] or zeros)

        return {
            "left": hands_out["left"],
            "right": hands_out["right"],
            "num_hands": num_hands,
            "combined": combined,
        }

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