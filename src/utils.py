"""Utility functions: FPS counter, drawing helpers, prediction smoother."""

import cv2
import time
import logging
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FPSCounter:
    """Tracks and displays frames per second."""

    def __init__(self, avg_frames: int = 10) -> None:
        self.avg_frames = avg_frames
        self.frame_times = deque(maxlen=avg_frames)
        self.last_time = time.time()

    def update(self) -> None:
        now = time.time()
        self.frame_times.append(now - self.last_time)
        self.last_time = now

    @property
    def fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

    def draw(self, frame: cv2.Mat) -> None:
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


class PredictionSmoother:
    """
    Smooths per‑frame gesture predictions and debounces triggers.
    Requires a stable prediction for `stable_frames` consecutive frames,
    then enforces a cooldown before the same gesture can be triggered again.
    """

    def __init__(self, stable_frames: int = 5, cooldown_seconds: float = 2.0) -> None:
        self.stable_frames = stable_frames
        self.cooldown = cooldown_seconds
        self.history: deque = deque(maxlen=stable_frames)
        self.last_trigger_time: dict = {}  # gesture -> timestamp

    def update(self, gesture: str, confidence: float) -> Tuple[bool, str, float]:
        """
        Feed a new prediction.
        Returns (should_trigger, gesture, avg_confidence).
        """
        self.history.append((gesture, confidence))
        if len(self.history) < self.stable_frames:
            return False, gesture, confidence

        # Check if all recent predictions are the same
        gestures = [g for g, _ in self.history]
        if len(set(gestures)) == 1:
            stable_gesture = gestures[0]
            avg_conf = sum(c for _, c in self.history) / len(self.history)
            now = time.time()
            last = self.last_trigger_time.get(stable_gesture, 0)
            if now - last >= self.cooldown:
                self.last_trigger_time[stable_gesture] = now
                return True, stable_gesture, avg_conf
        return False, gesture, confidence


def draw_info(frame: cv2.Mat, gesture: str, confidence: float) -> None:
    """Overlay gesture name and confidence on the frame."""
    text = f"Gesture: {gesture} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)