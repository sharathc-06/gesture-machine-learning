"""Webcam capture using OpenCV."""

import cv2
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Camera:
    """Manages webcam capture and provides frames."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """Open the default camera and set properties."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info("Camera started (%dx%d @ %d fps)", self.width, self.height, self.fps)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Capture a frame. Returns (success, frame)."""
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() first.")
        return self.cap.read()

    def release(self) -> None:
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
        cv2.destroyAllWindows()