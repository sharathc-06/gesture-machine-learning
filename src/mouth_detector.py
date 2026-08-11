"""Placeholder mouth-open-O detector for the Fireball jutsu's final step.

STATUS: STUB - NOT IMPLEMENTED.

The project currently has no facial-landmark detection anywhere (MediaPipe is
only ever used for hands in this codebase - see src/detector.py). Rather than
faking mouth-O detection with the hand classifier (explicitly not wanted) or
building a real face-mesh pipeline as a side effect of the sequence-engine
task, this module isolates the interface so a real implementation can drop
in later without touching JutsuSequenceEngine or the real-time loop at all.

TODO (future work): implement real detection, e.g. via MediaPipe Face Mesh -
extract mouth landmarks, compute a mouth aspect ratio (MAR) from
inner-lip landmark distances, and threshold it to distinguish an open "O"
shape from a closed/neutral mouth. Until then this always returns False.
"""
import numpy as np


def detect_mouth_o(frame: np.ndarray) -> bool:
    """Detect whether the person's mouth is forming an "O" shape in `frame`.

    STUB: always returns False. See module docstring for what a real
    implementation should do. Callers should not assume this ever returns
    True on its own - see scripts/run_naruto_image_classifier.py for a
    manual keyboard-triggered stand-in used for demoing/testing the Fireball
    jutsu's final step until a real detector exists.
    """
    return False
