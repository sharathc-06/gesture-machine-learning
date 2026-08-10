"""Real-time image classifier for Naruto hand signs.

Loads models/naruto_image_model.pth ONCE and reuses it for every prediction -
never reloads or re-initializes per frame. Classifies the raw camera image
directly (via src/image_preprocessing.py, the SAME preprocessing used by
scripts/train_naruto_image_classifier.py) rather than depending on MediaPipe
landmarks, per the project's architecture decision.
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NarutoImageClassifier:
    """Loads a trained MobileNetV3-based checkpoint once and classifies BGR
    webcam frames directly. Returns ("unknown", confidence) rather than a
    class name whenever the top prediction's confidence is below
    confidence_threshold, so a hesitant/ambiguous frame never masquerades as
    a confident recognition."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.75) -> None:
        import torch
        import torchvision.models as models
        import torch.nn as nn

        checkpoint = torch.load(model_path, map_location="cpu")
        self.classes = checkpoint["classes"]
        self.model_name = checkpoint.get("model_name", "mobilenet_v3_small")
        self.input_size = checkpoint.get("input_size", 224)
        self.confidence_threshold = confidence_threshold

        if self.model_name != "mobilenet_v3_small":
            raise ValueError(
                f"Checkpoint was trained with model_name={self.model_name!r}, but this "
                f"classifier only knows how to reconstruct 'mobilenet_v3_small'. If you "
                f"trained a different architecture, extend this class's model-building code."
            )

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(self.classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self._torch = torch
        self.model = model
        logger.info(
            f"Loaded Naruto image model from {model_path} "
            f"({self.model_name}, {len(self.classes)} classes, input_size={self.input_size})"
        )

    def predict(self, frame_bgr: np.ndarray) -> Tuple[str, float]:
        """Classify one BGR frame. Returns (label, confidence). label is
        "unknown" if the top confidence is below self.confidence_threshold."""
        from src.image_preprocessing import preprocess_for_model

        tensor = preprocess_for_model(frame_bgr, self.input_size).unsqueeze(0)
        with self._torch.no_grad():
            logits = self.model(tensor)
            probs = self._torch.nn.functional.softmax(logits, dim=1)[0]
            idx = int(probs.argmax())
            confidence = float(probs[idx])

        if confidence < self.confidence_threshold:
            return "unknown", confidence
        return self.classes[idx], confidence
