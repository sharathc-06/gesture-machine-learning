"""Load configuration from config.yaml."""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Application configuration loaded from a YAML file."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            self._data: Dict[str, Any] = yaml.safe_load(f)

        # Convenience attributes
        self.camera = self._data["camera"]
        self.mediapipe = self._data["mediapipe"]
        self.classifier = self._data["classifier"]
        self.debounce = self._data["debounce"]
        self.smoothing = self._data["smoothing"]
        self.assets = self._data["assets"]
        self.gesture_mappings = self._data["gesture_mappings"]
        self.logging_config = self._data["logging"]
        self.naruto = self._data.get("naruto", {})

    @property
    def audio_dir(self) -> Path:
        return Path(self.assets["audio_dir"])

    @property
    def video_dir(self) -> Path:
        return Path(self.assets["video_dir"])

    @property
    def model_path(self) -> Path:
        return Path(self.classifier["model_path"])

    @property
    def label_map(self) -> list:
        return self.classifier["label_map"]

    @property
    def naruto_dataset_path(self) -> Path:
        return Path(self.naruto.get("dataset_path", "data/naruto_landmarks.csv"))

    @property
    def naruto_target_samples_per_class(self) -> int:
        return int(self.naruto.get("target_samples_per_class", 500))


def setup_logging(config: Config) -> None:
    """Configure root logger based on config."""
    logging.basicConfig(
        level=config.logging_config.get("level", "INFO"),
        format=config.logging_config.get("format", "%(message)s"),
    )