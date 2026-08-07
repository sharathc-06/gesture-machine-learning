"""Asynchronous playback of audio and video files."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pygame

logger = logging.getLogger(__name__)


class MediaPlayer:
    """Plays audio (using pygame) and video (using system default player) without blocking."""

    def __init__(self, audio_dir: Path, video_dir: Path) -> None:
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        # Initialise pygame mixer for audio
        pygame.mixer.init()
        logger.info("MediaPlayer initialised")

    def play_audio(self, filename: str) -> None:
        """Play an audio file asynchronously."""
        path = self.audio_dir / filename
        if not path.exists():
            logger.error(f"Audio file not found: {path}")
            return
        try:
            sound = pygame.mixer.Sound(str(path))
            sound.play()
            logger.info(f"Playing audio: {path}")
        except Exception as e:
            logger.error(f"Failed to play audio {path}: {e}")

    def play_video(self, filename: str) -> None:
        """Open a video file with the system's default player (non-blocking)."""
        path = self.video_dir / filename
        if not path.exists():
            logger.error(f"Video file not found: {path}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            logger.info(f"Opening video: {path}")
        except Exception as e:
            logger.error(f"Failed to open video {path}: {e}")

    def trigger_gesture(self, gesture: str, config: dict) -> None:
        """Look up gesture mapping and play associated media."""
        mapping = config.get(gesture)
        if mapping is None:
            logger.warning(f"No mapping for gesture '{gesture}'")
            return
        audio_file = mapping.get("audio")
        video_file = mapping.get("video")
        if audio_file:
            self.play_audio(audio_file)
        if video_file:
            self.play_video(video_file)

    def cleanup(self) -> None:
        pygame.mixer.quit()