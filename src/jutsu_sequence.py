"""Jutsu sequence engine: tracks progress through the three Naruto jutsu
hand-sign sequences, independently of and on top of the existing
NarutoImageClassifier real-time predictions.

Responsibilities are deliberately kept separate from the classifier:
  - NarutoImageClassifier (src/naruto_image_classifier.py) only ever answers
    "what sign is visible right now" - it knows nothing about jutsu, steps,
    or sequences.
  - JutsuSequenceEngine (this module) only ever answers "does this
    prediction match the next step of some jutsu I'm tracking" - it has no
    idea how the prediction was produced, and never touches a camera frame
    or a model.
  - Effects/what-happens-on-completion is the caller's job (see
    scripts/run_naruto_image_classifier.py for the current display-only
    integration) - this module just reports completions, it doesn't play
    sounds or animations itself.

BEHAVIOR (see JutsuTracker for the actual step logic):
  - A single matching prediction/frame is enough to accept a step - there is
    deliberately NO stable-frame requirement here (unlike
    src/utils.py:PredictionSmoother, which is a different, unrelated
    stability mechanism used for the older landmark-based gesture system).
  - Any non-matching prediction is silently ignored - progress never resets
    or goes backwards because of a wrong/irrelevant prediction.
  - Each of the three jutsu is tracked independently and in parallel: the
    same prediction stream is fed to all three trackers every frame, and
    each one advances (or not) purely based on its own current expected
    step. It's entirely possible, and fine, for two trackers to both
    advance on the same prediction (e.g. "tiger" is step 1 of Rasengan AND
    step 2 of Fireball, if Fireball's step 1 "horse" was already accepted).
  - After a jutsu completes, that tracker resets to step 0 and enters a
    cooldown window during which it ignores all predictions - this is what
    stops the same completing gesture from being read again next frame and
    immediately re-triggering.
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Naruto hand signs from the image classifier; "mouth_o" is NOT a classifier
# output - it's a separate boolean signal (see src/mouth_detector.py) fed
# into JutsuSequenceEngine.update() alongside the classifier's prediction.
JUTSU_SEQUENCES: Dict[str, List[str]] = {
    "rasengan": ["tiger", "bowl"],
    "chidori": ["ox", "hare", "monkey", "bowl"],
    "fireball": ["horse", "tiger", "snake", "ram", "monkey", "boar", "horse", "tiger", "mouth_o"],
}

DEFAULT_COOLDOWN_SECONDS = 3.0


@dataclass
class StepProgress:
    """Snapshot of one jutsu tracker's state, for display or logic use."""
    name: str
    steps: List[str]
    accepted: List[bool]           # accepted[i] True if steps[i] has been matched this run
    current_step: Optional[str]    # next required step, or None if complete/on cooldown
    just_completed: bool           # True only on the exact update() call that finished the sequence
    on_cooldown: bool
    cooldown_remaining: float      # seconds left, 0.0 if not on cooldown


class JutsuTracker:
    """Tracks progress through ONE jutsu's step sequence. See module
    docstring for the accept/ignore/never-reset/cooldown behavior."""

    def __init__(self, name: str, steps: List[str], cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS):
        self.name = name
        self.steps = list(steps)
        self.cooldown_seconds = cooldown_seconds
        self.current_index = 0
        self.cooldown_until = 0.0

    def update(self, prediction: Optional[str], mouth_o: bool, now: Optional[float] = None) -> bool:
        """Feed one frame's signals in. Returns True iff this call completed
        the sequence (i.e. the final step was just accepted)."""
        now = time.time() if now is None else now
        if now < self.cooldown_until:
            return False  # ignore everything while on cooldown
        if self.current_index >= len(self.steps):
            # Shouldn't normally happen (completion resets to 0 immediately),
            # but never index out of range if it somehow does.
            return False

        expected = self.steps[self.current_index]
        matched = mouth_o if expected == "mouth_o" else (prediction == expected)
        if not matched:
            return False  # wrong/irrelevant prediction - ignored, no reset (requirement 3/4)

        self.current_index += 1  # step accepted (requirement 1/2/5)
        if self.current_index >= len(self.steps):
            logger.info(f"Jutsu completed: {self.name}")
            self.current_index = 0
            self.cooldown_until = now + self.cooldown_seconds
            return True
        return False

    def reset(self, clear_cooldown: bool = True, now: Optional[float] = None) -> None:
        """External/manual reset. clear_cooldown=False can be used to reset
        progress without lifting an active cooldown, if ever needed."""
        self.current_index = 0
        if clear_cooldown:
            self.cooldown_until = 0.0

    def get_progress(self, now: Optional[float] = None) -> StepProgress:
        now = time.time() if now is None else now
        on_cooldown = now < self.cooldown_until
        accepted = [i < self.current_index for i in range(len(self.steps))]
        current_step = None if on_cooldown or self.current_index >= len(self.steps) else self.steps[self.current_index]
        return StepProgress(
            name=self.name,
            steps=self.steps,
            accepted=accepted,
            current_step=current_step,
            just_completed=False,  # only ever True in the direct return value of update()
            on_cooldown=on_cooldown,
            cooldown_remaining=max(0.0, self.cooldown_until - now),
        )


class JutsuSequenceEngine:
    """Owns one JutsuTracker per jutsu in JUTSU_SEQUENCES and feeds every
    update() call to all of them independently. See module docstring."""

    def __init__(self, sequences: Optional[Dict[str, List[str]]] = None,
                 cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS):
        sequences = sequences if sequences is not None else JUTSU_SEQUENCES
        self.trackers: Dict[str, JutsuTracker] = {
            name: JutsuTracker(name, steps, cooldown_seconds) for name, steps in sequences.items()
        }

    def update(self, prediction: Optional[str], mouth_o: bool = False,
               now: Optional[float] = None) -> List[str]:
        """Feed one frame's signals to every tracked jutsu. Returns the list
        of jutsu names that completed on THIS call (usually empty, sometimes
        one name, in principle could be more than one at once)."""
        now = time.time() if now is None else now
        completed = []
        for name, tracker in self.trackers.items():
            if tracker.update(prediction, mouth_o, now=now):
                completed.append(name)
        return completed

    def reset(self, jutsu_name: Optional[str] = None, clear_cooldown: bool = True) -> None:
        """Reset one jutsu's progress, or all of them if jutsu_name is None."""
        if jutsu_name is None:
            for tracker in self.trackers.values():
                tracker.reset(clear_cooldown=clear_cooldown)
        else:
            self.trackers[jutsu_name].reset(clear_cooldown=clear_cooldown)

    def get_progress(self, jutsu_name: Optional[str] = None, now: Optional[float] = None):
        now = time.time() if now is None else now
        if jutsu_name is not None:
            return self.trackers[jutsu_name].get_progress(now=now)
        return {name: tracker.get_progress(now=now) for name, tracker in self.trackers.items()}

    def get_current_step(self, jutsu_name: str) -> Optional[str]:
        return self.trackers[jutsu_name].get_progress().current_step
