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
  - Exactly one jutsu is ever "active" (selected) at a time, via
    JutsuSequenceEngine.select(). Only the active tracker receives
    predictions through update() - every other tracker is simply never
    called, so it cannot advance no matter what the active jutsu shares
    with it. (Earlier versions of this engine fed every tracker every
    prediction in parallel; that caused jutsu sharing hand signs, e.g.
    Chidori and Fireball, to cross-advance while performing a different
    one - selection replaces that design.)
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
    """Owns one JutsuTracker per jutsu in JUTSU_SEQUENCES. Only ONE jutsu can
    be "active" (selected) at a time - see select(). update() only ever
    feeds the active tracker; every other tracker is simply never called,
    so it cannot advance no matter what predictions come in. This is what
    stops jutsu that share hand signs (e.g. Chidori and Fireball both use
    several of the same signs) from cross-advancing while the user is
    performing a different one.

    Selecting a jutsu does NOT reset its progress - switching to Chidori,
    performing 2 of its 4 steps, switching away, then switching back to
    Chidori later resumes at step 2, it doesn't restart at 0. This was a
    deliberate choice to keep the change minimal (no new reset-on-select
    behavior invented) rather than a strong design opinion either way - call
    engine.reset(jutsu_name) explicitly (already existed) if you want a
    "start over" action instead.
    """

    def __init__(self, sequences: Optional[Dict[str, List[str]]] = None,
                 cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
                 active_jutsu: Optional[str] = None):
        sequences = sequences if sequences is not None else JUTSU_SEQUENCES
        self.trackers: Dict[str, JutsuTracker] = {
            name: JutsuTracker(name, steps, cooldown_seconds) for name, steps in sequences.items()
        }
        self.active_jutsu: Optional[str] = None
        if active_jutsu is not None:
            self.select(active_jutsu)

    def select(self, jutsu_name: str) -> None:
        """Make `jutsu_name` the only tracker that receives future update()
        calls. Raises KeyError for an unknown name (fail loudly rather than
        silently no-op on a typo)."""
        if jutsu_name not in self.trackers:
            raise KeyError(f"Unknown jutsu: {jutsu_name!r} (known: {list(self.trackers)})")
        self.active_jutsu = jutsu_name

    def deselect(self) -> None:
        """Clear the active jutsu - update() becomes a no-op until select()
        is called again."""
        self.active_jutsu = None

    def update(self, prediction: Optional[str], mouth_o: bool = False,
               now: Optional[float] = None) -> List[str]:
        """Feed one frame's signals to the ACTIVE jutsu's tracker only.
        Every other tracker is untouched and cannot advance. Returns a list
        containing the active jutsu's name if it completed on this call,
        otherwise an empty list (kept as a list, not a bool, for interface
        compatibility with the previous parallel-tracking behavior)."""
        if self.active_jutsu is None:
            return []
        now = time.time() if now is None else now
        tracker = self.trackers[self.active_jutsu]
        if tracker.update(prediction, mouth_o, now=now):
            return [self.active_jutsu]
        return []

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
