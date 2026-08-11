"""Image dataset collector for the Naruto image-classification pipeline.

Separate from, and does not touch, scripts/collect_naruto_data.py (the
landmark-based collector) or scripts/collect_data.py (generic gestures).
This one saves raw camera frames - no MediaPipe landmark extraction is used
for the saved training data, per the architecture decision to classify the
image directly rather than depend on landmarks.

WORKFLOW (countdown + burst capture)
--------------------------------------
Two-handed signs can't practically be captured by "press a number key while
also holding a two-handed pose" - both hands are occupied. Instead:

  1. Press the number key for the desired sign.
  2. A ~2s "Get ready" countdown gives you time to form the complete
     two-handed sign with both hands free to move into position.
  3. When the countdown reaches 0, a burst of 10 frames is captured
     automatically (no further keypress needed), with a small delay between
     frames so they're not all byte-identical.
  4. A short cooldown follows so the same keypress can't be misread as
     wanting a second burst.

While in the countdown/capturing/cooldown state, number keys are ignored
(only 'q' still works, to quit at any time) - this keeps one burst's flow
uninterrupted and unambiguous rather than trying to interleave a second
request mid-burst.

SESSION-AWARE FILENAMES
------------------------
Every run of this script gets one session id (a timestamp). Every saved
image is named:

    data/naruto_images/<class>/<class>_<session_id>_<seq>.jpg

The session id lets scripts/train_naruto_image_classifier.py split the
dataset by *session* rather than by individual image (see that script's
docstring) - splitting individual frames randomly would leak near-duplicate
frames from the same sitting between train and validation/test, producing
misleadingly high accuracy that doesn't reflect live webcam performance.
This is the same failure mode flagged during the earlier landmark-model
evaluation; collecting across multiple separate sessions (re-running this
script on a different day/lighting/position) is what actually fixes it - the
filename scheme just makes that possible for the training script to detect
and use later. This logic is unchanged from before this burst-capture
rework: sequence numbers still seed themselves from files already on disk
for the current session id, so a same-second restart still can't collide.

MediaPipe is used here for two things, both non-authoritative for what gets
saved as training data:
  - an optional on-screen landmark overlay to help you frame two-hand signs
  - a hand-count check during each burst frame, used only to WARN you if
    fewer hands than the sign requires were visible (see HAND_REQUIREMENTS) -
    it never blocks or discards a capture. If MediaPipe fails to initialize
    for any reason, collection continues without either feature.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.camera import Camera
from src.config import Config, setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("collect_naruto_images")

NARUTO_KEYS = {
    ord('1'): "tiger",
    ord('2'): "ox",
    ord('3'): "hare",
    ord('4'): "monkey",
    ord('5'): "horse",
    ord('6'): "snake",
    ord('7'): "ram",
    ord('8'): "boar",
    ord('9'): "bowl",
}

CLASS_NAMES = list(NARUTO_KEYS.values())

# All 8 real Naruto seals are two-handed; "bowl" is our custom one-hand pose.
# Same mapping used by scripts/collect_naruto_data.py, for the same reason.
HAND_REQUIREMENTS = {
    "tiger": 2, "ox": 2, "hare": 2, "monkey": 2,
    "horse": 2, "snake": 2, "ram": 2, "boar": 2,
    "bowl": 1,
}

COUNTDOWN_SECONDS = 2.0
BURST_SIZE = 10
BURST_INTERVAL_SECONDS = 0.15   # delay between frames within one burst
BURST_COOLDOWN_SECONDS = 1.0    # delay after a burst before another can start
STATUS_DISPLAY_SECONDS = 2.0
BURST_RESULT_DISPLAY_SECONDS = 3.0  # burst-complete warnings stay up longer


def _try_init_optional_detector():
    """Best-effort MediaPipe hands for the overlay AND the hand-count warning
    check. Returns None (and logs why) if it can't be constructed, rather
    than crashing collection over features that are explicitly non-blocking -
    this project's MediaPipe legacy `solutions` API has been flaky across
    environments during development."""
    try:
        from src.detector import HandDetector
        return HandDetector(max_num_hands=2, model_complexity=0,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:  # noqa: BLE001 - genuinely best-effort, any failure is fine
        logger.warning(f"Optional MediaPipe overlay/hand-check unavailable ({e}); continuing without it.")
        return None


def count_existing_images(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(1 for _ in class_dir.glob("*.jpg"))


def draw_hud(frame, *, last_label, counts, total, status,
             countdown_label=None, countdown_remaining=None, capture_progress=None, capture_label=None):
    y = 25
    line_h = 22

    def put(text, color=(255, 255, 255), scale=0.55, thickness=1):
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_h

    put("NARUTO IMAGE COLLECTION", (0, 255, 255), 0.65)

    if countdown_remaining is not None:
        put(f"GET READY: {countdown_label.upper()}", (0, 200, 255), 0.7, 2)
        put(f"Get ready: {countdown_remaining:.1f}s", (0, 200, 255), 0.7, 2)
    elif capture_progress is not None:
        i, n = capture_progress
        put(f"CAPTURING: {i}/{n}  ({capture_label.upper()})",
            (0, 0, 255), 0.75, 2)
    else:
        if last_label:
            put(f"Current sign: {last_label.upper()}   Images: {counts[last_label]}")
        else:
            put("Current sign: -")

    put(f"Total images: {total}")
    if status:
        put(status, (0, 165, 255))

    y += 6
    put("Press: 1=tiger 2=ox 3=hare 4=monkey 5=horse", (200, 200, 200), 0.5)
    put("       6=snake 7=ram 8=boar 9=bowl   q=quit", (200, 200, 200), 0.5)


def main():
    parser = argparse.ArgumentParser(description="Collect Naruto hand-sign images")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Root directory for the image dataset (defaults to config.yaml naruto.image_dataset.dir)")
    parser.add_argument("--no-overlay", action="store_true",
                         help="Disable the optional MediaPipe hand-landmark overlay and hand-count check")
    args = parser.parse_args()

    config = Config()
    setup_logging(config)
    root = Path(args.output_dir) if args.output_dir else config.naruto_image_dir
    for cls in CLASS_NAMES:
        (root / cls).mkdir(parents=True, exist_ok=True)

    counts = {cls: count_existing_images(root / cls) for cls in CLASS_NAMES}
    total = sum(counts.values())
    logger.info(f"Saving into {root} - existing counts: {counts}")

    session_id = time.strftime("%Y%m%d_%H%M%S")
    # Seed each class's sequence counter from any files that ALREADY exist for
    # this exact session id, rather than assuming it starts at 0 (protects
    # against a same-second restart reusing/overwriting filenames).
    seq_in_session = {
        cls: sum(1 for _ in (root / cls).glob(f"{cls}_{session_id}_*.jpg"))
        for cls in CLASS_NAMES
    }

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()
    detector = None if args.no_overlay else _try_init_optional_detector()
    if detector is None:
        logger.warning("Hand-count warnings are DISABLED for this run (no MediaPipe overlay available).")

    print("NARUTO IMAGE COLLECTION")
    print(f"Session id: {session_id}")
    print("Press 1-9 to select a sign: a ~2s countdown starts, then a 10-frame burst is")
    print("captured automatically - get both hands into position during the countdown.")
    print("Press 'q' to quit at any time. Progress resumes automatically next run.")
    print("Vary hand position/distance/rotation/lighting between bursts - see script docstring.")

    # ---- capture state machine: "idle" -> "countdown" -> "capturing" -> "cooldown" -> "idle" ----
    state = "idle"
    countdown_label = None
    countdown_end_time = 0.0
    capture_label = None
    capture_index = 0
    next_capture_time = 0.0
    insufficient_hand_frames = 0
    cooldown_end_time = 0.0

    last_label = None
    status = ""
    status_until = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # NOTE: the flipped frame is what gets saved AND what will be fed to
            # the classifier at inference time - keeping this orientation
            # consistent between training data and live inference matters for
            # signs where hand geometry is asymmetric. Unchanged from before.
            display_frame = frame.copy()

            results = None
            if detector is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.hands.process(rgb)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            detector.mp_drawing.draw_landmarks(
                                display_frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS
                            )
                except Exception as e:  # noqa: BLE001 - overlay/check is best-effort only
                    logger.debug(f"Overlay/hand-check failed, continuing without it this frame: {e}")
                    results = None

            now = time.time()
            if now > status_until:
                status = ""

            countdown_remaining = None
            capture_progress = None

            if state == "countdown":
                countdown_remaining = countdown_end_time - now
                if countdown_remaining <= 0:
                    state = "capturing"
                    capture_label = countdown_label
                    capture_index = 0
                    insufficient_hand_frames = 0
                    next_capture_time = now
                    countdown_remaining = None

            if state == "capturing":
                capture_progress = (capture_index, BURST_SIZE)
                if now >= next_capture_time and capture_index < BURST_SIZE:
                    required = HAND_REQUIREMENTS[capture_label]
                    if detector is not None:
                        num_hands = len(results.multi_hand_landmarks) if (results and results.multi_hand_landmarks) else 0
                        if num_hands < required:
                            insufficient_hand_frames += 1
                    # Saved regardless of the hand-count check - this is a
                    # warning signal for you to consider redoing the sign,
                    # not an automatic quality gate that silently drops data.
                    seq_in_session[capture_label] += 1
                    filename = f"{capture_label}_{session_id}_{seq_in_session[capture_label]:04d}.jpg"
                    out_path = root / capture_label / filename
                    ok = cv2.imwrite(str(out_path), frame)
                    if ok:
                        counts[capture_label] += 1
                        total += 1
                        capture_index += 1
                        next_capture_time = now + BURST_INTERVAL_SECONDS
                        logger.info(f"Saved {out_path.name} ({capture_index}/{BURST_SIZE}) | "
                                    f"{capture_label} total: {counts[capture_label]} | overall: {total}")
                    else:
                        logger.warning(f"cv2.imwrite failed for {out_path}, retrying next frame")
                    capture_progress = (capture_index, BURST_SIZE)

                if capture_index >= BURST_SIZE:
                    if detector is None:
                        status = f"Captured {BURST_SIZE}/{BURST_SIZE} for '{capture_label}' (hand-count check unavailable)"
                    elif insufficient_hand_frames > 0:
                        status = (f"WARNING: {insufficient_hand_frames}/{BURST_SIZE} frames had fewer than "
                                  f"{HAND_REQUIREMENTS[capture_label]} hand(s) for '{capture_label}' - consider redoing")
                        logger.warning(status)
                    else:
                        status = f"Captured {BURST_SIZE}/{BURST_SIZE} for '{capture_label}' - hands OK every frame"
                    status_until = now + BURST_RESULT_DISPLAY_SECONDS
                    last_label = capture_label
                    capture_label = None
                    state = "cooldown"
                    cooldown_end_time = now + BURST_COOLDOWN_SECONDS
                    capture_progress = None

            if state == "cooldown" and now >= cooldown_end_time:
                state = "idle"

            draw_hud(
                display_frame, last_label=last_label, counts=counts, total=total, status=status,
                countdown_label=countdown_label, countdown_remaining=countdown_remaining,
                capture_progress=capture_progress, capture_label=capture_label,
            )
            cv2.imshow("Naruto Image Collection", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key in NARUTO_KEYS and state == "idle":
                countdown_label = NARUTO_KEYS[key]
                countdown_end_time = now + COUNTDOWN_SECONDS
                state = "countdown"
                status = ""
    finally:
        camera.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()

    logger.info(f"Session {session_id} done. Final counts: {counts}")


if __name__ == "__main__":
    main()
