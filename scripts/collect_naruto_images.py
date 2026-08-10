"""Image dataset collector for the Naruto image-classification pipeline.

Separate from, and does not touch, scripts/collect_naruto_data.py (the
landmark-based collector) or scripts/collect_data.py (generic gestures).
This one saves raw camera frames - no MediaPipe landmark extraction is used
for the saved training data, per the architecture decision to classify the
image directly rather than depend on landmarks.

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
and use later.

MediaPipe is used here ONLY as an optional on-screen visual aid to help you
frame two-hand signs while collecting - never to gate or filter what gets
saved, and never as part of the saved data itself. If it fails to initialize
for any reason, collection continues without the overlay.
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

# Minimum time between two saved images, so holding/mashing a key can't dump
# dozens of near-identical consecutive frames into the dataset.
CAPTURE_COOLDOWN_SECONDS = 0.3
STATUS_DISPLAY_SECONDS = 2.0


def _try_init_optional_detector():
    """Best-effort MediaPipe hand overlay for framing help only. Returns None
    (and logs why) if it can't be constructed, rather than crashing collection
    over a purely cosmetic feature - this project's MediaPipe legacy `solutions`
    API has been flaky across environments during development."""
    try:
        from src.detector import HandDetector
        return HandDetector(max_num_hands=2, model_complexity=0,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:  # noqa: BLE001 - genuinely best-effort, any failure is fine
        logger.warning(f"Optional MediaPipe overlay unavailable ({e}); continuing without it.")
        return None


def count_existing_images(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(1 for _ in class_dir.glob("*.jpg"))


def draw_hud(frame, *, last_label, counts, total, status):
    y = 25
    line_h = 22

    def put(text, color=(255, 255, 255), scale=0.55):
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_h

    put("NARUTO IMAGE COLLECTION", (0, 255, 255), 0.65)
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
                         help="Disable the optional MediaPipe hand-landmark overlay")
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
    # this exact session id, rather than assuming it starts at 0. session_id
    # only has second-resolution, so a restart within the same second as a
    # previous run (easy to hit when testing, or just re-running the script
    # quickly) would otherwise reuse filenames and silently overwrite the
    # earlier run's images instead of appending to them.
    seq_in_session = {
        cls: sum(1 for _ in (root / cls).glob(f"{cls}_{session_id}_*.jpg"))
        for cls in CLASS_NAMES
    }

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()
    detector = None if args.no_overlay else _try_init_optional_detector()

    print("NARUTO IMAGE COLLECTION")
    print(f"Session id: {session_id}")
    print("Press 1-9 to capture an image for the corresponding sign.")
    print("Press 'q' to quit. Progress resumes automatically next run.")
    print("Vary hand position/distance/rotation/lighting between captures - see script docstring.")

    last_label = None
    last_capture_time = 0.0
    status = ""
    status_until = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # NOTE: the flipped frame is what gets saved AND what will be fed to
            # the classifier at inference time (see src/naruto_image_classifier.py
            # and scripts/run_naruto_image_classifier.py) - keeping this
            # orientation consistent between training data and live inference
            # matters, since a systematic left/right mismatch would quietly hurt
            # accuracy on signs where hand geometry is asymmetric.
            display_frame = frame.copy()

            if detector is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.hands.process(rgb)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            detector.mp_drawing.draw_landmarks(
                                display_frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS
                            )
                except Exception as e:  # noqa: BLE001 - overlay is best-effort only
                    logger.debug(f"Overlay draw failed, continuing without it this frame: {e}")

            now = time.time()
            if now > status_until:
                status = ""

            draw_hud(display_frame, last_label=last_label, counts=counts, total=total, status=status)
            cv2.imshow("Naruto Image Collection", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key in NARUTO_KEYS:
                label = NARUTO_KEYS[key]
                if now - last_capture_time < CAPTURE_COOLDOWN_SECONDS:
                    status, status_until = "Cooldown, wait a moment", now + STATUS_DISPLAY_SECONDS
                else:
                    seq_in_session[label] += 1
                    filename = f"{label}_{session_id}_{seq_in_session[label]:04d}.jpg"
                    out_path = root / label / filename
                    ok = cv2.imwrite(str(out_path), frame)
                    if not ok:
                        status, status_until = f"Failed to save image for '{label}'", now + STATUS_DISPLAY_SECONDS
                        logger.warning(f"cv2.imwrite failed for {out_path}")
                    else:
                        counts[label] += 1
                        total += 1
                        last_label = label
                        last_capture_time = now
                        status = f"Captured '{label}' ({counts[label]} total)"
                        status_until = now + STATUS_DISPLAY_SECONDS
                        logger.info(f"Saved {out_path.name} | {label} total: {counts[label]} | overall: {total}")
    finally:
        camera.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()

    logger.info(f"Session {session_id} done. Final counts: {counts}")


if __name__ == "__main__":
    main()
