"""Data collection script for the Naruto two-hand sign dataset.

This is a SEPARATE collector from scripts/collect_data.py (which continues to
collect the original single-hand, 63-feature, generic-gesture dataset into
data/landmarks.csv unchanged). This script collects a new, independent
two-hand, 126-feature dataset into data/naruto_landmarks.csv.

TWO-HAND REPRESENTATION
------------------------
Every sample is 126 floats: the left hand's 63-float wrist-relative landmark
vector (see src/detector.py, unchanged normalization) followed by the right
hand's 63-float vector:

    left_x0,left_y0,left_z0, ..., left_x20,left_y20,left_z20,
    right_x0,right_y0,right_z0, ..., right_x20,right_y20,right_z20

Handedness is taken from MediaPipe's own classification (via
HandDetector.process_two_hands), not from detection order, so a single
visible hand is always written into the correct left/right slot rather than
always being treated as "the first hand". If a hand is not detected, its 63
slots are filled with zeros - this keeps every row a fixed 126-feature shape
regardless of how many hands were visible, which is required both for a
consistent CSV shape and because "zero vector" is an unambiguous, easy-to-learn
stand-in for "this hand is absent" (as opposed to e.g. duplicating the other
hand's landmarks or omitting the columns).

HAND REQUIREMENTS PER SIGN
---------------------------
All 8 canonical Naruto seals shown in the reference image are two-handed
poses (both hands interlocked/joined in some way). "bowl" is a custom pose
for this game, not part of the reference image; it's treated as a one-hand
cupped/bowl shape (the natural reading of "bowl hand" as a Rasengan-style
held-out palm) - flag this assumption to confirm/override later if that's not
the intended pose. A sample is only recorded if the sign's required hand
count is currently satisfied; otherwise nothing is written and the on-screen
status explains why (e.g. "Need both hands").
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.camera import Camera
from src.detector import HandDetector
from src.config import Config, setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("collect_naruto_data")

# Key -> label. Matches the 9 classes from PHASE 4.
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

# How many hands must be visible to record a sample of this sign.
# All 8 real Naruto seals are two-handed; "bowl" is our custom one-hand pose.
HAND_REQUIREMENTS = {
    "tiger": 2,
    "ox": 2,
    "hare": 2,
    "monkey": 2,
    "horse": 2,
    "snake": 2,
    "ram": 2,
    "boar": 2,
    "bowl": 1,
}

CSV_HEADER = (
    ["label"]
    + [f"left_{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
    + [f"right_{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
)

# Minimum time between two recorded samples (any label), so holding a key
# down or mashing it can't dump dozens of near-identical consecutive frames
# into the dataset. The person is still expected to physically move/re-form
# the sign between captures for real variation (see module docstring).
CAPTURE_COOLDOWN_SECONDS = 0.25

STATUS_DISPLAY_SECONDS = 2.0


def load_existing_counts(csv_path: Path) -> dict:
    """Load per-label sample counts from an existing dataset, if any, so a
    restarted session continues counting instead of losing track of progress.
    Does not read the file into memory beyond counting - fine at this scale
    (thousands of rows), and avoids adding a pandas dependency to this script."""
    counts = {label: 0 for label in NARUTO_KEYS.values()}
    if not csv_path.exists():
        return counts
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label")
            if label in counts:
                counts[label] += 1
    return counts


def draw_two_hand_landmarks(frame, results, detector: HandDetector) -> None:
    """Draw landmarks with distinct colors per hand (left=green, right=red) so
    it's visually clear which hand is which while forming two-handed seals."""
    if not results.multi_hand_landmarks:
        return
    left_spec = detector.mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=2)
    right_spec = detector.mp_drawing.DrawingSpec(color=(0, 0, 220), thickness=2, circle_radius=2)
    for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness_info.classification[0].label
        spec = left_spec if label == "Left" else right_spec
        detector.mp_drawing.draw_landmarks(
            frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS, spec, spec
        )


def draw_hud(frame, *, last_label: str, counts: dict, target: int, total: int,
             num_hands: int, left_present: bool, right_present: bool, status: str) -> None:
    y = 25
    line_h = 22

    def put(text, color=(255, 255, 255), scale=0.55):
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_h

    put("NARUTO DATA COLLECTION", (0, 255, 255), 0.65)
    put(f"Last sign: {last_label or '-'}   Samples: {counts.get(last_label, 0)}/{target}"
        if last_label else f"Samples so far: {total}")
    put(f"Total samples: {total}")
    hand_desc = f"{num_hands} ({'L' if left_present else '-'}{'R' if right_present else '-'})"
    put(f"Hands detected: {hand_desc}")
    if status:
        put(status, (0, 165, 255))

    y += 6
    put("Press: 1=tiger 2=ox 3=hare 4=monkey 5=horse", (200, 200, 200), 0.5)
    put("       6=snake 7=ram 8=boar 9=bowl   q=quit", (200, 200, 200), 0.5)


def main():
    parser = argparse.ArgumentParser(description="Collect two-hand Naruto sign landmarks")
    parser.add_argument("--output", type=str, default=None,
                         help="Output CSV file (defaults to config.yaml naruto.dataset_path)")
    args = parser.parse_args()

    config = Config()
    setup_logging(config)
    out_path = Path(args.output) if args.output else config.naruto_dataset_path
    target = config.naruto_target_samples_per_class
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = load_existing_counts(out_path)
    total = sum(counts.values())
    file_exists = out_path.exists()
    logger.info(f"Resuming into {out_path} - {total} existing samples: {counts}")

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()
    # Two hands are required for this dataset regardless of the main app's
    # single-hand config, so max_num_hands is fixed at 2 here rather than
    # reusing config.mediapipe["max_num_hands"] (which stays 1 for app.py).
    detector = HandDetector(
        max_num_hands=2,
        model_complexity=config.mediapipe["model_complexity"],
        min_detection_confidence=config.mediapipe["min_detection_confidence"],
        min_tracking_confidence=config.mediapipe["min_tracking_confidence"],
    )

    print("NARUTO DATA COLLECTION")
    print("Press 1-9 to record a sample for the corresponding sign (see on-screen legend).")
    print("Two-handed signs require both hands visible; 'bowl' requires one hand.")
    print("Press 'q' to quit. Progress is saved incrementally and resumes next run.")

    last_label = None
    last_capture_time = 0.0
    status = ""
    status_until = 0.0

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)

        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Two-hand extraction for saving, raw results (from the same frame)
            # for drawing. Re-running detector.hands.process for drawing mirrors
            # the existing double-process pattern already used in app.py /
            # collect_data.py in this codebase.
            hands_data = detector.process_two_hands(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.hands.process(rgb)
            draw_two_hand_landmarks(frame, results, detector)

            now = time.time()
            if now > status_until:
                status = ""

            draw_hud(
                frame,
                last_label=last_label,
                counts=counts,
                target=target,
                total=total,
                num_hands=hands_data["num_hands"],
                left_present=hands_data["left"] is not None,
                right_present=hands_data["right"] is not None,
                status=status,
            )

            cv2.imshow("Naruto Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key in NARUTO_KEYS:
                label = NARUTO_KEYS[key]
                required = HAND_REQUIREMENTS[label]
                if now - last_capture_time < CAPTURE_COOLDOWN_SECONDS:
                    status, status_until = "Cooldown, wait a moment", now + STATUS_DISPLAY_SECONDS
                elif hands_data["num_hands"] < required:
                    needed = "both hands" if required == 2 else "one hand"
                    status = f"Need {needed} for '{label}' (have {hands_data['num_hands']})"
                    status_until = now + STATUS_DISPLAY_SECONDS
                elif len(hands_data["combined"]) != 126:
                    # Defensive: should never happen given process_two_hands' contract,
                    # but never write a malformed row if it somehow did.
                    status, status_until = "Invalid landmark data, sample skipped", now + STATUS_DISPLAY_SECONDS
                    logger.warning(f"Skipped malformed sample for '{label}': "
                                    f"got {len(hands_data['combined'])} features, expected 126")
                else:
                    writer.writerow([label] + hands_data["combined"])
                    f.flush()
                    counts[label] += 1
                    total += 1
                    last_label = label
                    last_capture_time = now
                    status, status_until = f"Recorded '{label}' ({counts[label]}/{target})", now + STATUS_DISPLAY_SECONDS
                    logger.info(f"Recorded: {label} | total for label: {counts[label]} | total overall: {total}")

    camera.release()
    detector.close()
    cv2.destroyAllWindows()
    logger.info(f"Data saved to {out_path}. Final counts: {counts}")


if __name__ == "__main__":
    main()
