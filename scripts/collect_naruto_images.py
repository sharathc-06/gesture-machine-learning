"""Image dataset collector for the Naruto image-classification pipeline.

Separate from, and does not touch, scripts/collect_naruto_data.py (the
landmark-based collector) or scripts/collect_data.py (generic gestures).

HAND-CROP REWORK
-----------------
This collector now saves a CROPPED hand region (via src/detector.py's
get_hand_bboxes() + src/image_preprocessing.py's union_bbox()/
crop_hand_region()) instead of the full raw camera frame, matching the
architecture decision to reduce background/position shortcut-learning (see
scripts/visual_test_hand_crop.py, which verified this same crop logic
visually before this change was made).

Saves into data/naruto_hand_crops/<class>/ by default - a NEW, separate
directory from the original data/naruto_images/ (which held 1,800 full-frame
images under the old architecture). The old directory is never read or
written by this script and is left completely untouched as a backup.

Because the saved data is now a CROP rather than a full frame, MediaPipe
hand detection is no longer optional/best-effort here the way it used to be
(previously it only drove an on-screen overlay and a non-blocking warning).
It is now REQUIRED - without a detected hand there is nothing valid to crop
- so detector construction failure is a hard error, and any capture attempt
that doesn't find enough hands for the sign being collected is skipped
entirely rather than saved. See the "VALID-CROP-ONLY POLICY" note below.

WORKFLOW (countdown + burst capture) - UNCHANGED
--------------------------------------------------
Two-handed signs can't practically be captured by "press a number key while
also holding a two-handed pose" - both hands are occupied. Instead:

  1. Press the number key for the desired sign.
  2. A ~2s "Get ready" countdown gives you time to form the complete
     two-handed sign with both hands free to move into position.
  3. When the countdown reaches 0, a burst of up to 10 capture ATTEMPTS
     happens automatically (no further keypress needed), with a small delay
     between attempts so they're not all byte-identical.
  4. A short cooldown follows so the same keypress can't be misread as
     wanting a second burst.

While in the countdown/capturing/cooldown state, number keys are ignored
(only 'q' still works, to quit at any time) - unchanged from before.

VALID-CROP-ONLY POLICY (the one real behavior change from the old collector)
------------------------------------------------------------------------------
The old collector's rule was "warn, don't silently accept a bad frame as
if it were fine" - but it still ALWAYS saved, because a full frame is a
valid (if noisy) training sample regardless of hand count.

That reasoning does not carry over to a crop-only dataset: a crop is only
meaningful if the required hand(s) were actually detected to crop around -
there is no "keep it anyway" option once the entire point of a saved sample
is the hand region itself. So for this dataset:

  - 0 hands detected -> always skipped (nothing to crop).
  - fewer hands detected than HAND_REQUIREMENTS[sign] (e.g. 1 hand for a
    2-handed seal) -> skipped. A union crop of a false subset of the hands
    a sign is defined by would be actively wrong, not just noisy.
  - >= the required hand count -> saved, using the union bbox of ALL
    currently-detected hands (not just the required count).

The burst still runs its full ~10-attempt cadence (timing unchanged) so you
still get the same amount of "attempt time" per burst, but a burst may now
end with FEWER than 10 saved images if some attempts didn't have hands
in position yet - this is expected and reported explicitly in the HUD/log
summary (saved vs skipped), unlike the old warning-only message.

SESSION-AWARE FILENAMES - UNCHANGED FORMAT
---------------------------------------------
Every run of this script gets one session id (a timestamp). Every SAVED
image is named:

    data/naruto_hand_crops/<class>/<class>_<session_id>_<seq>.jpg

Identical format to the old collector (still parseable by
scripts/train_naruto_image_classifier.py's session-aware split regex,
unmodified). The only difference is `seq` now increments only on an actual
SAVE, not on every attempt - so sequence numbers stay contiguous with no
gaps for skipped attempts, and a same-second restart still can't collide
(seeded from files already on disk for the current session id, same as
before).
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
from src.detector import HandDetector
from src.image_preprocessing import DEFAULT_HAND_CROP_PADDING, crop_hand_region, union_bbox

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
# Now also doubles as the crop-validity gate - see VALID-CROP-ONLY POLICY above.
HAND_REQUIREMENTS = {
    "tiger": 2, "ox": 2, "hare": 2, "monkey": 2,
    "horse": 2, "snake": 2, "ram": 2, "boar": 2,
    "bowl": 1,
}

# NEW dataset root - deliberately separate from config.naruto_image_dir
# (data/naruto_images), which held the old 1,800 full-frame images and is
# left completely untouched by this script. Not read from config.yaml since
# that key is still legitimately "the old dataset's location" for anything
# else that reads it (e.g. a future re-run of the old collector).
DEFAULT_OUTPUT_DIR = Path("data/naruto_hand_crops")

COUNTDOWN_SECONDS = 2.0
BURST_SIZE = 10
BURST_INTERVAL_SECONDS = 0.15   # delay between attempts within one burst
BURST_COOLDOWN_SECONDS = 1.0    # delay after a burst before another can start
STATUS_DISPLAY_SECONDS = 2.0
BURST_RESULT_DISPLAY_SECONDS = 3.0  # burst-complete summary stays up longer


def count_existing_images(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(1 for _ in class_dir.glob("*.jpg"))


def draw_hud(frame, *, last_label, counts, total, status,
             countdown_label=None, countdown_remaining=None, capture_progress=None, capture_label=None,
             num_hands_now=None):
    y = 25
    line_h = 22

    def put(text, color=(255, 255, 255), scale=0.55, thickness=1):
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_h

    put("NARUTO HAND-CROP COLLECTION", (0, 255, 255), 0.65)

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

    if num_hands_now is not None:
        put(f"Hands detected now: {num_hands_now}", (0, 255, 0) if num_hands_now > 0 else (0, 0, 255), 0.5)

    put(f"Total images: {total}")
    if status:
        put(status, (0, 165, 255))

    y += 6
    put("Press: 1=tiger 2=ox 3=hare 4=monkey 5=horse", (200, 200, 200), 0.5)
    put("       6=snake 7=ram 8=boar 9=bowl   q=quit", (200, 200, 200), 0.5)


def main():
    parser = argparse.ArgumentParser(description="Collect Naruto hand-CROP images (see module docstring)")
    parser.add_argument("--output-dir", type=str, default=None,
                         help=f"Root directory for the cropped-hand image dataset "
                              f"(default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5,
                         help="MediaPipe min_detection_confidence (default 0.5, unchanged from before). "
                              "Lower values let MediaPipe report lower-confidence hand candidates - can help "
                              "on touching/overlapping signs at the cost of more false positives. Use "
                              "scripts/diagnose_two_hand_detection.py to compare values on your own hands "
                              "before changing this default for a real collection run.")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5,
                         help="MediaPipe min_tracking_confidence (default 0.5, unchanged from before). "
                              "See --min-detection-confidence for how to test alternative values first.")
    args = parser.parse_args()

    config = Config()
    setup_logging(config)
    root = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    for cls in CLASS_NAMES:
        (root / cls).mkdir(parents=True, exist_ok=True)

    counts = {cls: count_existing_images(root / cls) for cls in CLASS_NAMES}
    total = sum(counts.values())
    logger.info(f"Saving CROPPED hand images into {root} - existing counts: {counts}")
    logger.info(f"Crop padding: {DEFAULT_HAND_CROP_PADDING} (src/image_preprocessing.DEFAULT_HAND_CROP_PADDING)")

    session_id = time.strftime("%Y%m%d_%H%M%S")
    # Seed each class's sequence counter from any files that ALREADY exist for
    # this exact session id, rather than assuming it starts at 0 (protects
    # against a same-second restart reusing/overwriting filenames). Unchanged
    # from the old collector, just pointed at the new directory.
    seq_in_session = {
        cls: sum(1 for _ in (root / cls).glob(f"{cls}_{session_id}_*.jpg"))
        for cls in CLASS_NAMES
    }

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()

    # Detector is now REQUIRED, not optional - see module docstring. Fail
    # loudly and immediately rather than silently continuing into a run that
    # could only ever produce zero valid crops.
    try:
        detector = HandDetector(
            max_num_hands=2,
            # Full model (1), not Lite (0): this call's output IS the crop
            # source now, not just an overlay - the accuracy/speed tradeoff
            # that favored Lite when this was a nice-to-have no longer
            # applies. See scripts/diagnose_two_hand_detection.py for how to
            # empirically compare complexity/threshold values against your
            # own hardest signs (e.g. Tiger, Ram) before relying on this
            # default for a large collection run.
            model_complexity=1,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    except Exception as e:
        logger.error(f"Could not initialize MediaPipe HandDetector ({e}). "
                      f"This collector cannot produce crops without it - aborting.")
        camera.release()
        sys.exit(1)

    print("NARUTO HAND-CROP COLLECTION")
    print(f"Session id: {session_id}")
    print(f"Saving into: {root}")
    print("Press 1-9 to select a sign: a ~2s countdown starts, then up to a 10-attempt burst")
    print("captures automatically - get both hands into position during the countdown.")
    print("Only attempts with enough detected hands for that sign are SAVED - see script docstring.")
    print("Press 'q' to quit at any time. Progress resumes automatically next run.")
    print("Vary hand position/distance/rotation/lighting between bursts - see script docstring.")

    # ---- capture state machine: "idle" -> "countdown" -> "capturing" -> "cooldown" -> "idle" ----
    state = "idle"
    countdown_label = None
    countdown_end_time = 0.0
    capture_label = None
    capture_index = 0       # counts ATTEMPTS this burst (0..BURST_SIZE), independent of how many were saved
    burst_saved = 0         # how many of this burst's attempts were actually saved
    burst_skipped = 0       # how many were skipped (insufficient hands)
    next_capture_time = 0.0
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
            # NOTE: the flipped frame is what detection/cropping/saving all
            # operate on, and is what will be fed to the classifier at
            # inference time too - keeping this orientation consistent
            # between training data and live inference matters for signs
            # where hand geometry is asymmetric. Unchanged from before.
            display_frame = frame.copy()

            # Single MediaPipe pass per displayed frame, reused for both the
            # landmark overlay (draw) and the bounding boxes (crop source) -
            # see src/detector.py's get_hand_bboxes() docstring for why this
            # replaces the old manual detector.hands.process(...) call here.
            results, bboxes = detector.get_hand_bboxes(frame)
            detector.draw(display_frame, results)

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
                    burst_saved = 0
                    burst_skipped = 0
                    next_capture_time = now
                    countdown_remaining = None

            if state == "capturing":
                capture_progress = (capture_index, BURST_SIZE)
                if now >= next_capture_time and capture_index < BURST_SIZE:
                    required = HAND_REQUIREMENTS[capture_label]
                    num_hands = len(bboxes)

                    if num_hands < required:
                        # VALID-CROP-ONLY POLICY: skip, don't save. See
                        # module docstring for why this differs from the
                        # old collector's "save anyway, just warn" behavior.
                        burst_skipped += 1
                        logger.warning(
                            f"Skipped attempt {capture_index + 1}/{BURST_SIZE} for '{capture_label}': "
                            f"{num_hands}/{required} hand(s) detected - no crop saved."
                        )
                    else:
                        box = union_bbox(bboxes)
                        cropped_bgr = crop_hand_region(frame, box, padding=DEFAULT_HAND_CROP_PADDING)
                        seq_in_session[capture_label] += 1
                        filename = f"{capture_label}_{session_id}_{seq_in_session[capture_label]:04d}.jpg"
                        out_path = root / capture_label / filename
                        ok = cv2.imwrite(str(out_path), cropped_bgr)
                        if ok:
                            counts[capture_label] += 1
                            total += 1
                            burst_saved += 1
                            logger.info(
                                f"Saved {out_path.name} (crop {cropped_bgr.shape[1]}x{cropped_bgr.shape[0]}) | "
                                f"{capture_label} total: {counts[capture_label]} | overall: {total}"
                            )
                        else:
                            # Roll back the sequence number so we don't leave
                            # a permanent gap for a write that never landed.
                            seq_in_session[capture_label] -= 1
                            logger.warning(f"cv2.imwrite failed for {out_path}, retrying next frame")

                    capture_index += 1
                    next_capture_time = now + BURST_INTERVAL_SECONDS
                    capture_progress = (capture_index, BURST_SIZE)

                if capture_index >= BURST_SIZE:
                    if burst_skipped == 0:
                        status = f"Captured {burst_saved}/{BURST_SIZE} valid crops for '{capture_label}' - hands OK every attempt"
                    elif burst_saved == 0:
                        status = (f"NO VALID CROPS for '{capture_label}' - all {BURST_SIZE} attempts had "
                                  f"insufficient hands. Consider redoing this burst.")
                        logger.warning(status)
                    else:
                        status = (f"Captured {burst_saved}/{BURST_SIZE} valid crops for '{capture_label}' "
                                  f"({burst_skipped} skipped - insufficient hands)")
                        logger.warning(status)
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
                num_hands_now=len(bboxes),
            )
            cv2.imshow("Naruto Hand-Crop Collection", display_frame)
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
        detector.close()
        cv2.destroyAllWindows()

    logger.info(f"Session {session_id} done. Final counts: {counts}")


if __name__ == "__main__":
    main()
