"""TEMPORARY diagnostic for the TWO-HAND DETECTION reliability problem
(touching/interlocked signs like Tiger and Ram).

This is separate from scripts/visual_test_hand_crop.py (which already
confirmed the crop pipeline looks correct once MediaPipe finds both hands).
This script exists to answer: HOW OFTEN and WHY does MediaPipe fail to
report both hands for poses where the hands are touching/overlapping, and
(this update) whether a short, conservative TEMPORAL STABILIZATION window
can bridge brief single-frame detection drops without hallucinating hands.

Does NOT save any images, does NOT run the CNN, does NOT touch
data/naruto_hand_crops or data/naruto_images in any way.

DIAGNOSTIC RESULTS SO FAR (informing this script's defaults)
------------------------------------------------------------------
Empirically, on this project's own recorded comparisons:
  complexity=1, det=0.3,  track=0.3,  static=False, scale=1.0 -> 22.4% 2-hand
  complexity=1, det=0.5,  track=0.5,  static=False, scale=1.0 -> 67.2% 2-hand
  complexity=1, det=0.5,  track=0.5,  static=True,  scale=1.0 ->  0.0% 2-hand
  complexity=1, det=0.5,  track=0.5,  static=False, scale=1.5 ->  5.4% 2-hand
So this script's defaults are now model_complexity=1,
min_detection_confidence=0.5, min_tracking_confidence=0.5,
static_image_mode=False, detection_scale=1.0 - the best-tested config so
far - with all of them still overridable via CLI for further comparison.

WHAT IT SHOWS
-------------
1. MediaPipe's detected hand landmarks (drawn, via HandDetector.draw())
2. RAW hand count this frame (0/1/2) plus rolling RAW 2-hand rate
3. EACH detected hand's individual bounding box (distinct color per hand),
   not just the union
4. The padded union bounding box (green) for the RAW detection
5. Handedness label ("Left"/"Right") next to each detected hand's box
6. NEW - STABILIZED state: RAW vs STABLE hand count, SOURCE
   (FRESH/RETAINED/NONE), and STALE (how many consecutive frames the
   current retained boxes have been reused)
7. NEW - two crop previews side by side conceptually (two separate
   windows): the RAW crop (from this frame's raw detection only, "NO HAND"
   if raw doesn't meet the target) and the STABILIZED crop (from the
   stabilizer's current state, "INVALID" if state is NONE) - so you can
   visually compare whether stabilization is actually giving you a better
   crop or just a stale one.
8. NEW - stabilization statistics: raw 2-hand rate, stabilized "valid"
   rate (FRESH+RETAINED), fresh-frame rate, retained-frame rate, and a
   bridge count (see TwoHandStabilizer docstring for the exact definition).

WHAT IT LETS YOU TEST
----------------------
All exposed as CLI flags so you can A/B test without editing code:

  --model-complexity {0,1}       MediaPipe Lite (0) vs Full (1)
  --min-detection-confidence F   MediaPipe min_detection_confidence
  --min-tracking-confidence F    MediaPipe min_tracking_confidence
  --static-image-mode            run full detection every frame (no tracking)
  --detection-scale F            resize before detection, then rescale boxes back
  --no-flip                      skip the horizontal flip before detection
  --target-hands N               how many hands "counts" as a valid detection
                                  for stabilization purposes (default 2, since
                                  this script is specifically about Tiger/Ram-
                                  style two-hand signs)
  --max-stale-frames N           how many consecutive sub-target frames a
                                  previous valid detection may be retained for
                                  before being discarded (default 3; try 0/1/3/5)

Press 'q' to quit (prints a final summary), 'r' to reset all stats/stabilizer
state without quitting.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.camera import Camera
from src.config import Config, setup_logging
from src.detector import HandDetector
from src.image_preprocessing import (
    DEFAULT_HAND_CROP_PADDING,
    crop_hand_region,
    letterbox_to_square,
    union_bbox,
    bgr_ndarray_to_pil,
)

CROP_PREVIEW_SIZE = 224
ROLLING_WINDOW = 60  # frames (~2s at 30fps) for the "recent reliability" stats

HAND_COLORS = [(0, 0, 255), (255, 128, 0)]  # red, blue - per-hand individual bboxes
UNION_COLOR = (0, 255, 0)                    # green - the padded union box (raw)
STABLE_COLOR = (0, 255, 255)                 # yellow - the stabilized union box


class TwoHandStabilizer:
    """Bridges brief, single-frame drops below `target_hands` detected hands
    by conservatively retaining the most recent detection that DID meet the
    target, for up to `max_stale_frames` consecutive frames.

    Deliberately simple/explicit state machine, not a Kalman filter or any
    kind of prediction - it never invents box positions, it only re-uses the
    exact last-known-good boxes verbatim, and only for a short, configurable
    window. This is intentional per the safety requirements: a hallucinated
    guess at where hands "probably" are would be worse than no detection.

    States (see .update()):
      FRESH    - this frame's raw detection itself met target_hands.
      RETAINED - this frame's raw detection did NOT meet target_hands, but a
                 recent FRESH detection is still within its retention window,
                 so that detection's boxes are reused as-is.
      NONE     - no valid boxes to offer (either never had a fresh detection,
                 or the retention window has been exceeded - in which case
                 the stale boxes are discarded, not just left unused, so a
                 later query can't accidentally resurrect them).

    max_stale_frames=0 means no retention at all (NONE the instant raw drops
    below target) - i.e. behaves exactly like the un-stabilized raw signal.
    """

    def __init__(self, target_hands: int = 2, max_stale_frames: int = 3):
        self.target_hands = target_hands
        self.max_stale_frames = max_stale_frames
        self._retained_bboxes = None
        self._stale_count = 0
        self._prev_source = "NONE"  # for bridge-detection, see update()

        # Stats
        self.total_frames = 0
        self.raw_valid_frames = 0       # raw_count >= target_hands
        self.fresh_frames = 0
        self.retained_frames = 0
        self.none_frames = 0
        self.bridge_count = 0           # see update() docstring below

    def update(self, raw_bboxes):
        """Feed one frame's raw bboxes in. Returns (stable_bboxes_or_None,
        source, stale_count) where source is "FRESH", "RETAINED", or "NONE".

        A "bridge" is counted whenever a RETAINED stretch is immediately
        followed by a FRESH frame (i.e. retention successfully carried
        through a temporary drop until real detection resumed, rather than
        the window expiring into NONE first) - this is the concrete count
        for "number of times stabilization successfully bridged a temporary
        detection loss".
        """
        self.total_frames += 1
        raw_count = len(raw_bboxes)
        if raw_count >= self.target_hands:
            self.raw_valid_frames += 1

        if raw_count >= self.target_hands:
            source = "FRESH"
            self._retained_bboxes = list(raw_bboxes)
            self._stale_count = 0
            stable = self._retained_bboxes
        elif self._retained_bboxes is not None and self._stale_count < self.max_stale_frames:
            self._stale_count += 1
            source = "RETAINED"
            stable = self._retained_bboxes
        else:
            # Either never had a fresh detection, or the window is exceeded -
            # discard so a later frame can't accidentally reuse this.
            self._retained_bboxes = None
            self._stale_count = 0
            source = "NONE"
            stable = None

        if source == "FRESH" and self._prev_source == "RETAINED":
            self.bridge_count += 1

        if source == "FRESH":
            self.fresh_frames += 1
        elif source == "RETAINED":
            self.retained_frames += 1
        else:
            self.none_frames += 1

        self._prev_source = source
        return stable, source, self._stale_count

    def reset(self):
        self._retained_bboxes = None
        self._stale_count = 0
        self._prev_source = "NONE"
        self.total_frames = 0
        self.raw_valid_frames = 0
        self.fresh_frames = 0
        self.retained_frames = 0
        self.none_frames = 0
        self.bridge_count = 0


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose two-hand detection reliability + temporal stabilization")
    p.add_argument("--model-complexity", type=int, choices=[0, 1], default=1,
                    help="MediaPipe model complexity: 0=Lite, 1=Full (default 1 - best-tested so far)")
    p.add_argument("--min-detection-confidence", type=float, default=0.5,
                    help="default 0.5 - best-tested so far (0.3 was substantially worse)")
    p.add_argument("--min-tracking-confidence", type=float, default=0.5,
                    help="default 0.5 - best-tested so far")
    p.add_argument("--static-image-mode", action="store_true",
                    help="Run full detection every frame (empirically much WORSE - off by default)")
    p.add_argument("--detection-scale", type=float, default=1.0,
                    help="Resize factor before detection (empirically 1.5x was much WORSE; default 1.0)")
    p.add_argument("--no-flip", action="store_true",
                    help="Skip the horizontal flip before detection")
    p.add_argument("--padding", type=float, default=DEFAULT_HAND_CROP_PADDING,
                    help=f"Union-bbox padding fraction for crop previews (default {DEFAULT_HAND_CROP_PADDING})")
    p.add_argument("--target-hands", type=int, default=2,
                    help="Hand count that counts as a valid/fresh detection for stabilization (default 2)")
    p.add_argument("--max-stale-frames", type=int, default=3,
                    help="Consecutive sub-target frames a prior valid detection may be retained for "
                         "(default 3; try 0, 1, 3, 5 - see module docstring)")
    return p.parse_args()


def draw_status(frame, args, raw_count, stable_count, source, stale_count,
                 rolling_raw, rolling_stable_valid, handedness_labels):
    y = 20
    line_h = 19

    def put(text, color=(255, 255, 255), scale=0.5, thickness=1):
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_h

    put("TWO-HAND DETECTION + STABILIZATION DIAGNOSTIC", (0, 255, 255), 0.52, 2)
    put(f"complexity={args.model_complexity} det_conf={args.min_detection_confidence} "
        f"track_conf={args.min_tracking_confidence} static={args.static_image_mode} "
        f"scale={args.detection_scale}", (200, 200, 200), 0.42)
    put(f"target_hands={args.target_hands}  max_stale_frames={args.max_stale_frames}", (200, 200, 200))

    raw_color = (0, 255, 0) if raw_count >= args.target_hands else ((0, 165, 255) if raw_count > 0 else (0, 0, 255))
    put(f"RAW: {raw_count}/{args.target_hands}", raw_color, 0.6, 2)

    if source == "NONE":
        put("STABLE: INVALID   SOURCE: NONE", (0, 0, 255), 0.6, 2)
    else:
        stable_color = (0, 255, 0) if source == "FRESH" else (0, 255, 255)
        put(f"STABLE: {stable_count}/{args.target_hands}   SOURCE: {source}   STALE: {stale_count}",
            stable_color, 0.6, 2)

    if rolling_raw:
        n = len(rolling_raw)
        raw_rate = sum(rolling_raw) / n
        stable_rate = sum(rolling_stable_valid) / n
        put(f"Last {n} frames: RAW valid={raw_rate*100:.0f}%  STABLE valid={stable_rate*100:.0f}%",
            (0, 255, 0) if stable_rate > 0.8 else (0, 165, 255))

    if handedness_labels:
        put(f"Handedness (raw): {handedness_labels}", (180, 180, 255), 0.42)

    y += 4
    put("Press 'q' to quit (prints summary), 'r' to reset stats", (150, 150, 150), 0.42)


def make_crop_preview(frame, bboxes, padding, label_if_empty):
    if not bboxes:
        preview = np.zeros((CROP_PREVIEW_SIZE, CROP_PREVIEW_SIZE, 3), dtype=np.uint8)
        cv2.putText(preview, label_if_empty, (12, CROP_PREVIEW_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return preview
    box = union_bbox(bboxes) if len(bboxes) > 1 else bboxes[0]
    cropped_bgr = crop_hand_region(frame, box, padding=padding)
    pil_crop = bgr_ndarray_to_pil(cropped_bgr)
    letterboxed = letterbox_to_square(pil_crop, CROP_PREVIEW_SIZE)
    return cv2.cvtColor(np.array(letterboxed), cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()

    config = Config()
    setup_logging(config)

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()
    detector = HandDetector(
        max_num_hands=max(2, args.target_hands),
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        static_image_mode=args.static_image_mode,
    )
    stabilizer = TwoHandStabilizer(target_hands=args.target_hands, max_stale_frames=args.max_stale_frames)

    print("Two-hand detection + stabilization diagnostic running.")
    print(f"Args: {vars(args)}")
    print("Perform Tiger/Ram (or any sign) in front of the camera. Press 'q' to quit and see a summary.")

    rolling_raw = deque(maxlen=ROLLING_WINDOW)           # 1 if raw_count >= target, else 0
    rolling_stable_valid = deque(maxlen=ROLLING_WINDOW)  # 1 if source != NONE, else 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            if not args.no_flip:
                frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            if args.detection_scale != 1.0:
                h, w = frame.shape[:2]
                scaled = cv2.resize(frame, (int(w * args.detection_scale), int(h * args.detection_scale)))
                results, scaled_bboxes = detector.get_hand_bboxes(scaled)
                inv = 1.0 / args.detection_scale
                raw_bboxes = [(x1 * inv, y1 * inv, x2 * inv, y2 * inv) for (x1, y1, x2, y2) in scaled_bboxes]
            else:
                results, raw_bboxes = detector.get_hand_bboxes(frame)

            detector.draw(display_frame, results)

            handedness_labels = []
            if results.multi_hand_landmarks:
                handedness_list = results.multi_handedness if results.multi_handedness else []
                for i, box in enumerate(raw_bboxes):
                    color = HAND_COLORS[i % len(HAND_COLORS)]
                    x1, y1, x2, y2 = (int(v) for v in box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = "?"
                    if i < len(handedness_list):
                        label = handedness_list[i].classification[0].label
                    handedness_labels.append(label)
                    cv2.putText(display_frame, label, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            stable_bboxes, source, stale_count = stabilizer.update(raw_bboxes)

            if stable_bboxes:
                box = union_bbox(stable_bboxes) if len(stable_bboxes) > 1 else stable_bboxes[0]
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), STABLE_COLOR, 2)

            rolling_raw.append(1 if len(raw_bboxes) >= args.target_hands else 0)
            rolling_stable_valid.append(0 if source == "NONE" else 1)

            draw_status(display_frame, args, len(raw_bboxes),
                        len(stable_bboxes) if stable_bboxes else 0, source, stale_count,
                        rolling_raw, rolling_stable_valid, handedness_labels)

            raw_crop = make_crop_preview(frame, raw_bboxes, args.padding, "RAW: NO HAND")
            stable_crop = make_crop_preview(frame, stable_bboxes or [], args.padding, "STABLE: INVALID")

            cv2.imshow("Two-Hand Detection + Stabilization Diagnostic", display_frame)
            cv2.imshow("RAW crop preview", raw_crop)
            cv2.imshow("STABILIZED crop preview", stable_crop)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                rolling_raw.clear()
                rolling_stable_valid.clear()
                stabilizer.reset()
                print("Stats and stabilizer state reset.")
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    if stabilizer.total_frames:
        n = stabilizer.total_frames
        print(f"\n=== SESSION SUMMARY ({n} frames) ===")
        print(f"Args: {vars(args)}")
        print(f"RAW >= target_hands:        {stabilizer.raw_valid_frames}/{n} "
              f"({stabilizer.raw_valid_frames/n*100:.1f}%)")
        print(f"STABLE valid (FRESH+RETAINED): {stabilizer.fresh_frames + stabilizer.retained_frames}/{n} "
              f"({(stabilizer.fresh_frames + stabilizer.retained_frames)/n*100:.1f}%)")
        print(f"  FRESH:    {stabilizer.fresh_frames}/{n} ({stabilizer.fresh_frames/n*100:.1f}%)")
        print(f"  RETAINED: {stabilizer.retained_frames}/{n} ({stabilizer.retained_frames/n*100:.1f}%)")
        print(f"  NONE:     {stabilizer.none_frames}/{n} ({stabilizer.none_frames/n*100:.1f}%)")
        print(f"Bridge count (RETAINED stretch -> FRESH resumed): {stabilizer.bridge_count}")
        print("Compare STABLE-valid% and bridge count across different --max-stale-frames values "
              "(0/1/3/5) to see whether stabilization is actually helping vs. just retaining stale boxes.")


if __name__ == "__main__":
    main()
