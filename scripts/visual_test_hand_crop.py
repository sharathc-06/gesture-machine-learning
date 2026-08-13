"""TEMPORARY standalone visual proof-of-concept for the hand-crop pipeline.

This is NOT part of the real collection/training/inference pipeline and is
not imported by anything else. It exists purely so you can visually confirm,
before any dataset work happens, that:

  - the union bounding box (padded) reliably contains the full hand sign
  - the resulting 224x224 crop looks like something a CNN could actually
    learn from, rather than mostly background

It deliberately does NOT:
  - save any images to disk
  - run the CNN / load any model
  - touch data/naruto_images or models/naruto_image_model.pth in any way
  - modify src/jutsu_sequence.py or any jutsu behavior (not even imported)

Run with:  python scripts/visual_test_hand_crop.py
Quit with: 'q'

Two windows are shown side by side:
  "Webcam (landmarks + bbox)" - the normal flipped camera view, with
      MediaPipe hand landmarks drawn, plus the padded union bounding box
      (green) that would be cropped, plus status text.
  "224x224 crop (CNN input preview)" - the actual crop, letterboxed to
      224x224 EXACTLY the way src/image_preprocessing.letterbox_to_square
      does it today, i.e. the real preprocessing function is reused as-is
      rather than reimplemented here, so what you see is what the CNN
      would eventually receive.
"""
import sys
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

# Easy single place to tune padding while eyeballing results - see
# src/image_preprocessing.DEFAULT_HAND_CROP_PADDING for the exact semantics
# (fraction of box width/height added on each side).
PADDING = DEFAULT_HAND_CROP_PADDING

CROP_PREVIEW_SIZE = 224


def draw_status(frame, num_hands, has_crop, bbox_int):
    y = 25
    line_h = 24

    def put(text, color=(255, 255, 255), scale=0.6, thickness=2):
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_h

    put("HAND-CROP VISUAL TEST (no save, no model)", (0, 255, 255), 0.6, 2)
    put(f"Hands detected: {num_hands}/2")
    put(f"Crop: {'YES' if has_crop else 'NO'}", (0, 255, 0) if has_crop else (0, 0, 255))
    if bbox_int is not None:
        x1, y1, x2, y2 = bbox_int
        put(f"Padded bbox: ({x1},{y1})-({x2},{y2})", (200, 200, 200), 0.5, 1)
    put("Press 'q' to quit", (150, 150, 150), 0.5, 1)


def main():
    config = Config()
    setup_logging(config)

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()
    # max_num_hands is intentionally forced to 2 here regardless of
    # config.yaml's current value (still 1 at the time of this test) - the
    # visual test needs to be able to show two-hand union crops even before
    # config.yaml itself is updated for the real pipeline.
    detector = HandDetector(
        max_num_hands=2,
        model_complexity=config.mediapipe["model_complexity"],
        min_detection_confidence=config.mediapipe["min_detection_confidence"],
        min_tracking_confidence=config.mediapipe["min_tracking_confidence"],
    )

    print("Hand-crop visual test running. Press 'q' to quit.")
    print(f"Padding = {PADDING} (fraction of bbox width/height, each side)")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # same convention as the rest of the codebase
            display_frame = frame.copy()

            results, bboxes = detector.get_hand_bboxes(frame)

            # Draw raw landmarks (reuses the existing draw() method as-is).
            detector.draw(display_frame, results)

            has_crop = len(bboxes) > 0
            bbox_int = None
            crop_preview = None

            if has_crop:
                box = union_bbox(bboxes) if len(bboxes) > 1 else bboxes[0]
                cropped_bgr = crop_hand_region(frame, box, padding=PADDING)

                # Recover the actual padded+clipped box in pixel ints, purely
                # for drawing/status text - crop_hand_region already does the
                # equivalent clipping internally; this mirrors that here so
                # what's drawn matches what's cropped.
                frame_h, frame_w = frame.shape[:2]
                x_min, y_min, x_max, y_max = box
                box_w = max(1.0, x_max - x_min)
                box_h = max(1.0, y_max - y_min)
                pad_x = box_w * PADDING
                pad_y = box_h * PADDING
                x1 = max(0, min(int(round(x_min - pad_x)), frame_w - 1))
                y1 = max(0, min(int(round(y_min - pad_y)), frame_h - 1))
                x2 = max(x1 + 1, min(int(round(x_max + pad_x)), frame_w))
                y2 = max(y1 + 1, min(int(round(y_max + pad_y)), frame_h))
                bbox_int = (x1, y1, x2, y2)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Exact same letterbox function used by training/inference
                # today - not reimplemented, so this preview is trustworthy.
                pil_crop = bgr_ndarray_to_pil(cropped_bgr)
                letterboxed = letterbox_to_square(pil_crop, CROP_PREVIEW_SIZE)
                crop_preview = cv2.cvtColor(np.array(letterboxed), cv2.COLOR_RGB2BGR)

            draw_status(display_frame, len(bboxes), has_crop, bbox_int)

            if crop_preview is None:
                crop_preview = np.zeros((CROP_PREVIEW_SIZE, CROP_PREVIEW_SIZE, 3), dtype=np.uint8)
                cv2.putText(crop_preview, "NO HAND", (30, CROP_PREVIEW_SIZE // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Webcam (landmarks + bbox)", display_frame)
            cv2.imshow("224x224 crop (CNN input preview)", crop_preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
