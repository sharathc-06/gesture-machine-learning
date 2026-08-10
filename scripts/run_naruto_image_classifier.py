"""Basic real-time demo for the Naruto image classifier: camera feed with
predicted sign + confidence overlaid. No sequence engine, no game menu, no
effects - just proving the classifier works live, per this phase's scope.

Loads the model exactly once via NarutoImageClassifier's constructor, then
reuses it every frame.
"""
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.camera import Camera
from src.config import Config, setup_logging
from src.naruto_image_classifier import NarutoImageClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_naruto_image_classifier")


def draw_overlay(frame, label, confidence, fps):
    color = (0, 255, 0) if label != "unknown" else (0, 165, 255)
    cv2.putText(frame, f"Sign: {label.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {confidence*100:.0f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    config = Config()
    setup_logging(config)

    model_path = config.naruto_image_model_path
    if not model_path.exists():
        logger.error(
            f"No trained model found at {model_path}. Run "
            f"scripts/train_naruto_image_classifier.py first."
        )
        sys.exit(1)

    classifier = NarutoImageClassifier(
        str(model_path), confidence_threshold=config.naruto_image_confidence_threshold
    )
    infer_every_n = max(1, config.naruto_inference_every_n_frames)

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()

    print("NARUTO IMAGE CLASSIFIER - live demo")
    print("Press 'q' to quit.")

    frame_count = 0
    last_label, last_confidence = "unknown", 0.0
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # same orientation convention used during data collection

            if frame_count % infer_every_n == 0:
                last_label, last_confidence = classifier.predict(frame)
            frame_count += 1

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            draw_overlay(frame, last_label, last_confidence, fps)
            cv2.imshow("Naruto Image Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
