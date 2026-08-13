"""Real-time demo for the Naruto image classifier PLUS the jutsu sequence
engine: camera feed with predicted sign + confidence, and live progress
through the currently-SELECTED jutsu only (press 1/2/3 to select
Rasengan/Chidori/Fireball - see JutsuSequenceEngine.select()).

Reuses the exact same camera/classifier loop from before (no second
pipeline) - the only additions are: construct one JutsuSequenceEngine, let
1/2/3 select which jutsu is active, feed it each frame's classifier
prediction (+ the mouth-O stub/manual override), and draw its progress.

Loads the model exactly once via NarutoImageClassifier's constructor, then
reuses it every frame - unchanged from before.
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
from src.jutsu_sequence import JutsuSequenceEngine, JUTSU_SEQUENCES
from src.mouth_detector import detect_mouth_o

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_naruto_image_classifier")

JUTSU_DISPLAY_NAMES = {"rasengan": "RASENGAN", "chidori": "CHIDORI", "fireball": "FIREBALL"}
JUTSU_SELECT_KEYS = {ord('1'): "rasengan", ord('2'): "chidori", ord('3'): "fireball"}
BANNER_DISPLAY_SECONDS = 2.0


def draw_overlay(frame, label, confidence, fps):
    color = (0, 255, 0) if label != "unknown" else (0, 165, 255)
    cv2.putText(frame, f"Sign: {label.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {confidence*100:.0f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Hold 'm' for mouth-O (placeholder - see src/mouth_detector.py)",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


def draw_jutsu_progress(frame, engine: JutsuSequenceEngine, x_start: int):
    """Draws ONLY the currently-selected jutsu's checklist (if any). Other
    trackers exist internally (engine.trackers) but are intentionally not
    shown here, mirroring that they're not receiving predictions either -
    showing them as if equally "live" would be misleading now that only one
    is actually active."""
    x, y = x_start, 30

    if engine.active_jutsu is None:
        cv2.putText(frame, "No jutsu selected", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2, cv2.LINE_AA)
        y += 24
        cv2.putText(frame, "Press 1/2/3 to select", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        return

    jutsu_name = engine.active_jutsu
    progress = engine.get_progress(jutsu_name)

    cv2.putText(frame, f"Selected Jutsu: {JUTSU_DISPLAY_NAMES[jutsu_name]}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 200), 2, cv2.LINE_AA)
    y += 30

    for step, accepted in zip(progress.steps, progress.accepted):
        mark = "[x]" if accepted else "[ ]"
        is_current = (step == progress.current_step)
        color = (0, 255, 255) if is_current else ((0, 255, 0) if accepted else (180, 180, 180))
        cv2.putText(frame, f"{mark} {step.upper()}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 20

    y += 6
    if progress.on_cooldown:
        cv2.putText(frame, f"(cooldown {progress.cooldown_remaining:.1f}s)", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    elif progress.current_step:
        cv2.putText(frame, f"Current: {progress.current_step.upper()}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_completion_banner(frame, jutsu_name: str):
    text = f"{JUTSU_DISPLAY_NAMES[jutsu_name]}!"
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    x, y = (w - tw) // 2, h // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4, cv2.LINE_AA)


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
    sequence_engine = JutsuSequenceEngine()

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()

    print("NARUTO IMAGE CLASSIFIER + JUTSU SEQUENCE ENGINE - live demo")
    print("Press 1=Rasengan 2=Chidori 3=Fireball to select a jutsu. 'q' to quit.")
    print("Hold 'm' to simulate the mouth-O gesture (placeholder).")

    frame_count = 0
    last_label, last_confidence = "unknown", 0.0
    prev_time = time.time()
    fps = 0.0
    banner_text_until = 0.0
    banner_jutsu = None

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # same orientation convention used during data collection

            if frame_count % infer_every_n == 0:
                last_label, last_confidence = classifier.predict(frame)
            frame_count += 1

            # mouth_o: real detection is a stub (src/mouth_detector.py, always False for
            # now) - holding 'm' is a manual stand-in so Fireball's last step is
            # demoable/testable before a real detector exists. Checked via cv2's key
            # state below (key == ord('m')), not held-state, so it's a per-frame pulse.
            mouth_o_signal = detect_mouth_o(frame)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            draw_overlay(frame, last_label, last_confidence, fps)
            draw_jutsu_progress(frame, sequence_engine, x_start=250)
            if now < banner_text_until and banner_jutsu:
                draw_completion_banner(frame, banner_jutsu)

            cv2.imshow("Naruto Image Classifier", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in JUTSU_SELECT_KEYS:
                sequence_engine.select(JUTSU_SELECT_KEYS[key])
                logger.info(f"Selected jutsu: {JUTSU_SELECT_KEYS[key]}")
            if key == ord('m'):
                mouth_o_signal = True  # manual placeholder trigger, see note above

            completed = sequence_engine.update(last_label, mouth_o=mouth_o_signal, now=now)
            if completed:
                banner_jutsu = completed[0]
                banner_text_until = now + BANNER_DISPLAY_SECONDS
                logger.info(f"JUTSU COMPLETED: {completed}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
