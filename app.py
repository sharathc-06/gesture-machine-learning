"""Real‑time hand gesture meme player."""
import cv2
import logging
from pathlib import Path

from src.config import Config, setup_logging
from src.camera import Camera
from src.detector import HandDetector
from src.gesture_classifier import (
    BaseGestureClassifier,
    RuleBasedClassifier,
    MLGestureClassifier,
)
from src.player import MediaPlayer
from src.utils import FPSCounter, PredictionSmoother, draw_info

logger = logging.getLogger("app")


def build_classifier(config: Config) -> BaseGestureClassifier:
    """Instantiate the appropriate gesture classifier based on config."""
    ctype = config.classifier["type"]
    if ctype == "ml":
        model_path = config.model_path
        if not model_path.exists():
            logger.warning(f"ML model not found at {model_path}, falling back to rule‑based classifier.")
            return RuleBasedClassifier()
        return MLGestureClassifier(str(model_path), config.label_map)
    else:
        return RuleBasedClassifier()


def main():
    config = Config()
    setup_logging(config)

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()

    detector = HandDetector(
        max_num_hands=config.mediapipe["max_num_hands"],
        model_complexity=config.mediapipe["model_complexity"],
        min_detection_confidence=config.mediapipe["min_detection_confidence"],
        min_tracking_confidence=config.mediapipe["min_tracking_confidence"],
    )

    classifier = build_classifier(config)
    player = MediaPlayer(config.audio_dir, config.video_dir)
    fps_counter = FPSCounter()
    smoother = PredictionSmoother(
        stable_frames=config.debounce["stable_frames"],
        cooldown_seconds=config.debounce["cooldown_seconds"],
    )

    logger.info("Starting main loop. Press 'q' to quit.")
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Hand detection
        landmarks, handedness = detector.process(frame)

        # Draw skeleton
        # We need the raw results for drawing, but process() doesn't return them.
        # We'll call hands.process() again for drawing (inefficient but acceptable).
        # Better: modify detector to return results. For brevity, we re‑process.
        # Here we redesign a little: add a draw method that accepts frame and results.
        # Since we already processed, we can store the results object. Let's adjust:
        # We'll modify detector.process to also return results object. For now, re‑process.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(rgb)
        if results.multi_hand_landmarks:
            detector.draw(frame, results)

        fps_counter.update()
        fps_counter.draw(frame)

        if landmarks:
            gesture, confidence = classifier.predict(landmarks)
            # Smooth and debounce
            should_trigger, final_gesture, avg_conf = smoother.update(gesture, confidence)
            draw_info(frame, final_gesture, avg_conf)

            if should_trigger:
                logger.info(f"Triggering gesture: {final_gesture}")
                player.trigger_gesture(final_gesture, config.gesture_mappings)
        else:
            draw_info(frame, "no hand", 0.0)

        cv2.imshow("Gesture Meme Player", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    detector.close()
    player.cleanup()
    cv2.destroyAllWindows()
    logger.info("Application exited.")


if __name__ == "__main__":
    main()