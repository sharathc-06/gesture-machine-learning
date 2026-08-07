"""Script to collect hand landmark data for training."""
import argparse
import csv
import cv2
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.camera import Camera
from src.detector import HandDetector
from src.config import Config, setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("collect_data")

GESTURE_KEYS = {
    ord('1'): "thumbs_up",
    ord('2'): "peace",
    ord('3'): "ok",
    ord('4'): "rock",
    ord('5'): "open_palm",
    ord('6'): "fist",
    ord('7'): "point_left",
    ord('8'): "point_right",
    ord('9'): "finger_gun",
}

def main():
    parser = argparse.ArgumentParser(description="Collect MediaPipe hand landmarks")
    parser.add_argument("--output", type=str, default="data/landmarks.csv", help="Output CSV file")
    args = parser.parse_args()

    config = Config()
    setup_logging(config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    camera = Camera(config.camera["width"], config.camera["height"])
    camera.start()
    detector = HandDetector(
        max_num_hands=config.mediapipe["max_num_hands"],
        model_complexity=config.mediapipe["model_complexity"],
        min_detection_confidence=config.mediapipe["min_detection_confidence"],
        min_tracking_confidence=config.mediapipe["min_tracking_confidence"],
    )

    file_exists = out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
            writer.writerow(header)

        print("Press keys 1-9 to record a sample for the corresponding gesture.")
        print("Press 'q' to quit.")

        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            landmarks, handedness = detector.process(frame)
            if landmarks:
                detector.draw(frame, detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cv2.imshow("Collect Data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in GESTURE_KEYS and landmarks is not None:
                label = GESTURE_KEYS[key]
                writer.writerow([label] + landmarks)
                logger.info(f"Recorded: {label} | Total landmarks: {len(landmarks)}")

    camera.release()
    detector.close()
    cv2.destroyAllWindows()
    logger.info(f"Data saved to {out_path}")

if __name__ == "__main__":
    main()