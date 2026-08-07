"""Train a gesture classifier from collected landmarks and save the best model."""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import Config, setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_classifier")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/landmarks.csv")
    parser.add_argument("--model-out", default="models/gesture_model.pkl")
    args = parser.parse_args()

    config = Config()
    setup_logging(config)

    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} samples")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"{name} Accuracy: {acc:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        # cross validation
        scores = cross_val_score(model, X, y, cv=3)
        logger.info(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
            best_name = name

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)
    logger.info(f"Best model ({best_name}) saved to {out_path}")

if __name__ == "__main__":
    main()