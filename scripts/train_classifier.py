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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
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
    y_raw = df["label"].values

    # Encode string gesture labels to a fixed set of integer indices. RandomForest/SVM
    # would happily fit on the raw strings and hand them straight back from predict(),
    # but XGBoost requires numeric class labels. Encoding up front makes every model's
    # predict()/predict_proba() output consistent (integer indices into label_encoder.classes_),
    # so the inference side never has to guess which convention a given model uses.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    logger.info(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

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
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report_dict = classification_report(
            y_test, y_pred, labels=range(len(label_encoder.classes_)),
            target_names=label_encoder.classes_, zero_division=0, output_dict=True,
        )
        macro_f1 = report_dict["macro avg"]["f1-score"]
        weighted_f1 = report_dict["weighted avg"]["f1-score"]
        cm = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))

        logger.info(f"{name} Test Accuracy: {acc:.4f}")
        logger.info(f"{name} Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
        logger.info(
            "Classification Report (precision / recall / f1 per class):\n"
            + classification_report(
                y_test, y_pred, labels=range(len(label_encoder.classes_)),
                target_names=label_encoder.classes_, zero_division=0,
            )
        )
        logger.info(
            f"Confusion Matrix (rows=true, cols=predicted, order={list(label_encoder.classes_)}):\n{cm}"
        )
        # cross validation
        scores = cross_val_score(model, X, y, cv=3)
        logger.info(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        results[name] = {"test_acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "cv_acc": scores.mean()}
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
            best_name = name

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Bundle the model together with the exact label ordering it was trained on, so
    # gesture_classifier.py never has to rely on config.yaml's hand-maintained
    # label_map (which can silently drift out of sync with the training data / the
    # order sklearn assigns to model.classes_) to decode predictions.
    artifact = {"model": best_model, "classes": list(label_encoder.classes_)}
    joblib.dump(artifact, out_path)

    logger.info("=== FINAL SUMMARY ===")
    for name, r in results.items():
        marker = " <== SELECTED" if name == best_name else ""
        logger.info(
            f"{name}: test_acc={r['test_acc']:.4f} macro_f1={r['macro_f1']:.4f} "
            f"weighted_f1={r['weighted_f1']:.4f} cv_acc={r['cv_acc']:.4f}{marker}"
        )
    logger.info(f"Best model ({best_name}) saved to {out_path}")

if __name__ == "__main__":
    main()