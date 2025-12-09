from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DEFAULT_TABLE = Path(__file__).resolve().parent.parent / "model_prepare" / "face_landmarks.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RandomForest emotion classifier from landmark CSV data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TABLE,
        help="Path to CSV exported by model_prepare/extract_face_landmarks.py",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "random_forest.joblib",
        help="Destination for the trained RandomForest model.",
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "label_encoder.joblib",
        help="Destination for the fitted LabelEncoder.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON file to store evaluation metrics.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="class_name",
        help="Column that contains class labels.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for validation/testing.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the RandomForest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (None lets trees grow fully).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV.")

    drop_cols = {"file_name", target_column}
    feature_columns = [col for col in df.columns if col not in drop_cols]
    if not feature_columns:
        raise ValueError("No feature columns detected. Did you pass the right CSV?")

    df = df.dropna(subset=feature_columns + [target_column])
    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df[target_column].to_numpy()
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> dict:
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=args.random_state,
    )
    clf.fit(X_train, y_train_encoded)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model": clf,
        "label_encoder": label_encoder,
        "accuracy": accuracy,
        "classification_report": report,
    }


def save_artifacts(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.label_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    joblib.dump(label_encoder, args.label_output)
    logging.info("Saved model to %s", args.model_output)
    logging.info("Saved label encoder to %s", args.label_output)

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "accuracy": metrics["accuracy"],
            "classification_report": metrics["classification_report"],
        }
        with args.metrics_output.open("w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)
        logging.info("Saved metrics to %s", args.metrics_output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    X, y = load_dataset(args.input, args.target_column)
    logging.info("Loaded dataset with %d samples and %d features.", X.shape[0], X.shape[1])

    results = train_model(X, y, args)
    logging.info("Validation accuracy: %.4f", results["accuracy"])

    save_artifacts(
        model=results["model"],
        label_encoder=results["label_encoder"],
        metrics=results,
        args=args,
    )


if __name__ == "__main__":
    main()

