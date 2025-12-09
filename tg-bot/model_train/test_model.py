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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

DEFAULT_TEST_TABLE = Path(__file__).resolve().parent.parent / "model_prepare" / "test_face_landmarks.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MODEL = DEFAULT_ARTIFACT_DIR / "random_forest.joblib"
DEFAULT_LABELS = DEFAULT_ARTIFACT_DIR / "label_encoder.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test trained RandomForest model on test dataset and output quality metrics."
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=DEFAULT_TEST_TABLE,
        help="Path to test CSV file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to trained RandomForest model (joblib).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Path to fitted LabelEncoder (joblib).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="class_name",
        help="Column that contains class labels.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON file to store evaluation metrics.",
    )
    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=None,
        help="Optional CSV file to save confusion matrix.",
    )
    return parser.parse_args()


def load_model_and_encoder(
    model_path: Path, labels_path: Path
) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """Load trained model and label encoder."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Label encoder file not found: {labels_path}")

    logging.info("Loading model from %s", model_path)
    model: RandomForestClassifier = joblib.load(model_path)
    logging.info("Loading label encoder from %s", labels_path)
    label_encoder: LabelEncoder = joblib.load(labels_path)

    return model, label_encoder


def load_test_dataset(csv_path: Path, target_column: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load test dataset and return features, labels, and full dataframe."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    logging.info("Loading test dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV columns: {df.columns.tolist()}")

    drop_cols = {"file_name", target_column}
    feature_columns = [col for col in df.columns if col not in drop_cols]
    if not feature_columns:
        raise ValueError("No feature columns detected. Did you pass the right CSV?")

    # Remove rows with missing values
    df_clean = df.dropna(subset=feature_columns + [target_column])
    if len(df_clean) < len(df):
        logging.warning(
            "Removed %d rows with missing values (%d remaining)",
            len(df) - len(df_clean),
            len(df_clean),
        )

    X = df_clean[feature_columns].to_numpy(dtype=np.float32)
    y = df_clean[target_column].to_numpy()

    logging.info("Loaded test dataset: %d samples, %d features", X.shape[0], X.shape[1])

    return X, y, df_clean


def evaluate_model(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate model on test set and return comprehensive metrics."""
    # Encode labels
    y_test_encoded = label_encoder.transform(y_test)

    # Check feature dimension match
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features and X_test.shape[1] != expected_features:
        raise ValueError(
            f"Feature dimension mismatch: model expects {expected_features} features, "
            f"but test data has {X_test.shape[1]} features."
        )

    # Make predictions
    logging.info("Making predictions on test set...")
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    precision_macro = precision_score(y_test_encoded, y_pred_encoded, average="macro", zero_division=0)
    recall_macro = recall_score(y_test_encoded, y_pred_encoded, average="macro", zero_division=0)
    f1_macro = f1_score(y_test_encoded, y_pred_encoded, average="macro", zero_division=0)
    precision_weighted = precision_score(
        y_test_encoded, y_pred_encoded, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_test_encoded, y_pred_encoded, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test_encoded, y_pred_encoded, average="weighted", zero_division=0)

    # Classification report
    class_report = classification_report(
        y_test_encoded,
        y_pred_encoded,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred_encoded, labels=range(len(label_encoder.classes_)))

    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_true_encoded": y_test_encoded,
        "y_pred_encoded": y_pred_encoded,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": class_report,
        "confusion_matrix": cm.tolist(),
        "class_names": label_encoder.classes_.tolist(),
    }


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a readable format."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nTest Set Size: {len(metrics['y_true'])} samples")
    print(f"Number of Classes: {len(metrics['class_names'])}")

    print("\n" + "-" * 70)
    print("Overall Metrics:")
    print("-" * 70)
    print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (weighted):  {metrics['f1_weighted']:.4f}")

    # Class distribution
    print("\n" + "-" * 70)
    print("Class Distribution in Test Set:")
    print("-" * 70)
    from collections import Counter

    class_counts = Counter(metrics["y_true"])
    for class_name in metrics["class_names"]:
        count = class_counts.get(class_name, 0)
        percentage = count / len(metrics["y_true"]) * 100 if len(metrics["y_true"]) > 0 else 0
        print(f"  {class_name:20s}: {count:4d} samples ({percentage:5.1f}%)")

    # Per-class metrics
    print("\n" + "-" * 70)
    print("Per-Class Metrics:")
    print("-" * 70)
    class_report = metrics["classification_report"]
    for class_name in metrics["class_names"]:
        if class_name in class_report:
            metrics_dict = class_report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics_dict.get('precision', 0):.4f}")
            print(f"  Recall:    {metrics_dict.get('recall', 0):.4f}")
            print(f"  F1-Score:  {metrics_dict.get('f1-score', 0):.4f}")
            print(f"  Support:   {metrics_dict.get('support', 0):.0f}")

    # Confusion matrix
    print("\n" + "-" * 70)
    print("Confusion Matrix:")
    print("-" * 70)
    cm = np.array(metrics["confusion_matrix"])
    class_names = metrics["class_names"]

    # Print header
    print(f"{'Actual \\ Predicted':20s}", end="")
    for name in class_names:
        print(f"{name[:10]:>12s}", end="")
    print()

    # Print rows
    for i, true_class in enumerate(class_names):
        print(f"{true_class[:20]:20s}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:12d}", end="")
        print(f"  (Total: {cm[i, :].sum()})")

    print("\n" + "=" * 70)


def save_results(metrics: dict, metrics_output: Path, confusion_matrix_output: Path | None) -> None:
    """Save evaluation results to files."""
    if metrics_output:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metrics payload (exclude arrays from JSON)
        metrics_payload = {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
            "classification_report": metrics["classification_report"],
            "confusion_matrix": metrics["confusion_matrix"],
            "class_names": metrics["class_names"],
            "test_set_size": len(metrics["y_true"]),
        }

        with metrics_output.open("w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2, ensure_ascii=False)
        logging.info("Saved metrics to %s", metrics_output)

    if confusion_matrix_output:
        confusion_matrix_output.parent.mkdir(parents=True, exist_ok=True)
        cm = np.array(metrics["confusion_matrix"])
        class_names = metrics["class_names"]

        # Create DataFrame with proper labels
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(confusion_matrix_output)
        logging.info("Saved confusion matrix to %s", confusion_matrix_output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    # Load model and encoder
    model, label_encoder = load_model_and_encoder(args.model, args.labels)

    # Load test dataset
    X_test, y_test, df_test = load_test_dataset(args.test_input, args.target_column)

    # Show class distribution
    from collections import Counter

    class_dist = Counter(y_test)
    logging.info("Test set class distribution: %s", dict(class_dist))

    # Evaluate model
    metrics = evaluate_model(model, label_encoder, X_test, y_test)

    # Print results
    print_metrics(metrics)

    # Save results if requested
    save_results(metrics, args.metrics_output, args.confusion_matrix_output)

    logging.info("Model evaluation completed!")


if __name__ == "__main__":
    main()

