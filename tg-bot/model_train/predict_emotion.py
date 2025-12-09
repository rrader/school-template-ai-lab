from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence

import joblib
import mediapipe as mp
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

NUM_LANDMARKS = 478
DEFAULT_IMAGE_SIZE = (224, 224)  # Pillow uses this only for debugging; FaceMesh needs raw image.
DEFAULT_MODEL = Path(__file__).resolve().parent / "artifacts" / "random_forest.joblib"
DEFAULT_LABELS = Path(__file__).resolve().parent / "artifacts" / "label_encoder.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict facial emotion for a single photo using the trained RandomForest model."
    )
    parser.add_argument("image", type=Path, help="Path to the input face photo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the trained RandomForest model (joblib).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Path to the fitted LabelEncoder (joblib).",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe Face Mesh.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many class probabilities to print (ignored when model lacks predict_proba).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional file path to dump the prediction payload as JSON.",
    )
    return parser.parse_args()


def load_artifacts(model_path: Path, labels_path: Path) -> tuple[RandomForestClassifier, LabelEncoder]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label encoder file not found: {labels_path}")

    model: RandomForestClassifier = joblib.load(model_path)
    label_encoder: LabelEncoder = joblib.load(labels_path)
    return model, label_encoder


def extract_landmarks(image_path: Path, detection_confidence: float) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=detection_confidence,
    ) as face_mesh:
        results = face_mesh.process(image_array)

    if not results.multi_face_landmarks:
        raise RuntimeError(f"Unable to detect face landmarks in {image_path}")

    landmarks = results.multi_face_landmarks[0].landmark
    if len(landmarks) != NUM_LANDMARKS:
        logging.warning(
            "Expected %d landmarks but got %d for %s", NUM_LANDMARKS, len(landmarks), image_path
        )

    features: List[float] = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])

    return np.asarray(features, dtype=np.float32)


def format_top_probabilities(
    probabilities: Sequence[float], label_encoder: LabelEncoder, top_k: int | None
) -> list[dict[str, float]]:
    if not probabilities:
        return []

    encoded_indices = np.arange(len(probabilities))
    class_names = label_encoder.inverse_transform(encoded_indices)
    sorted_indices = np.argsort(probabilities)[::-1]
    if top_k:
        sorted_indices = sorted_indices[:top_k]

    result = []
    for idx in sorted_indices:
        result.append(
            {
                "label": str(class_names[idx]),
                "probability": float(probabilities[idx]),
            }
        )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    model, label_encoder = load_artifacts(args.model, args.labels)
    features = extract_landmarks(args.image, args.detection_confidence)

    expected_features = getattr(model, "n_features_in_", features.size)
    if features.size != expected_features:
        raise ValueError(
            f"Feature length mismatch. Model expects {expected_features} values, got {features.size}."
        )

    features_batch = features.reshape(1, -1)
    encoded_prediction = model.predict(features_batch)[0]
    predicted_label = label_encoder.inverse_transform([encoded_prediction])[0]

    payload = {"label": str(predicted_label)}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_batch)[0]
        payload["top_probabilities"] = format_top_probabilities(probabilities, label_encoder, args.top_k)

    print(f"Predicted emotion: {payload['label']}")
    top_probs = payload.get("top_probabilities")
    if top_probs:
        for entry in top_probs:
            label = entry["label"]
            prob = entry["probability"]
            print(f"  {label}: {prob:.3f}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logging.info("Saved prediction payload to %s", args.json_output)


if __name__ == "__main__":
    main()

