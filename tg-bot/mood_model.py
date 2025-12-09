from __future__ import annotations

from pathlib import Path
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from model_train.predict_emotion import (
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    extract_landmarks,
    load_artifacts,
)

DEFAULT_DETECTION_CONFIDENCE = 0.5

_MODEL: RandomForestClassifier | None = None
_LABEL_ENCODER: LabelEncoder | None = None


def _ensure_model_loaded(
    model_path: Path = DEFAULT_MODEL, labels_path: Path = DEFAULT_LABELS
) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """Load RandomForest artifacts once and reuse them between predictions."""
    global _MODEL, _LABEL_ENCODER
    if _MODEL is None or _LABEL_ENCODER is None:
        _MODEL, _LABEL_ENCODER = load_artifacts(model_path, labels_path)
    return _MODEL, _LABEL_ENCODER


def predict(image_path: str | Path, detection_confidence: float = DEFAULT_DETECTION_CONFIDENCE) -> str:
    """Predict mood label for a photo using the trained RandomForest model."""
    model, label_encoder = _ensure_model_loaded()
    features = extract_landmarks(Path(image_path), detection_confidence)

    expected_features = getattr(model, "n_features_in_", features.size)
    if features.size != expected_features:
        raise ValueError(
            f"Feature length mismatch. Model expects {expected_features} values, got {features.size}."
        )

    features_batch = features.reshape(1, -1)
    encoded_prediction = model.predict(features_batch)[0]
    predicted_label = label_encoder.inverse_transform([encoded_prediction])[0]
    return str(predicted_label)
