from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()  # Enable HEIC/HEIF decoding via Pillow.
except ImportError:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning(
        "pillow-heif is not installed; HEIC/HEIF files may fail to load."
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
NUM_LANDMARKS = 478


def iter_dataset_images(dataset_root: Path) -> Iterable[tuple[str, Path]]:
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            yield class_name, image_path


def extract_landmark_row(
    image_path: Path, class_name: str, face_mesh: mp.solutions.face_mesh.FaceMesh
) -> Optional[List[float]]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logging.warning("Failed to open %s: %s", image_path, exc)
        return None

    image_array = np.array(image)
    results = face_mesh.process(image_array)

    if not results.multi_face_landmarks:
        logging.warning("No face landmarks detected in %s", image_path)
        return None

    landmark_points = results.multi_face_landmarks[0].landmark
    if len(landmark_points) != NUM_LANDMARKS:
        logging.warning(
            "Unexpected landmark count (%d) in %s", len(landmark_points), image_path
        )

    row: List[float] = [image_path.name, class_name]
    for landmark in landmark_points:
        row.extend([landmark.x, landmark.y, landmark.z])

    return row


def build_column_headers() -> List[str]:
    headers = ["file_name", "class_name"]
    for idx in range(NUM_LANDMARKS):
        headers.extend([f"x_{idx}", f"y_{idx}", f"z_{idx}"])
    return headers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV table with MediaPipe Face Landmarks for every image in "
            "a dataset where each subfolder is treated as a class label."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).parent / "dataset",
        help="Root directory containing class subfolders with images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "face_landmarks.csv",
        help="Destination CSV file.",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe Face Mesh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    output_path: Path = args.output

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    rows: List[List[float]] = []
    headers = build_column_headers()

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.detection_confidence,
    ) as face_mesh:
        for class_name, image_path in iter_dataset_images(dataset_dir):
            logging.info("Processing %s", image_path)
            row = extract_landmark_row(image_path, class_name, face_mesh)
            if row is not None:
                rows.append(row)

    if not rows:
        logging.warning("No landmarks were extracted. CSV will not be created.")
        return

    df = pd.DataFrame(rows, columns=headers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Saved %d rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()

