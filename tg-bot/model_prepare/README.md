# Face Landmark Table Preparation

Use `extract_face_landmarks.py` to scan every image inside `dataset/`, detect
MediaPipe Face Mesh landmarks, and export a CSV table for later classification.

## Quick start (with Mamba)

```bash
cd tg-bot
cd model_prepare
python extract_face_landmarks.py --dataset-dir dataset --output face_landmarks.csv
```

Arguments:

- `--dataset-dir` – root folder that contains one subfolder per class.
- `--output` – path to the CSV file that will contain the flattened landmark
  coordinates (`file_name`, `class_name`, `x_i`, `y_i`, `z_i` for 468 points).
- `--detection-confidence` – tweak MediaPipe detection confidence threshold.

Always activate your `moodbot` environment (`mamba activate moodbot`) before installing dependencies or running scripts. To process HEIC/HEIF photos, ensure `pillow-heif` is installed (already listed in `requirements.txt`).

