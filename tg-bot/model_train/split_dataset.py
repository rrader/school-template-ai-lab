from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_TABLE = Path(__file__).resolve().parent.parent / "model_prepare" / "face_landmarks.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dataset CSV into train and test CSV files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TABLE,
        help="Path to input CSV file with dataset.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Path to output train CSV file (default: train_<input_filename>).",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=None,
        help="Path to output test CSV file (default: test_<input_filename>).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="class_name",
        help="Column that contains class labels for stratification (default: class_name).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Whether to shuffle data before splitting (default: True).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable shuffling before splitting.",
    )
    return parser.parse_args()


def split_dataset(
    csv_path: Path,
    train_output: Path,
    test_output: Path,
    test_size: float,
    target_column: str,
    random_state: int,
    shuffle: bool,
) -> None:
    """Split dataset CSV into train and test CSV files."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    logging.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV columns: {df.columns.tolist()}")

    logging.info("Dataset loaded: %d samples, %d columns", len(df), len(df.columns))
    logging.info("Target column: %s", target_column)

    # Check class distribution
    class_counts = df[target_column].value_counts()
    logging.info("Class distribution:\n%s", class_counts.to_string())

    # Determine if stratification is possible
    stratify = None
    unique_classes = df[target_column].unique()
    if len(unique_classes) > 1 and shuffle:
        # Use stratification if we have multiple classes
        stratify = df[target_column]
        logging.info("Using stratified split with %d classes", len(unique_classes))
    else:
        logging.info("Not using stratification")

    # Split the dataset
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    logging.info("Split completed:")
    logging.info("  Train set: %d samples (%.1f%%)", len(train_df), len(train_df) / len(df) * 100)
    logging.info("  Test set: %d samples (%.1f%%)", len(test_df), len(test_df) / len(df) * 100)

    # Show class distribution in splits
    if len(unique_classes) > 1:
        logging.info("Train set class distribution:\n%s", train_df[target_column].value_counts().to_string())
        logging.info("Test set class distribution:\n%s", test_df[target_column].value_counts().to_string())

    # Ensure output directories exist
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    # Save train and test sets
    train_df.to_csv(train_output, index=False)
    logging.info("Saved train set to %s", train_output)

    test_df.to_csv(test_output, index=False)
    logging.info("Saved test set to %s", test_output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    # Determine output file paths if not provided
    if args.train_output is None:
        input_name = args.input.stem
        input_parent = args.input.parent
        args.train_output = input_parent / f"train_{input_name}.csv"

    if args.test_output is None:
        input_name = args.input.stem
        input_parent = args.input.parent
        args.test_output = input_parent / f"test_{input_name}.csv"

    split_dataset(
        csv_path=args.input,
        train_output=args.train_output,
        test_output=args.test_output,
        test_size=args.test_size,
        target_column=args.target_column,
        random_state=args.random_state,
        shuffle=args.shuffle,
    )

    logging.info("Dataset splitting completed successfully!")


if __name__ == "__main__":
    main()

