import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from src.config import LABELS_PATH, MODEL_PATH
from src.dataset import collect_audio_files
from src.features import file_to_feature


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the saved speech emotion model.")
    parser.add_argument("--data-dir", default="data/raw", help="Folder containing downloaded datasets.")
    parser.add_argument("--limit-per-class", type=int, default=0, help="Use 0 to evaluate all files.")
    parser.add_argument(
        "--split",
        choices=["all", "balanced", "validation"],
        default="validation",
        help="Evaluate all files, a balanced subset, or the validation split used by train.py.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument("--report-only", action="store_true", help="Print only the classification report.")
    return parser.parse_args()


def main():
    args = parse_args()
    labels = json.loads(Path(LABELS_PATH).read_text(encoding="utf-8"))
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    rows = collect_audio_files(args.data_dir)

    if args.split == "validation":
        y = np.asarray([label_to_id[emotion] for _, emotion in rows], dtype=np.int64)
        _, rows = train_test_split(
            rows,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
    elif args.split == "balanced" and args.limit_per_class > 0:
        counts = {label: 0 for label in labels}
        limited_rows = []
        for path, emotion in rows:
            if counts[emotion] >= args.limit_per_class:
                continue
            limited_rows.append((path, emotion))
            counts[emotion] += 1
        rows = limited_rows

    if not rows:
        raise RuntimeError(f"No supported audio files found in {args.data_dir}")

    model = load_model(MODEL_PATH)
    x = np.asarray([file_to_feature(path) for path, _ in rows], dtype=np.float32)
    y_true = np.asarray([label_to_id[emotion] for _, emotion in rows], dtype=np.int64)
    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)

    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    Path("models").mkdir(exist_ok=True)
    Path("models/evaluation_report.txt").write_text(report, encoding="utf-8")
    Path("models/evaluation_confusion_matrix.json").write_text(
        json.dumps({"labels": labels, "matrix": matrix}, indent=2),
        encoding="utf-8",
    )

    print(report)
    if not args.report_only:
        print("Confusion matrix")
        print(json.dumps({"labels": labels, "matrix": matrix}, indent=2))


if __name__ == "__main__":
    main()
