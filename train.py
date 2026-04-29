import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import set_random_seed
import numpy as np

from src.config import LABELS_PATH, MODEL_PATH, SUPPORTED_EMOTIONS
from src.dataset import build_arrays
from src.model import build_cnn


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN speech emotion recognizer.")
    parser.add_argument("--data-dir", default="data/raw", help="Folder containing downloaded datasets.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", help="Enable spectrogram augmentation during training.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    labels = SUPPORTED_EMOTIONS
    x, y = build_arrays(args.data_dir, labels)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    Path("models").mkdir(exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    model = build_cnn(num_classes=len(labels), augment=args.augment)
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    model.save(MODEL_PATH)

    probabilities = model.predict(x_val, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    report = classification_report(y_val, y_pred, target_names=labels, digits=4)
    matrix = confusion_matrix(y_val, y_pred).tolist()

    with open("models/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)
    with open("models/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open("models/confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "matrix": matrix}, f, indent=2)

    print(report)
    print(f"Saved model to {MODEL_PATH}")
    print("Saved evaluation files to models/classification_report.txt and models/confusion_matrix.json")


if __name__ == "__main__":
    main()
