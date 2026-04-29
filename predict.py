import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from src.config import LABELS_PATH, MODEL_PATH
from src.features import file_to_feature


def predict_file(audio_path: str | Path) -> tuple[str, dict[str, float]]:
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    feature = file_to_feature(audio_path)
    probabilities = model.predict(np.expand_dims(feature, axis=0), verbose=0)[0]
    scores = {label: float(prob) for label, prob in zip(labels, probabilities)}
    return max(scores, key=scores.get), scores


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file.")
    parser.add_argument("audio")
    args = parser.parse_args()

    emotion, scores = predict_file(args.audio)
    print(f"Emotion: {emotion}")
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{label}: {score:.3f}")


if __name__ == "__main__":
    main()
