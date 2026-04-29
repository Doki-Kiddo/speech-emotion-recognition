from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import SUPPORTED_EMOTIONS
from src.features import file_to_feature

RAVDESS_CODES = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

CREMAD_CODES = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

TESS_NAMES = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprise",
    "pleasant_surprise": "surprise",
}

SAVEE_CODES = {
    "a": "angry",
    "d": "disgust",
    "f": "fear",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprise",
}


def infer_emotion(path: Path) -> str | None:
    name = path.stem
    parts = name.replace("-", "_").split("_")

    if name.count("-") >= 6:
        ravdess_code = name.split("-")[2]
        return RAVDESS_CODES.get(ravdess_code)

    if len(parts) >= 3 and parts[2].upper() in CREMAD_CODES:
        return CREMAD_CODES[parts[2].upper()]

    lowered = name.lower()
    for token, emotion in TESS_NAMES.items():
        if lowered.endswith(f"_{token}") or f"_{token}_" in lowered:
            return emotion

    savee_token = lowered.split("_")[-1]
    savee_code = "".join(ch for ch in savee_token if ch.isalpha())
    if savee_code.startswith("sa"):
        return "sad"
    if savee_code.startswith("su"):
        return "surprise"
    if savee_code[:1] in SAVEE_CODES:
        return SAVEE_CODES[savee_code[:1]]

    return None


def collect_audio_files(data_dir: str | Path) -> list[tuple[Path, str]]:
    root = Path(data_dir)
    rows = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        emotion = infer_emotion(path)
        if emotion in SUPPORTED_EMOTIONS:
            rows.append((path, emotion))
    return rows


def build_arrays(data_dir: str | Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = collect_audio_files(data_dir)
    if not rows:
        raise RuntimeError(f"No supported audio files found in {data_dir}")

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    features = []
    targets = []

    for path, emotion in tqdm(rows, desc="Extracting mel spectrograms"):
        try:
            features.append(file_to_feature(path))
            targets.append(label_to_id[emotion])
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64)
