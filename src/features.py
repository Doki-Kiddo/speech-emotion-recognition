from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.config import DURATION_SECONDS, HOP_LENGTH, MAX_FRAMES, N_FFT, N_MELS, SAMPLE_RATE


def _fit_length(audio: np.ndarray) -> np.ndarray:
    target_len = int(SAMPLE_RATE * DURATION_SECONDS)
    if len(audio) > target_len:
        return audio[:target_len]
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio


def load_audio(path: str | Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return _fit_length(audio)


def load_audio_bytes(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    return _fit_length(audio.astype(np.float32))


def audio_to_mel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] > MAX_FRAMES:
        log_mel = log_mel[:, :MAX_FRAMES]
    elif log_mel.shape[1] < MAX_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, MAX_FRAMES - log_mel.shape[1])))

    mean = log_mel.mean()
    std = log_mel.std() or 1.0
    normalized = (log_mel - mean) / std
    return normalized[..., np.newaxis].astype(np.float32)


def file_to_feature(path: str | Path) -> np.ndarray:
    return audio_to_mel(load_audio(path))
