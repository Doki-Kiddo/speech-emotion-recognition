SAMPLE_RATE = 22050
DURATION_SECONDS = 3.0
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
MAX_FRAMES = 130

MODEL_PATH = "models/emotion_cnn.keras"
LABELS_PATH = "models/labels.json"

SUPPORTED_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
