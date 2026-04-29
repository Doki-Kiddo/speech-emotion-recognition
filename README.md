# Speech Emotion Recognition with CNN

Mini-project for detecting emotion from speech using a CNN trained on log-mel spectrograms. It includes:

- training pipeline for multiple public speech-emotion datasets
- Flask backend for audio prediction
- browser frontend that records microphone audio and displays emotion probabilities

## Recommended Datasets

Use at least two datasets for a stronger mini-project:

1. **RAVDESS**: balanced, clean acted speech/song emotion dataset. Good primary dataset.  
   https://zenodo.org/records/1188976
2. **CREMA-D**: larger, diverse speakers, strong secondary dataset.  
   https://github.com/CheyneyComputerScience/CREMA-D
3. **TESS**: optional, useful for improving female-speaker coverage, but it is very clean/acted.  
   https://doi.org/10.5683/SP2/E8H2MF
4. **SAVEE**: optional, smaller male-speaker dataset.  
   https://kahlan.eps.surrey.ac.uk/savee/

Supported emotion labels are normalized to:

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

RAVDESS `calm` is skipped by default because most other datasets do not share that label.

## Folder Layout

Put downloaded datasets inside `data/raw`:

```text
data/raw/
  RAVDESS/
  CREMA-D/
  TESS/
  SAVEE/
```

The parser is flexible and scans recursively, so nested actor folders are fine.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --data-dir data/raw --epochs 40 --batch-size 32
```

Training saves validation diagnostics to:

```text
models/classification_report.txt
models/confusion_matrix.json
models/training_history.json
```

You can optionally try spectrogram augmentation while training:

```bash
python train.py --data-dir data/raw --epochs 40 --batch-size 32 --augment
```

The trained model is saved to:

```text
models/emotion_cnn.keras
models/labels.json
```

## Evaluate

Evaluate the active saved model:

```bash
python evaluate.py --data-dir data/raw
```

For a faster balanced check:

```bash
python evaluate.py --data-dir data/raw --limit-per-class 80
```

Evaluation saves:

```text
models/evaluation_report.txt
models/evaluation_confusion_matrix.json
```

## Predict From Audio File

```bash
python predict.py path\to\sample.wav
```

## Run Web App

```bash
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

Click record, speak for 2-4 seconds, stop, then classify.

## Project Notes

This is a CNN-on-spectrogram project, not a raw waveform model. Audio is converted into a fixed-size log-mel spectrogram, normalized per sample, then treated like a grayscale image by the CNN.

For best results:

- combine RAVDESS and CREMA-D first
- add TESS/SAVEE only if accuracy is unstable
- keep train/validation split stratified
- check `models/evaluation_report.txt` for weak classes before trusting predictions
- record test samples around 3 seconds
- use a quiet room and a consistent microphone
- high-intensity happy speech can be confused with angry/fear; add more happy samples or retrain if this happens often
