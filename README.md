# Speech Emotion Recognition with CNN

Deep learning mini-project for detecting emotion from speech. Audio is converted to fixed-size log-mel spectrograms and classified by a CNN.

The project includes:

- training pipeline for speech emotion datasets
- saved CNN model for seven emotion classes
- Flask backend for prediction
- browser frontend with microphone recording and audio-file upload
- notebook workflow for setup, evaluation, optional training, and app launch

## Current Model

Active model:

```text
models/emotion_cnn.keras
models/labels.json
```

Supported emotion labels:

```text
angry, disgust, fear, happy, neutral, sad, surprise
```

Current validation-style evaluation:

```text
accuracy: 78.64%
macro F1-score: 79.38%
weighted F1-score: 77.96%
```

The faster balanced sanity check gives about `82.86%`, but the validation-style score is the better number to report.

Known weak classes:

```text
happy
fear
```

Happy can sometimes be confused with surprise or angry, especially on RAVDESS samples.

## Datasets

Put datasets inside `data/raw`:

```text
data/raw/
  RAVDESS/
  CREMA-D/
  TESS/
  SAVEE/
```

The current project has RAVDESS and CREMA-D. Adding TESS/SAVEE can improve coverage, especially for weak classes.

## Setup

```powershell
cd "C:\Users\ABHISHEK SARKAR\Documents\dlPBL\speech-emotion-cnn"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run From Notebook

Open:

```text
run_project.ipynb
```

Use this kernel:

```text
Python (.venv speech-emotion-cnn)
```

Notebook flow:

```text
1. Check project and model files
2. Optional training
3. Validation-style evaluation report
4. Start Flask web app on http://127.0.0.1:5050
```

For normal use, keep:

```python
TRAIN_MODEL = False
```

Set it to `True` only when intentionally retraining.

## Run Web App

PowerShell:

```powershell
cd "C:\Users\ABHISHEK SARKAR\Documents\dlPBL\speech-emotion-cnn"
.\.venv\Scripts\python.exe backend\app.py
```

Open:

```text
http://127.0.0.1:5000
```

Features:

- record microphone audio
- upload an audio file
- preview audio
- classify emotion
- view probability bars for every emotion

## Evaluate Model

Recommended validation-style evaluation:

```powershell
.\.venv\Scripts\python.exe evaluate.py --data-dir data/raw --split validation --test-size 0.2 --seed 42 --report-only
```

Quick balanced sanity check:

```powershell
.\.venv\Scripts\python.exe evaluate.py --data-dir data/raw --split balanced --limit-per-class 80 --report-only
```

Evaluation outputs are saved to:

```text
models/evaluation_report.txt
models/evaluation_confusion_matrix.json
```

## Train Model

```powershell
.\.venv\Scripts\python.exe train.py --data-dir data/raw --epochs 40 --batch-size 32
```

Training uses:

- stratified validation split
- class weights
- model checkpointing
- early stopping
- learning-rate reduction

Optional spectrogram augmentation:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir data/raw --epochs 40 --batch-size 32 --augment
```

Training outputs:

```text
models/emotion_cnn.keras
models/labels.json
models/classification_report.txt
models/confusion_matrix.json
models/training_history.json
```

## Predict One File

```powershell
.\.venv\Scripts\python.exe predict.py "path\to\sample.wav"
```

## Project Notes

This is a CNN-on-spectrogram project, not a raw-waveform model.

Pipeline:

```text
audio -> mono 22050 Hz -> 3 seconds -> log-mel spectrogram -> normalization -> CNN -> softmax emotion probabilities
```

Ways to improve accuracy:

- add TESS/SAVEE or more custom recordings
- use speaker-independent train/test splitting
- try CNN + BiLSTM + attention
- try pretrained Wav2Vec2/HuBERT/WavLM features
- collect more happy and fear samples

## Troubleshooting

If Jupyter shows a Keras `BatchNormalization` error, the notebook is probably using global Python instead of `.venv`. Change kernel to:

```text
Python (.venv speech-emotion-cnn)
```

If PowerShell cannot find `models/...`, first move into the project folder:

```powershell
cd "C:\Users\ABHISHEK SARKAR\Documents\dlPBL\speech-emotion-cnn"
```

If the frontend does not update, hard refresh:

```text
Ctrl + F5
```
