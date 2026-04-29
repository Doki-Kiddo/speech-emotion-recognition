const recordBtn = document.querySelector("#recordBtn");
const stopBtn = document.querySelector("#stopBtn");
const audioFile = document.querySelector("#audioFile");
const predictBtn = document.querySelector("#predictBtn");
const preview = document.querySelector("#preview");
const statusEl = document.querySelector("#status");
const emotionEl = document.querySelector("#emotion");
const scoresEl = document.querySelector("#scores");
const meterBar = document.querySelector("#meterBar");

let audioContext;
let source;
let processor;
let stream;
let chunks = [];
let recordedBlob;
let activeAudioBlob;
let activeAudioName = "speech.wav";
let previewUrl;
let analyser;
let animationId;
let peakLevel = 0;

async function checkHealth() {
  const res = await fetch("/api/health");
  const data = await res.json();
  statusEl.textContent = data.trained ? "Model ready" : "Train model first";
}

function encodeWav(buffers, sampleRate) {
  const length = buffers.reduce((sum, buffer) => sum + buffer.length, 0);
  const samples = new Float32Array(length);
  let offset = 0;
  for (const buffer of buffers) {
    samples.set(buffer, offset);
    offset += buffer.length;
  }

  const wav = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(wav);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let index = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(index, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    index += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

function drawMeter() {
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  const peak = data.reduce((max, value) => Math.max(max, Math.abs(value - 128)), 0);
  peakLevel = Math.max(peakLevel, peak);
  meterBar.style.height = `${Math.max(8, Math.min(100, peak * 1.6))}%`;
  animationId = requestAnimationFrame(drawMeter);
}

function setPreview(blob, name) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  activeAudioBlob = blob;
  activeAudioName = name || "speech.wav";
  previewUrl = URL.createObjectURL(blob);
  preview.src = previewUrl;
  predictBtn.disabled = false;
}

function resetResult() {
  emotionEl.textContent = "Waiting for speech";
  scoresEl.innerHTML = "";
}

async function startRecording() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  source = audioContext.createMediaStreamSource(stream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  analyser = audioContext.createAnalyser();
  peakLevel = 0;
  chunks = [];

  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };

  source.connect(analyser);
  source.connect(processor);
  processor.connect(audioContext.destination);
  drawMeter();

  recordedBlob = null;
  activeAudioBlob = null;
  audioFile.value = "";
  preview.removeAttribute("src");
  predictBtn.disabled = true;
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "Recording...";
  resetResult();
}

function stopRecording() {
  cancelAnimationFrame(animationId);
  meterBar.style.height = "8%";
  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((track) => track.stop());

  recordedBlob = encodeWav(chunks, audioContext.sampleRate);
  audioContext.close();
  if (peakLevel >= 3 && recordedBlob.size >= 1000) {
    setPreview(recordedBlob, "recorded-speech.wav");
  } else {
    activeAudioBlob = null;
    predictBtn.disabled = true;
  }
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = predictBtn.disabled ? "No voice detected" : "Audio captured";
}

function chooseUploadedFile() {
  const file = audioFile.files[0];
  if (!file) return;
  const supportedExtension = /\.(wav|mp3|flac|ogg|m4a|webm)$/i.test(file.name);
  if (file.type && !file.type.startsWith("audio/") && !supportedExtension) {
    statusEl.textContent = "Choose an audio file";
    audioFile.value = "";
    return;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  setPreview(file, file.name);
  meterBar.style.height = "8%";
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "Audio uploaded";
  resetResult();
}

async function classifyAudio() {
  if (!activeAudioBlob) return;
  const form = new FormData();
  form.append("audio", activeAudioBlob, activeAudioName);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90000);

  statusEl.textContent = "Classifying...";
  predictBtn.disabled = true;
  let res;
  let data;
  try {
    res = await fetch("/api/predict", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    const text = await res.text();
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text || `HTTP ${res.status}` };
    }
  } catch (error) {
    statusEl.textContent = error.name === "AbortError" ? "Prediction timed out" : "Prediction failed";
    emotionEl.textContent = error.message || "Try again";
    predictBtn.disabled = false;
    clearTimeout(timeout);
    return;
  }
  clearTimeout(timeout);
  predictBtn.disabled = false;

  if (!res.ok) {
    statusEl.textContent = "Model unavailable";
    emotionEl.textContent = data.error || "Prediction failed";
    return;
  }

  statusEl.textContent = "Prediction complete";
  emotionEl.textContent = data.emotion;
  renderScores(data.scores);
}

function renderScores(scores) {
  scoresEl.innerHTML = "";
  Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, score]) => {
      const row = document.createElement("div");
      row.className = "score";
      row.innerHTML = `
        <strong>${label}</strong>
        <div class="bar"><span style="width: ${Math.round(score * 100)}%"></span></div>
        <span>${Math.round(score * 100)}%</span>
      `;
      scoresEl.appendChild(row);
    });
}

recordBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
audioFile.addEventListener("change", chooseUploadedFile);
predictBtn.addEventListener("click", classifyAudio);

checkHealth().catch(() => {
  statusEl.textContent = "Backend offline";
});
