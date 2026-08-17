const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");
const csvPreviewContainer = document.getElementById("csvPreviewContainer");
const csvName = document.getElementById("csvName");
const results = document.getElementById("results");
const statusEl = document.getElementById("status");

let thermalFile = null;
let acousticFile = null;

function fmt(n, digits = 4) {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 100) return n.toFixed(1);
  if (Math.abs(n) >= 1) return n.toFixed(3);
  return n.toFixed(digits);
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function updatePreviews() {
  if (thermalFile) {
    const url = URL.createObjectURL(thermalFile);
    preview.src = url;
    imagePreviewContainer.classList.remove("hidden");
  } else {
    imagePreviewContainer.classList.add("hidden");
  }
  
  if (acousticFile) {
    csvName.textContent = acousticFile.name;
    csvPreviewContainer.classList.remove("hidden");
  } else {
    csvPreviewContainer.classList.add("hidden");
  }

  if (thermalFile || acousticFile) {
    previewWrap.classList.remove("hidden");
    dropzone.classList.add("has-image");
  } else {
    previewWrap.classList.add("hidden");
    dropzone.classList.remove("has-image");
  }
}

function renderResult(data) {
  // Support both legacy single-file format and dual-upload format
  const isDual = data.prediction !== undefined;
  
  const quality = isDual ? data.prediction.label : data.quality;
  const qualityLabel = isDual ? data.prediction.label : data.quality_label;
  const conf = isDual ? data.prediction.confidence : data.confidence;
  
  // Try to map to CPB or healthy logic
  let isCPB = false;
  if (isDual) {
    isCPB = qualityLabel === "Cocoa Pod Borer";
  } else {
    isCPB = quality === "CPB";
  }
  
  // Set quality code based on label if not provided
  let qCode = isDual ? qualityLabel.charAt(0) : quality;
  if (qCode === "C") qCode = "CPB";
  if (qCode === "O") qCode = "OR";
  if (qCode === "U") qCode = "UR";
  
  document.getElementById("qualityCode").textContent = qCode;
  document.getElementById("qualityLabel").textContent = qualityLabel;
  document.getElementById("confidence").textContent = `${(conf * 100).toFixed(1)}%`;

  const isHealthy = !isCPB;
  const healthEl = document.getElementById("healthStatus");
  healthEl.textContent = isHealthy ? "Healthy" : "Infested";
  healthEl.style.color = isHealthy ? "var(--accent)" : "var(--danger)";
  healthEl.style.textShadow = isHealthy ? "0 0 15px var(--accent-glow)" : "0 0 15px rgba(255, 85, 85, 0.4)";

  // Features
  let amp, freq, pwr;
  if (isDual) {
    amp = data.features.amplitude;
    freq = data.features.frequency;
    pwr = data.features.power;
    document.getElementById("amplitude").textContent = fmt(amp);
    document.getElementById("amplitudeUnit").textContent = "RMS";
    document.getElementById("frequency").textContent = fmt(freq, 1);
    document.getElementById("frequencyUnit").textContent = "Hz";
    document.getElementById("power").textContent = fmt(pwr);
    document.getElementById("powerUnit").textContent = "mean-square";
  } else {
    document.getElementById("amplitude").textContent = fmt(data.amplitude.value);
    document.getElementById("amplitudeUnit").textContent = data.amplitude.unit || "RMS";
    document.getElementById("frequency").textContent = fmt(data.frequency.value, 1);
    document.getElementById("frequencyUnit").textContent = data.frequency.unit || "Hz";
    document.getElementById("power").textContent = fmt(data.power.value);
    document.getElementById("powerUnit").textContent = data.power.unit || "mean-square";
  }

  const probs = document.getElementById("probs");
  probs.innerHTML = "";
  
  const classProbs = isDual ? data.probabilities : Object.entries(data.class_probabilities || {}).map(([c, p]) => ({label: c, confidence: p}));
  
  classProbs.forEach((item) => {
    let cls = item.label || item[0];
    let p = item.confidence !== undefined ? item.confidence : item[1];
    const div = document.createElement("div");
    div.className = "prob";
    div.innerHTML = `<div><strong>${cls}</strong> ${(p * 100).toFixed(1)}%</div>
      <div class="bar"><span style="width:${(p * 100).toFixed(1)}%"></span></div>`;
    probs.appendChild(div);
  });

  results.classList.remove("hidden");
}

async function predict() {
  if (!thermalFile || !acousticFile) {
    setStatus("Please upload both an image and a CSV file.", true);
    return;
  }

  results.classList.add("hidden");
  dropzone.classList.add("loading");
  setStatus(`Analyzing...`);

  const body = new FormData();
  body.append("thermal_image", thermalFile);
  body.append("acoustic_file", acousticFile);

  try {
    const res = await fetch("/api/predict", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed");
    }
    renderResult(data);
    setStatus(`Done · Result ready`);
  } catch (err) {
    setStatus(err.message || String(err), true);
  } finally {
    dropzone.classList.remove("loading");
  }
}

function processFiles(files) {
  if (!files || files.length === 0) return;
  
  let newThermal = false;
  let newAcoustic = false;
  
  Array.from(files).forEach(file => {
    const isImage = file.type.startsWith("image/");
    const isCsv = file.name.toLowerCase().endsWith(".csv") || file.type.includes("csv") || file.name.toLowerCase().endsWith(".tdms");
    
    if (isImage) {
      thermalFile = file;
      newThermal = true;
    } else if (isCsv) {
      acousticFile = file;
      newAcoustic = true;
    }
  });
  
  updatePreviews();
  
  if (thermalFile && acousticFile && (newThermal || newAcoustic)) {
    predict();
  } else if (!thermalFile && acousticFile) {
    setStatus("Waveform added. Now add a thermal image.", false);
  } else if (thermalFile && !acousticFile) {
    setStatus("Image added. Now add an acoustic waveform.", false);
  }
}

browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => processFiles(fileInput.files));

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => processFiles(e.dataTransfer.files));
