/* ══════════════════════════════════════════════════════════════
   SCENE – Intel Image Classifier  |  Frontend JS
═══════════════════════════════════════════════════════════════ */

"use strict";

// ── DOM refs ─────────────────────────────────────────────────
const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("fileInput");
const dropzoneIdle   = document.getElementById("dropzoneIdle");
const dropzonePreview= document.getElementById("dropzonePreview");
const previewImg     = document.getElementById("previewImg");
const previewFilename= document.getElementById("previewFilename");
const removeBtn      = document.getElementById("removeBtn");
const classifyBtn    = document.getElementById("classifyBtn");
const btnSpinner     = document.getElementById("btnSpinner");
const resultsEmpty   = document.getElementById("resultsEmpty");
const resultsFilled  = document.getElementById("resultsFilled");
const resultsError   = document.getElementById("resultsError");
const resultEmoji    = document.getElementById("resultEmoji");
const resultClassName= document.getElementById("resultClassName");
const confBarFill    = document.getElementById("confBarFill");
const confValue      = document.getElementById("confValue");
const probBars       = document.getElementById("probBars");
const resultModelBadge = document.getElementById("resultModelBadge");

let selectedFile = null;

// ── File selection helpers ────────────────────────────────────

function showPreview(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewFilename.textContent = file.name;
  dropzoneIdle.style.display   = "none";
  dropzonePreview.style.display = "";
  classifyBtn.disabled = false;
  showEmpty();          // reset results on new image
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  if (previewImg.src) URL.revokeObjectURL(previewImg.src);
  previewImg.src = "";
  dropzonePreview.style.display = "none";
  dropzoneIdle.style.display    = "";
  classifyBtn.disabled = true;
  showEmpty();
}

// ── File input ────────────────────────────────────────────────
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) showPreview(e.target.files[0]);
});

removeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearFile();
});

// ── Drag-and-drop ─────────────────────────────────────────────
dropzone.addEventListener("click", (e) => {
  if (e.target === dropzone || e.target === dropzoneIdle ||
      e.target.closest("#dropzoneIdle")) {
    if (!selectedFile) fileInput.click();
  }
});

["dragenter", "dragover"].forEach(ev =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach(ev =>
  dropzone.addEventListener(ev, () => dropzone.classList.remove("drag-over"))
);
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) showPreview(file);
});

// ── Result panel state helpers ────────────────────────────────

function showEmpty() {
  resultsFilled.style.display  = "none";
  resultsError.style.display   = "none";
  resultsEmpty.style.display   = "";
}

function showError(msg) {
  resultsFilled.style.display  = "none";
  resultsEmpty.style.display   = "none";
  resultsError.style.display   = "";
  document.getElementById("errorMessage").textContent = msg;
}

window.resetError = function () { showEmpty(); };

function showResult(data) {
  resultsEmpty.style.display   = "none";
  resultsError.style.display   = "none";
  resultsFilled.style.display  = "";

  // Emoji + class name
  resultEmoji.textContent     = data.emoji || "🖼️";
  resultClassName.textContent = data.predicted_class;

  // Model badge
  resultModelBadge.textContent = data.model_used === "pytorch"
    ? "PyTorch  .pth"
    : "TensorFlow  .keras";

  // Confidence bar (animated)
  confValue.textContent = data.confidence.toFixed(1) + "%";
  requestAnimationFrame(() => {
    confBarFill.style.width = data.confidence + "%";
  });

  // Probability bars (sorted descending)
  const probs = data.probabilities;
  const sorted = Object.entries(probs).sort(([,a],[,b]) => b - a);
  const max    = sorted[0][1];

  probBars.innerHTML = "";
  sorted.forEach(([cls, pct], idx) => {
    const isTop = idx === 0;
    const isMid = pct >= 5;
    const tier  = isTop ? "top" : isMid ? "mid" : "low";

    const row  = document.createElement("div");
    row.className = "prob-row";
    row.style.animationDelay = (idx * 0.07) + "s";

    const label = document.createElement("span");
    label.className = "prob-label";
    label.textContent = cls;

    const track = document.createElement("div");
    track.className = "prob-track";

    const fill  = document.createElement("div");
    fill.className = `prob-fill ${tier}`;
    track.appendChild(fill);

    const pctEl = document.createElement("span");
    pctEl.className = `prob-pct ${tier}`;
    pctEl.textContent = pct.toFixed(1) + "%";

    row.append(label, track, pctEl);
    probBars.appendChild(row);

    // Animate bar width after paint
    setTimeout(() => {
      fill.style.width = pct + "%";
    }, 80 + idx * 70);
  });
}

// ── Classify ──────────────────────────────────────────────────

classifyBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const model = document.querySelector('input[name="model"]:checked').value;

  // Loading state
  classifyBtn.classList.add("loading");
  classifyBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("model", model);

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || `Server error (${res.status})`);
    } else {
      showResult(data);
    }
  } catch (err) {
    showError("Network error – is the server running?");
    console.error(err);
  } finally {
    classifyBtn.classList.remove("loading");
    classifyBtn.disabled = false;
  }
});

// ── Keyboard shortcut: Enter to classify ─────────────────────
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && selectedFile && !classifyBtn.disabled) {
    classifyBtn.click();
  }
});
