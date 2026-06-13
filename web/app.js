const fileInput = document.querySelector("#fileInput");
const contextInput = document.querySelector("#contextInput");
const outputText = document.querySelector("#outputText");
const statusText = document.querySelector("#statusText");
const confidenceText = document.querySelector("#confidenceText");
const modeText = document.querySelector("#modeText");
const correctionText = document.querySelector("#correctionText");
const predictButton = document.querySelector("#predictButton");
const continueButton = document.querySelector("#continueButton");
const modeInputs = Array.from(document.querySelectorAll("input[name='mode']"));
const strictnessInputs = Array.from(document.querySelectorAll("input[name='strictness']"));
const correctionInput = document.querySelector("#correctionInput");

const MODE_LABELS = {
  matching: "Matching",
};

const STRICTNESS_LABELS = {
  strict: "Strict",
  balanced: "Balanced",
  tolerant: "Tolerant",
};

function getMode() {
  return "matching";
}

function setMode(mode) {
  const nextMode = MODE_LABELS[mode] ? mode : "matching";
  modeInputs.forEach((input) => {
    input.checked = input.value === nextMode;
  });
  updateReadout();
  localStorage.setItem("lyricpredict.mode", nextMode);
}

function getStrictness() {
  return strictnessInputs.find((input) => input.checked)?.value || "balanced";
}

function setStrictness(strictness) {
  const nextStrictness = STRICTNESS_LABELS[strictness] ? strictness : "balanced";
  strictnessInputs.forEach((input) => {
    input.checked = input.value === nextStrictness;
  });
  updateReadout();
  localStorage.setItem("lyricpredict.strictness", nextStrictness);
}

function setCorrection(enabled) {
  correctionInput.checked = Boolean(enabled);
  localStorage.setItem("lyricpredict.correction", correctionInput.checked ? "true" : "false");
}

function updateReadout() {
  modeText.textContent = `${MODE_LABELS[getMode()]} / ${STRICTNESS_LABELS[getStrictness()]}`;
}

function setStatus(text, lowConfidence = false) {
  statusText.textContent = text;
  statusText.classList.toggle("low-confidence", lowConfidence);
}

function displayReason(reason) {
  if (reason === "retrieval") return "Matched";
  if (reason && reason.startsWith("char_match_half")) return "Partial matched";
  if (["char_match_suffix", "char_match_prefix", "char_match_overlap"].includes(reason)) return "Matched";
  if (reason === "char_match_ambiguous") return "Ambiguous match";
  if (reason === "char_match_threshold") return "Low confidence";
  if (reason === "char_match_no_candidate") return "No match";
  if (reason && reason.startsWith("verified_transformer:ngram_exact")) return "Verified";
  if (reason && reason.startsWith("verified_transformer:ngram_fuzzy")) return "Verified with correction";
  if (reason === "low_final_confidence") return "Low confidence";
  if (reason === "no_transformer_candidate") return "No usable output";
  if (reason === "no_model_match") return "No match";
  return String(reason || "No output").replaceAll("_", " ");
}

function normalizeBoundary(current, text) {
  let nextText = text.trim().replace(/([,.，。])\1+/g, "$1");
  if (/[,，.。]$/.test(current.trim()) && /^[,，.。]/.test(nextText)) {
    nextText = nextText.replace(/^[,，.。]+/, "").trim();
  }
  return nextText;
}

function appendAccepted(text) {
  const current = contextInput.value;
  const nextText = normalizeBoundary(current, text);
  const separator = current && !/[\n,，.。]$/.test(current) && !/^[,，.。]/.test(nextText) ? "\n" : "";
  contextInput.value = `${current}${separator}${nextText}`;
  outputText.textContent = contextInput.value;
}

async function importLyrics() {
  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    return;
  }
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  setStatus("Importing...");
  const response = await fetch("/api/import", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error);
  }
  const result = await response.json();
  const stats = result.stats || {};
  setStatus(`Imported ${stats.files || files.length} files, ${stats.lines || 0} lines`);
}

async function predictNext() {
  const context = contextInput.value.trim();
  if (!context) {
    setStatus("Context is empty", true);
    return;
  }
  setStatus("Predicting...");
  predictButton.disabled = true;
  continueButton.disabled = true;
  try {
    const mode = getMode();
    const strictness = getStrictness();
    const correction = correctionInput.checked;
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context, continue: true, mode, strictness, correction }),
    });
    if (!response.ok) {
      const error = await response.text();
      throw new Error(error);
    }
    const result = await response.json();
    confidenceText.textContent = Number(result.confidence || 0).toFixed(3);
    if (!result.accepted) {
      setStatus(`No output: ${displayReason(result.reason)}`, true);
      return;
    }
    if (result.corrected_context) {
      contextInput.value = result.corrected_context;
      correctionText.textContent = `Corrected context: ${result.corrected_context}`;
    } else {
      correctionText.textContent = "";
    }
    appendAccepted(result.text);
    setStatus(`Accepted: ${MODE_LABELS[mode]} / ${STRICTNESS_LABELS[getStrictness()]}`);
  } finally {
    predictButton.disabled = false;
    continueButton.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  importLyrics().catch((error) => setStatus(error.message, true));
});

modeInputs.forEach((input) => {
  input.addEventListener("change", () => setMode(input.value));
});

strictnessInputs.forEach((input) => {
  input.addEventListener("change", () => setStrictness(input.value));
});

correctionInput.addEventListener("change", () => setCorrection(correctionInput.checked));

predictButton.addEventListener("click", () => {
  outputText.textContent = contextInput.value;
  predictNext().catch((error) => setStatus(error.message, true));
});

continueButton.addEventListener("click", () => {
  predictNext().catch((error) => setStatus(error.message, true));
});

window.addEventListener("keydown", (event) => {
  if (event.key === "F8") {
    event.preventDefault();
    predictNext().catch((error) => setStatus(error.message, true));
  }
});

setMode(localStorage.getItem("lyricpredict.mode") || "auto");
setStrictness(localStorage.getItem("lyricpredict.strictness") || "balanced");
setCorrection(localStorage.getItem("lyricpredict.correction") === "true");
