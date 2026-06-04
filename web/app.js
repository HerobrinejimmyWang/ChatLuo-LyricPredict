const fileInput = document.querySelector("#fileInput");
const contextInput = document.querySelector("#contextInput");
const outputText = document.querySelector("#outputText");
const statusText = document.querySelector("#statusText");
const confidenceText = document.querySelector("#confidenceText");
const predictButton = document.querySelector("#predictButton");
const continueButton = document.querySelector("#continueButton");

function setStatus(text, lowConfidence = false) {
  statusText.textContent = text;
  statusText.classList.toggle("low-confidence", lowConfidence);
}

function appendAccepted(text) {
  const separator = contextInput.value && !contextInput.value.endsWith("\n") ? "\n" : "";
  contextInput.value = `${contextInput.value}${separator}${text}`;
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
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context, continue: true }),
    });
    if (!response.ok) {
      const error = await response.text();
      throw new Error(error);
    }
    const result = await response.json();
    confidenceText.textContent = Number(result.confidence || 0).toFixed(3);
    if (!result.accepted) {
      setStatus(`No output: ${result.reason}`, true);
      return;
    }
    appendAccepted(result.text);
    setStatus("Accepted");
  } finally {
    predictButton.disabled = false;
    continueButton.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  importLyrics().catch((error) => setStatus(error.message, true));
});

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
