/* SafeNet AI — Frontend JS */
"use strict";

// ── Theme toggle ──────────────────────────────────────────
const root = document.documentElement;
const themeBtn = document.getElementById("themeBtn");
const stored = localStorage.getItem("safenet-theme") || "dark";
root.setAttribute("data-theme", stored);
if (themeBtn) {
  themeBtn.textContent = stored === "dark" ? "☀️" : "🌙";
  themeBtn.addEventListener("click", () => {
    const current = root.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("safenet-theme", next);
    themeBtn.textContent = next === "dark" ? "☀️" : "🌙";
  });
}

// ── Char counter ─────────────────────────────────────────
const inputText = document.getElementById("inputText");
const charCount  = document.getElementById("charCount");
if (inputText && charCount) {
  inputText.addEventListener("input", () => {
    charCount.textContent = inputText.value.length + " / 1000";
  });
}

// ── Example buttons ───────────────────────────────────────
document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    if (inputText) {
      inputText.value = btn.dataset.text || "";
      inputText.dispatchEvent(new Event("input"));
    }
  });
});

// ── Analyse button ────────────────────────────────────────
const analyseBtn  = document.getElementById("analyseBtn");
const resultCard  = document.getElementById("resultCard");
const spinner     = document.getElementById("spinner");

function showSpinner(show) {
  if (!spinner || !resultCard) return;
  spinner.classList.toggle("hidden", !show);
  resultCard.classList.add("hidden");
}

function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.min(pct, 100) + "%";
}

function renderResult(data) {
  if (data.error) { alert(data.error); return; }

  // Header badge
  const badge = document.getElementById("resultBadge");
  if (badge) {
    badge.textContent = data.emoji + " " + data.label;
    badge.className   = "result-badge badge-" + data.color;
  }

  const rText = document.getElementById("resultText");
  const rTime = document.getElementById("resultTime");
  if (rText) rText.textContent = '"' + data.text.substring(0, 80) + (data.text.length > 80 ? "…" : "") + '"';
  if (rTime) rTime.textContent = data.timestamp;

  // Bars
  setBar("confBar", data.confidence);
  setBar("sevBar",  data.severity);

  const confVal = document.getElementById("confValue");
  const sevVal  = document.getElementById("sevValue");
  if (confVal) confVal.textContent = data.confidence + "%";
  if (sevVal)  sevVal.textContent  = data.severity + " / 100";

  // Label probabilities
  const probsGrid = document.getElementById("probsGrid");
  if (probsGrid && data.label_probs) {
    probsGrid.innerHTML = "";
    const colorMap = {"Hate Speech": "danger", "Offensive": "warning", "Neutral": "success"};
    Object.entries(data.label_probs).forEach(([name, prob]) => {
      const pct  = Math.round(prob * 100);
      const col  = colorMap[name] || "info";
      const div  = document.createElement("div");
      div.style.cssText = "flex:1;min-width:120px;padding:.7rem;background:var(--bg);border-radius:8px;border:1px solid var(--border);text-align:center";
      div.innerHTML = (
        '<p style="font-size:.78rem;color:var(--text-muted);margin-bottom:.35rem">' + name + '</p>' +
        '<p style="font-size:1.3rem;font-weight:700;color:var(--' + col + ')">' + pct + '%</p>'
      );
      probsGrid.appendChild(div);
    });
  }

  // Top words
  const wordTags = document.getElementById("wordTags");
  if (wordTags) {
    wordTags.innerHTML = "";
    if (data.top_words && data.top_words.length > 0) {
      data.top_words.forEach(word => {
        const span = document.createElement("span");
        span.className   = "word-tag";
        span.textContent = word;
        wordTags.appendChild(span);
      });
      document.getElementById("explainSection").classList.remove("hidden");
    } else {
      document.getElementById("explainSection").classList.add("hidden");
    }
  }

  spinner.classList.add("hidden");
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

if (analyseBtn) {
  analyseBtn.addEventListener("click", async () => {
    const text = (inputText ? inputText.value : "").trim();
    if (!text) { alert("Please enter some text to analyse."); return; }

    showSpinner(true);
    try {
      const resp = await fetch("/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text }),
      });
      const data = await resp.json();
      renderResult(data);
    } catch (err) {
      showSpinner(false);
      alert("Error connecting to server: " + err.message);
    }
  });
}

// Allow Enter+Ctrl to submit
if (inputText) {
  inputText.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.ctrlKey && analyseBtn) analyseBtn.click();
  });
}
