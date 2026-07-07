const MIN_DUR = 30;
const MAX_DUR = 45;

const el = (id) => document.getElementById(id);
const fileInput = el("file");
const drop = el("drop");
const analyzeBtn = el("analyze");
const consent = el("consent");
const errorEl = el("error");

let selectedFile = null;
let selectedDuration = null;

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.toggle("hidden", !msg);
}

function updateAnalyzeState() {
  const durOk = selectedDuration !== null && selectedDuration >= MIN_DUR && selectedDuration <= MAX_DUR;
  analyzeBtn.disabled = !(selectedFile && durOk && consent.checked);
}

async function readDuration(file) {
  const url = URL.createObjectURL(file);
  try {
    const buf = await file.arrayBuffer();
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    const decoded = await ctx.decodeAudioData(buf.slice(0));
    ctx.close();
    return decoded.duration;
  } catch (e) {
    // Fallback: use the <audio> element metadata.
    return await new Promise((resolve, reject) => {
      const a = new Audio();
      a.preload = "metadata";
      a.onloadedmetadata = () => resolve(a.duration);
      a.onerror = () => reject(new Error("decode failed"));
      a.src = url;
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function handleFile(file) {
  showError("");
  selectedFile = file;
  selectedDuration = null;
  el("filename").textContent = file.name;
  el("player").src = URL.createObjectURL(file);
  el("filemeta").classList.remove("hidden");

  const durBadge = el("duration");
  durBadge.textContent = "reading…";
  durBadge.className = "badge";
  try {
    const dur = await readDuration(file);
    selectedDuration = dur;
    durBadge.textContent = `${dur.toFixed(1)}s`;
    if (dur < MIN_DUR || dur > MAX_DUR) {
      durBadge.classList.add("bad");
      showError(`Clip must be ${MIN_DUR}–${MAX_DUR}s. This one is ${dur.toFixed(1)}s.`);
    } else {
      durBadge.classList.add("ok");
    }
  } catch {
    durBadge.textContent = "unknown length";
    showError("Could not read this file's duration. Try a WAV or MP3.");
  }
  updateAnalyzeState();
}

el("browse").addEventListener("click", () => fileInput.click());
drop.addEventListener("click", (e) => { if (e.target === drop || e.target.classList.contains("drop-title") || e.target.classList.contains("drop-hint")) fileInput.click(); });
fileInput.addEventListener("change", (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });
consent.addEventListener("change", updateAnalyzeState);

["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); })
);
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  showError("");
  el("uploader").classList.add("hidden");
  el("results").classList.add("hidden");
  el("loading").classList.remove("hidden");

  const fd = new FormData();
  fd.append("audio", selectedFile);
  try {
    const resp = await fetch("/api/assess", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Assessment failed.");
    renderResults(data);
  } catch (e) {
    el("uploader").classList.remove("hidden");
    showError(e.message);
  } finally {
    el("loading").classList.add("hidden");
  }
});

el("again").addEventListener("click", () => {
  el("results").classList.add("hidden");
  el("uploader").classList.remove("hidden");
  el("wordcard").classList.add("hidden");
});

function colorFor(score) {
  if (score >= 80) return "var(--good)";
  if (score >= 55) return "var(--warn)";
  return "var(--bad)";
}

function renderResults(data) {
  el("results").classList.remove("hidden");
  el("overall").textContent = data.overall_score;
  const ring = el("ring");
  ring.style.setProperty("--val", data.overall_score);
  ring.style.setProperty("--col", colorFor(data.overall_score));
  el("fluency").textContent = data.fluency_score;
  el("c-good").textContent = data.summary.good;
  el("c-bad").textContent = data.summary.mispronounced;
  el("c-unclear").textContent = data.summary.unclear;

  const t = el("transcript");
  t.innerHTML = "";
  data.words.forEach((w, i) => {
    const span = document.createElement("span");
    span.className = `tok ${w.label}`;
    span.textContent = w.word + " ";
    if (w.label !== "good") {
      span.addEventListener("click", () => showWord(w, span));
    }
    t.appendChild(span);
  });
  el("wordcard").classList.add("hidden");
  el("results").scrollIntoView({ behavior: "smooth" });
}

function showWord(w, span) {
  document.querySelectorAll(".tok.active").forEach((s) => s.classList.remove("active"));
  span.classList.add("active");
  const card = el("wordcard");
  const exp = w.expected_phonemes.join(" ") || "—";
  const heard = w.heard_phonemes.join(" ") || "—";
  const issues = (w.issues || []).map((x) => `<li>${x}</li>`).join("");
  const labelText = w.label === "unclear" ? "Unclear segment" : "Mispronounced";
  card.innerHTML = `
    <h3>${w.word} <span class="badge">${labelText} · ${w.score.toFixed(0)}/100</span></h3>
    <p>Expected: <span class="ph exp">/${exp}/</span></p>
    <p>Heard: <span class="ph heard">/${heard}/</span></p>
    ${issues ? `<ul>${issues}</ul>` : ""}`;
  card.classList.remove("hidden");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
