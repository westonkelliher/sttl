/* sttl frontend: SSE-driven progressive UI. */
const $ = (id) => document.getElementById(id);
const QUICK = new URLSearchParams(location.search).has("quick");
const state = { status: "idle", segments: [], windows: [], unified: [], elapsed: 0, calib: {},
                cursor: 0, predictions: {} };
let levelHist = [];
let timerBase = 0, timerAnchor = null;

// ---------------- api

async function api(path, body) {
  const r = await fetch("/api/" + path, body !== undefined
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {});
  return r.json();
}

// ---------------- rendering

function fmt(sec) {
  sec = Math.max(0, Math.floor(sec));
  return String(Math.floor(sec / 60)).padStart(2, "0") + ":" + String(sec % 60).padStart(2, "0");
}

function setStatus(s) {
  state.status = s;
  const pill = $("statusPill");
  pill.className = "status-pill " + s;
  $("statusText").textContent = s;
  const rec = s === "recording";
  $("btnRecord").classList.toggle("active", rec);
  $("recLabel").textContent = rec ? "Pause" : (s === "paused" ? "Resume" : "Start");
  $("btnRecord").disabled = s === "finalizing";
  $("btnStop").disabled = !(rec || s === "paused");
  $("unified").contentEditable = s === "done" ? "plaintext-only" : "false";
  if (rec) { timerBase = state.elapsed; timerAnchor = performance.now(); }
  else { timerBase = state.elapsed; timerAnchor = null; }
}

function renderTimer() {
  let e = timerBase;
  if (timerAnchor !== null) e += (performance.now() - timerAnchor) / 1000;
  $("timer").textContent = fmt(e);
  requestAnimationFrame(renderTimer);
}

function renderCalib() {
  const c = state.calib || {};
  const el = $("calib");
  if (!c.calibrated) { el.textContent = "uncalibrated"; el.className = "calib"; return; }
  el.textContent = c.low_signal ? "calibrated (low signal!)" : "calibrated";
  el.className = "calib " + (c.low_signal ? "low" : "ok");
  if (c.calibrated) $("meterNote").textContent = "";
}

let player = null, playingSeg = null;
function playSeg(i) {
  if (playingSeg === i && player) { player.pause(); player = null; playingSeg = null; return; }
  if (player) player.pause();
  player = new Audio(`/api/audio/${state.sid}/${i}`);
  playingSeg = i;
  player.onended = () => { playingSeg = null; };
  player.play().catch(() => toast("Audio not available", true));
}

function renderTimeline() {
  const tl = $("timeline");
  tl.innerHTML = "";
  for (const s of state.segments) {
    const d = document.createElement("div");
    d.className = "seg " + s.state;
    d.onclick = () => playSeg(s.i);
    d.title = `#${s.i}  ${fmt(s.t0)}–${fmt(s.t0 + s.dur)}  ${s.state}` +
      (s.kept != null ? `  speech kept: ${Math.round(s.kept * 100)}%` : "") +
      (s.vad ? `  vad: ${s.vad}` : "") +
      "  (click to play)";
    if (s.kept != null && s.state !== "empty") {
      const k = document.createElement("div");
      k.className = "kept"; k.style.width = Math.round(s.kept * 100) + "%";
      d.appendChild(k);
    }
    tl.appendChild(d);
  }
}

function renderUnified() {
  const u = $("unified");
  u.innerHTML = "";
  let has = false;
  for (const c of state.unified) {
    const d = document.createElement("div");
    d.className = "chunk" + (c.fallback ? " fallback" : "");
    d.title = c.fallback ? "LLM unify failed — raw Pass A text" : "";
    d.textContent = c.text;
    u.appendChild(d);
    has = true;
  }
  // tail: single-pass text past the unify cursor, then the live prediction
  const singles = [];
  for (let i = state.cursor; i < state.segments.length; i++) {
    const s = state.segments[i];
    if (s && s.passA) singles.push(s.passA);
  }
  const preds = Object.keys(state.predictions).map(Number).sort((a, b) => a - b)
    .filter(i => {
      const s = state.segments[i];
      return !(s && (s.passA != null || ["empty", "failed", "done"].includes(s.state)));
    })
    .map(i => state.predictions[i]);
  if (singles.length || preds.length) {
    const tail = document.createElement("div");
    tail.className = "chunk";
    if (singles.length) {
      const sp = document.createElement("span");
      sp.className = "u-single";
      sp.title = "one pass done — solidifies after the LLM merge";
      sp.textContent = singles.join(" ") + " ";
      tail.appendChild(sp);
    }
    if (preds.length && (state.status === "recording" || state.status === "paused")) {
      const pr = document.createElement("span");
      pr.className = "u-predict";
      pr.title = "live guess — will be re-transcribed";
      pr.textContent = preds.join(" ");
      tail.appendChild(pr);
    }
    if (tail.childNodes.length) { u.appendChild(tail); has = true; }
  }
  if (!has) {
    u.innerHTML = '<span class="placeholder">' +
      (state.status === "idle" ? "Press Start (or Space) to begin." :
        "Listening… words appear within a few seconds.") + "</span>";
    return;
  }
  u.scrollTop = u.scrollHeight;
}

function renderPasses() {
  const p = $("passes");
  p.innerHTML = "";
  for (const s of state.segments) {
    const row = document.createElement("div");
    row.className = "pass-row";
    const w = state.windows[s.i];  // window i starts at seg i midpoint
    const a = s.state === "empty" ? "· silence ·"
      : s.state === "failed" ? "· failed ·"
      : s.passA != null ? s.passA : "…";
    const b = w == null ? ""
      : w.state === "failed" ? "· failed ·"
      : (w.text != null ? (w.text || "· silence ·") : "…");
    row.innerHTML =
      `<span class="t">${fmt(s.t0)}</span>` +
      `<span class="a ${s.passA == null && s.state !== "empty" ? "pending" : ""}">${esc(a)}</span>` +
      `<span class="b ${w && w.text == null ? "pending" : ""}">${esc(b)}</span>`;
    p.appendChild(row);
  }
  p.scrollTop = p.scrollHeight;
}

function esc(t) { const d = document.createElement("span"); d.textContent = t ?? ""; return d.innerHTML; }

function renderAll() { renderTimeline(); renderUnified(); renderPasses(); renderCalib(); }

// ---------------- level meter

const meter = $("meter"), mctx = meter.getContext("2d");
const DPR = window.devicePixelRatio || 1;
meter.width = 360 * DPR; meter.height = 34 * DPR;
meter.style.width = "360px"; meter.style.height = "34px";
mctx.scale(DPR, DPR);
function drawMeter() {
  const css = getComputedStyle(document.documentElement);
  const W = 360, H = 34, n = 90, bw = W / n;
  mctx.clearRect(0, 0, W, H);
  const thresh = levelHist.length ? levelHist[levelHist.length - 1].threshold : 0;
  for (let k = 0; k < levelHist.length; k++) {
    const { rms } = levelHist[k];
    const h = Math.min(1, Math.log10(1 + rms) / 4.2) * (H - 4);
    mctx.fillStyle = css.getPropertyValue(rms > thresh ? "--level-bar" : "--level-quiet");
    mctx.fillRect(k * bw + 1, H - 2 - h, bw - 2, h);
  }
  if (thresh > 0) {
    const ty = H - 2 - Math.min(1, Math.log10(1 + thresh) / 4.2) * (H - 4);
    mctx.strokeStyle = css.getPropertyValue("--level-thresh");
    mctx.setLineDash([3, 3]);
    mctx.beginPath(); mctx.moveTo(0, ty); mctx.lineTo(W, ty); mctx.stroke();
    mctx.setLineDash([]);
  }
}

// ---------------- SSE

function connect() {
  const es = new EventSource("/api/events");
  es.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "hello") {
      loadState();                       // resync after any reconnect
    } else if (m.type === "level") {
      levelHist.push(m); if (levelHist.length > 90) levelHist.shift();
      state.elapsed = m.elapsed;
      timerBase = m.elapsed; timerAnchor = performance.now();
      drawMeter();
    } else if (m.type === "segment") {
      state.segments[m.segment.i] = m.segment;
      const s = m.segment;
      if (s.passA != null || ["empty", "failed"].includes(s.state)) delete state.predictions[s.i];
      renderTimeline(); renderPasses(); renderUnified();
    } else if (m.type === "window") {
      state.windows[m.window.i] = m.window;
      renderPasses();
    } else if (m.type === "unified") {
      state.unified.push(m.chunk);
      if (m.cursor != null) state.cursor = Math.max(state.cursor, m.cursor);
      renderUnified();
    } else if (m.type === "predict") {
      state.predictions[m.i] = m.text;
      renderUnified();
    } else if (m.type === "calib") {
      state.calib = m; renderCalib();
    } else if (m.type === "status") {
      if (m.status === "recording" && m.session) resetSession();
      setStatus(m.status);
      if (m.status === "done") {
        state.predictions = {};
        renderUnified();
        toast("Done — transcript copied to clipboard");
        if (QUICK) setTimeout(() => window.close(), 1500);
      }
      if (m.status === "error") toast("Error: " + m.error, true);
    } else if (m.type === "raw_ready") {
      toast("Raw transcript copied — refining…");
    } else if (m.type === "warn") {
      toast(m.msg, true);
    }
  };
  es.onerror = () => { setTimeout(() => { es.close(); connect(); }, 1500); };
}

function resetSession() {
  state.segments = []; state.windows = []; state.unified = []; state.calib = {};
  state.cursor = 0; state.predictions = {};
  state.elapsed = 0; timerBase = 0; levelHist = [];
  $("meterNote").textContent = "speak normally for the first few seconds — it calibrates the silence detector";
  renderAll();
}

async function loadState() {
  const s = await api("state");
  if (s.status && s.status !== "idle") {
    Object.assign(state, {
      segments: s.segments || [], windows: s.windows || [], unified: s.unified || [],
      calib: s.calib || {}, elapsed: s.elapsed || 0, sid: s.id,
      cursor: s.unify_cursor || 0,
      predictions: s.prediction ? { [s.prediction.i]: s.prediction.text } : {},
    });
    timerBase = state.elapsed;
    setStatus(s.status);
    renderAll();
  }
}

// ---------------- actions

function unifiedText() {
  // after finalize the pane is editable — respect the user's fixes
  if (state.status === "done" && state.unified.length) {
    const t = $("unified").innerText.trim();
    if (t) return t;
  }
  return state.unified.map(c => c.text).join("\n\n").trim();
}

async function toggleRecord() {
  if (state.status === "recording" || state.status === "paused") await api("pause", {});
  else if (state.status !== "finalizing") {
    const r = await api("start", {});
    if (r.error) toast(r.error, true);
    else if (r.id) state.sid = r.id;
  }
}

async function stopRecord() {
  if (state.status === "recording" || state.status === "paused") await api("stop", {});
}

async function copyUnified() {
  const t = unifiedText();
  if (!t) return toast("Nothing to copy yet", true);
  try { await navigator.clipboard.writeText(t); } catch (e) { await api("copy", { text: t }); }
  toast("Copied");
}

function download() {
  const t = unifiedText();
  if (!t) return toast("Nothing to download yet", true);
  const blob = new Blob(["# sttl transcript\n\n" + t + "\n"], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "sttl-" + new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-") + ".md";
  a.click();
}

async function showHistory() {
  const list = await api("sessions");
  const el = $("historyList");
  el.innerHTML = list.length ? "" : '<span class="dim">No sessions yet.</span>';
  for (const s of list) {
    const d = document.createElement("div");
    d.className = "history-item";
    d.innerHTML = `<span class="id">${esc(s.id)}</span>` +
      `<span class="meta">${fmt(s.elapsed || 0)} · ${s.segments} seg · ${esc(s.status)}</span>` +
      `<span class="preview">${esc(s.preview || "")}</span>`;
    d.onclick = async () => {
      if (["recording", "paused", "finalizing"].includes(state.status))
        return toast("Finish the current session first", true);
      const full = await api("session/" + s.id);
      state.unified = full.unified || [];
      state.segments = full.segments || [];
      state.windows = full.windows || [];
      state.sid = full.id;
      state.cursor = full.unify_cursor ?? state.segments.length;
      state.predictions = {};
      renderAll();
      $("historyOverlay").classList.remove("open");
      toast("Loaded " + s.id);
    };
    el.appendChild(d);
  }
  $("historyOverlay").classList.add("open");
}

let toastTimer = null;
function toast(msg, warn) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show" + (warn ? " warn" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}

// ---------------- wiring

$("btnRecord").onclick = toggleRecord;
$("btnStop").onclick = stopRecord;
$("btnCopy").onclick = copyUnified;
$("btnDownload").onclick = download;
$("btnHelp").onclick = () => $("helpOverlay").classList.toggle("open");
$("btnHistory").onclick = showHistory;
$("passesToggle").onclick = () => {
  const p = $("passesPane");
  p.classList.toggle("collapsed");
  $("passesCaret").textContent = p.classList.contains("collapsed") ? "▸" : "▾";
};
for (const ov of ["helpOverlay", "historyOverlay"])
  $(ov).onclick = (e) => { if (e.target === $(ov)) $(ov).classList.remove("open"); };

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.isContentEditable || e.metaKey || e.ctrlKey) return;
  if (e.key === " ") { e.preventDefault(); toggleRecord(); }
  else if (e.key === "Enter") stopRecord();
  else if (e.key === "s" || e.key === "S") stopRecord();
  else if (e.key === "c" || e.key === "C") copyUnified();
  else if (e.key === "d" || e.key === "D") download();
  else if (e.key === "p" || e.key === "P") $("passesToggle").click();
  else if (e.key === "h" || e.key === "H") showHistory();
  else if (e.key === "?") $("helpOverlay").classList.toggle("open");
  else if (e.key === "Escape")
    for (const ov of ["helpOverlay", "historyOverlay"]) $(ov).classList.remove("open");
});

window.addEventListener("beforeunload", (e) => {
  if (state.status === "recording" || state.status === "paused") e.preventDefault();
});

setStatus("idle");
renderAll();
renderTimer();
connect();
loadState().then(() => {
  // quick mode (?quick=1): open → already recording; Enter/S stops; window closes when done
  if (QUICK && state.status === "idle") toggleRecord();
});
