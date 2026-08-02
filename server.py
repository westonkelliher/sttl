#!/usr/bin/env python3
"""sttl server: long-form progressive dual-pass speech-to-text.

Pipeline:
  arecord (16k mono s16le) -> 20s segments (saved to disk immediately)
  -> energy VAD (calibrated on first segment, adaptive noise floor)
  -> silence-trimmed Pass A (window-aligned) + Pass B (half-offset windows)
  -> faster-whisper on both passes (never touches the newest segment)
  -> MiniMax unification of A+B into a final transcript, chunk by chunk.
"""
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

# ---------------------------------------------------------------- config

RATE = 16000
SEG_SECONDS = float(os.environ.get("STTL_SEG_SECONDS", "8"))
SEG_SAMPLES = int(RATE * SEG_SECONDS)
SEG_SAMPLES_SOFT = int(SEG_SAMPLES * 0.9)    # from here, cut at the first quiet moment
SEG_SAMPLES_HARD = int(SEG_SAMPLES * 1.25)   # never let a segment exceed this
GAP_SEC = 0.15                   # silence inserted between trimmed speech regions
FRAME = 480                      # 30 ms VAD frames
PAD_SEC = 1.0                    # safety buffer kept around speech
MIN_SPEECH_SEC = 0.30            # below this a segment counts as empty
MIN_FLUSH_SEC = 0.75             # discard partial buffers shorter than this
MODEL_NAME = os.environ.get("STTL_MODEL", "medium")       # pass A
BEAM = int(os.environ.get("STTL_BEAM", "5"))
MODEL_NAME_B = os.environ.get("STTL_MODEL_B", "small")     # pass B: different model+beam
BEAM_B = int(os.environ.get("STTL_BEAM_B", "8"))           # so the passes err differently
UNIFY_MIN = int(os.environ.get("STTL_UNIFY_MIN", "2"))   # segments per LLM call (min)
UNIFY_MAX = int(os.environ.get("STTL_UNIFY_MAX", "6"))
PORT = int(os.environ.get("STTL_PORT", "7737"))
DATA_DIR = Path(os.environ.get("STTL_DATA", os.path.expanduser("~/.local/share/sttl")))
FAKE_STT = os.environ.get("STTL_FAKE_STT") == "1"
FAKE_LLM = os.environ.get("STTL_FAKE_LLM") == "1"
INPUT_WAV = os.environ.get("STTL_INPUT_WAV")             # test: feed wav instead of mic
INPUT_SPEED = float(os.environ.get("STTL_SPEED", "1"))   # test: >1 feeds faster than realtime
VAD_MODE = os.environ.get("STTL_VAD", "auto")            # auto = Silero neural, else "energy"
PREDICT = os.environ.get("STTL_PREDICT", "1") == "1"     # live best-guess tail
PREDICT_EVERY = float(os.environ.get("STTL_PREDICT_EVERY", "2.5"))
KEY_FILE = os.path.expanduser("~/.keys/.minimax2.5_tool_caller")
MINIMAX_URL = "https://api.minimax.io/v1/chat/completions"

UNIFY_SYSTEM = """You are merging two speech-to-text transcripts of the SAME audio into one.
Pass A was transcribed in windows aligned at 0s; Pass B in windows offset by half a window,
using a different STT model, so the two passes make different mistakes. STT errors cluster
near window edges, so where the passes disagree, prefer the pass whose window midpoint is
closer to that moment (Pass B is more reliable near Pass A's window boundaries).
Pass B may extend slightly before/after the span you must output.

Rules:
- Output the single best transcript for the CURRENT SPAN ONLY, continuing naturally from the
  previous-context tail without repeating it.
- Include EVERY part of Pass A's span, in order. Long-form dictation contains pauses, topic
  shifts, and disconnected fragments — keep them all; never drop a sentence. Your output should
  contain roughly as many words as Pass A's text.
- Fix STT artifacts: words duplicated or truncated at window boundaries, misheard homophones,
  punctuation, capitalization, sentence boundaries.
- Remove obvious filler (um, uh) but never summarize, rearrange, or invent content.
- When uncertain, prefer Pass A's reading.
- Output ONLY plain transcript text. No labels, no markdown, no explanations."""

# ---------------------------------------------------------------- audio / vad


def frame_rms(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[: n * FRAME].astype(np.float32).reshape(n, FRAME)
    return np.sqrt((f * f).mean(axis=1))


class Calibration:
    """Adaptive energy threshold. Calibrated on the first segment (user is asked
    to speak during it), then the noise floor tracks quiet frames with an EMA."""

    def __init__(self):
        self.noise = None
        self.speech = None
        self.threshold = 120.0     # provisional until calibrated
        self.calibrated = False
        self.low_signal = False

    def calibrate(self, rms: np.ndarray):
        if len(rms) < 20:
            return
        noise = float(np.percentile(rms, 10))    # p10: robust to speech-heavy segments
        speech = float(np.percentile(rms, 90))
        self.low_signal = speech < max(noise * 2.0, 50.0)
        self.noise = max(noise, 1.0)
        self.speech = max(speech, self.noise * 2)
        self._recompute()
        self.calibrated = True

    def adapt(self, rms: np.ndarray):
        if not self.calibrated or len(rms) == 0:
            return
        quiet = rms[rms < self.threshold]
        loud = rms[rms >= self.threshold]
        if len(quiet) > 5:
            self.noise = 0.85 * self.noise + 0.15 * float(np.percentile(quiet, 50))
        if len(loud) > 5:
            self.speech = 0.85 * self.speech + 0.15 * float(np.percentile(loud, 80))
        self._recompute()

    def _recompute(self):
        # geometric mean of noise/speech, clamped near the noise floor: missing quiet
        # speech costs words, while a false positive only means a little less trimming
        thr = (self.noise * self.speech) ** 0.5
        thr = max(thr, 1.3 * self.noise + 5)
        thr = min(thr, 4.0 * self.noise)
        self.threshold = max(thr, 30.0)

    def snapshot(self):
        return {
            "calibrated": self.calibrated,
            "low_signal": self.low_signal,
            "noise": round(self.noise or 0, 1),
            "threshold": round(self.threshold, 1),
        }


def trim_silence(audio: np.ndarray, calib: Calibration):
    """Return (trimmed_audio, kept_ratio, speech_seconds). Keeps PAD_SEC around speech."""
    rms = frame_rms(audio)
    if len(rms) == 0:
        return audio[:0], 0.0, 0.0
    mask = rms > calib.threshold
    speech_sec = float(mask.sum()) * FRAME / RATE
    if speech_sec < MIN_SPEECH_SEC:
        return audio[:0], 0.0, speech_sec
    pad = int(PAD_SEC * RATE / FRAME) + 1
    idx = np.flatnonzero(mask)
    keep = np.zeros(len(rms), dtype=bool)
    for i in idx:
        keep[max(0, i - pad): i + pad + 1] = True
    # collect kept regions
    parts = []
    start = None
    for i, k in enumerate(keep):
        if k and start is None:
            start = i
        elif not k and start is not None:
            parts.append(audio[start * FRAME: i * FRAME])
            start = None
    if start is not None:
        parts.append(audio[start * FRAME:])
    if parts:
        # keep a short gap between regions so whisper doesn't merge distant words
        gap = np.zeros(int(GAP_SEC * RATE), dtype=np.int16)
        joined = [parts[0]]
        for p in parts[1:]:
            joined += [gap, p]
        trimmed = np.concatenate(joined)
    else:
        trimmed = audio[:0]
    return trimmed, len(trimmed) / max(len(audio), 1), speech_sec


def detect_speech_spans(audio: np.ndarray):
    """Silero neural VAD (bundled with faster-whisper): distinguishes meaningful speech
    from mere noise (fans, keyboards, HVAC). Returns sample spans, or None to fall back
    to the energy VAD."""
    if VAD_MODE == "energy":
        return None
    try:
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        opts = VadOptions(threshold=0.35, min_speech_duration_ms=150,
                          min_silence_duration_ms=250, speech_pad_ms=0)
        spans = get_speech_timestamps(audio.astype(np.float32) / 32768.0, opts)
        return [(d["start"], d["end"]) for d in spans]
    except Exception:
        return None


def trim_speech(audio: np.ndarray, calib: Calibration):
    """Return (trimmed, kept_ratio, speech_sec, method). Neural VAD when available."""
    spans = detect_speech_spans(audio)
    if spans is None:
        t, k, s = trim_silence(audio, calib)
        return t, k, s, "energy"
    speech_sec = sum(e - s for s, e in spans) / RATE
    if speech_sec < MIN_SPEECH_SEC:
        return audio[:0], 0.0, speech_sec, "neural"
    pad = int(PAD_SEC * RATE)
    merged = []
    for s, e in spans:
        s, e = max(0, s - pad), min(len(audio), e + pad)
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    gap = np.zeros(int(GAP_SEC * RATE), dtype=np.int16)
    parts = []
    for j, (s, e) in enumerate(merged):
        if j:
            parts.append(gap)
        parts.append(audio[s:e])
    trimmed = np.concatenate(parts)
    return trimmed, len(trimmed) / max(len(audio), 1), speech_sec, "neural"


def write_wav(path: Path, audio: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())


# ---------------------------------------------------------------- transcriber


class Transcriber:
    """Holds both whisper models (pass A and pass B), loaded lazily, shared by sessions."""

    def __init__(self):
        self.models = {}
        self._lock = threading.Lock()
        self._fail_n = int(os.environ.get("STTL_FAIL_N", "0"))   # test hook: fail first N calls

    def _get(self, name: str):
        with self._lock:
            m = self.models.get(name)
            if m is None:
                from faster_whisper import WhisperModel
                device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
                ct = "float16" if device == "cuda" else "int8"
                try:
                    m = WhisperModel(name, device=device, compute_type=ct)
                except Exception:
                    m = WhisperModel(name, device="cpu", compute_type="int8")
                self.models[name] = m
            return m

    def warm(self):
        if FAKE_STT:
            return
        self._get(MODEL_NAME)
        self._get(MODEL_NAME_B)

    def transcribe(self, audio: np.ndarray, model: str = MODEL_NAME, beam: int = BEAM) -> str:
        if len(audio) < RATE // 4:
            return ""
        if self._fail_n > 0:
            self._fail_n -= 1
            raise RuntimeError("injected test failure")
        if FAKE_STT:
            return f"<fake {model} {len(audio)/RATE:.1f}s>"
        m = self._get(model)
        try:
            segs, _ = m.transcribe(
                audio.astype(np.float32) / 32768.0,
                beam_size=beam, best_of=1,
                condition_on_previous_text=False, language="en",
            )
            return " ".join(s.text.strip() for s in segs).strip()
        except RuntimeError as e:
            if "cuda" in str(e).lower() or "cublas" in str(e).lower():
                from faster_whisper import WhisperModel
                with self._lock:
                    self.models[model] = WhisperModel(model, device="cpu", compute_type="int8")
                return self.transcribe(audio, model, beam)
            raise


TRANSCRIBER = Transcriber()   # shared across sessions; model loads once


def call_minimax(system: str, user: str) -> str:
    if FAKE_LLM:
        return "<unified> " + re.sub(r"\s+", " ", user)[-200:]
    with open(KEY_FILE) as f:
        key = f.read().strip()
    body = json.dumps({
        "model": "MiniMax-M2.5",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": 4096,
    }).encode()
    req = urllib.request.Request(
        MINIMAX_URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()


# ---------------------------------------------------------------- session


def fmt_ts(sec: float) -> str:
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


class Session:
    """One recording session. Owns the recorder, job worker, and unify worker."""

    def __init__(self, broadcast):
        self.id = time.strftime("%Y%m%d_%H%M%S")
        self.dir = DATA_DIR / "sessions" / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.broadcast = broadcast
        self.lock = threading.RLock()
        self.status = "recording"          # recording|paused|finalizing|done|error
        self.error = None
        self.calib = Calibration()
        self.segments = []                 # dicts, see _close_segment
        self.raw = {}                      # seg index -> int16 array (in memory)
        self.windows = []                  # pass B: window i = seg[i] 2nd half + seg[i+1] 1st half
        self.unified = []                  # list of {span, text, fallback}
        self.unify_cursor = 0
        self.jobs = queue.Queue()
        self.jobs_pending = 0
        self.trans = TRANSCRIBER
        self.unify_q = queue.Queue()       # LLM merging runs off the STT thread
        self._stop_flag = False
        self._rec_proc = None
        self._buf = []
        self._buf_n = 0
        self.prediction = None             # live best-guess for the in-progress segment
        self._predict_busy = False
        self._last_predict = 0.0
        self._rec_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._work_thread = threading.Thread(target=self._work_loop, daemon=True)
        self._unify_thread = threading.Thread(target=self._unify_loop, daemon=True)
        self._rec_thread.start()
        self._work_thread.start()
        self._unify_thread.start()

    # ---------------- recording

    def _spawn_source(self):
        if INPUT_WAV:
            cmd = (["ffmpeg", "-v", "quiet"] + (["-re"] if INPUT_SPEED <= 1 else [])
                   + ["-i", INPUT_WAV, "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-"])
            return subprocess.Popen(cmd, stdout=subprocess.PIPE)
        return subprocess.Popen(
            ["arecord", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _record_loop(self):
        chunk = int(RATE * 0.1) * 2        # 100 ms
        try:
            self._rec_proc = self._spawn_source()
            while True:
                if self._stop_flag:
                    break
                proc = self._rec_proc
                if self.status == "paused" or proc is None:
                    time.sleep(0.05)
                    continue
                data = proc.stdout.read(chunk)
                if not data:
                    if self._stop_flag or INPUT_WAV:
                        break
                    if self.status == "paused":
                        continue          # source was killed by pause()
                    raise RuntimeError("audio source ended unexpectedly (mic busy?)")
                if self._stop_flag:
                    break
                arr = np.frombuffer(data, dtype=np.int16)
                lvl = float(np.sqrt((arr.astype(np.float32) ** 2).mean()))
                with self.lock:
                    self._buf.append(arr)
                    self._buf_n += len(arr)
                    n = self._buf_n
                self.broadcast({"type": "level", "rms": round(lvl, 1),
                                "threshold": round(self.calib.threshold, 1),
                                "elapsed": self.elapsed()})
                # live prediction: cheap greedy STT on the in-progress buffer
                if (PREDICT and not self._predict_busy and n >= RATE
                        and time.time() - self._last_predict >= PREDICT_EVERY):
                    self._last_predict = time.time()
                    self._predict_busy = True
                    with self.lock:
                        pbuf = np.concatenate(self._buf)
                        pidx = len(self.segments)
                    threading.Thread(target=self._predict, args=(pbuf, pidx), daemon=True).start()
                # prefer cutting at a quiet moment once near full; hard cut at 125%
                if (n >= SEG_SAMPLES_SOFT and lvl < self.calib.threshold) or n >= SEG_SAMPLES_HARD:
                    self._flush_buffer()
            self._flush_buffer(final=True)
        except Exception as e:
            with self.lock:
                self.status = "error"
                self.error = str(e)
            self.broadcast({"type": "status", "status": "error", "error": str(e)})
        finally:
            self._kill_source()
            if self.status != "error":
                self._finalize_schedule()

    def _kill_source(self):
        if self._rec_proc:
            try:
                self._rec_proc.kill()
                self._rec_proc.wait(timeout=2)
            except Exception:
                pass
            self._rec_proc = None

    def _flush_buffer(self, final=False):
        with self.lock:
            if self._buf_n == 0:
                return
            audio = np.concatenate(self._buf)
            self._buf, self._buf_n = [], 0
            if final and len(audio) < RATE * MIN_FLUSH_SEC:
                return
            self._close_segment(audio)

    def _close_segment(self, audio: np.ndarray):
        i = len(self.segments)
        t0 = sum(s["dur"] for s in self.segments)
        seg = {"i": i, "t0": t0, "dur": len(audio) / RATE, "state": "recorded",
               "kept": None, "speech_sec": None, "passA": None}
        self.segments.append(seg)
        self.raw[i] = audio
        write_wav(self.dir / f"seg_{i:04d}.wav", audio)   # crash-safety save
        if i >= 1:
            self.windows.append({"i": i - 1, "state": "pending", "text": None})
        self.broadcast({"type": "segment", "segment": self._seg_public(seg)})
        # transcribe pass A right away (realtime feel); window i-1 has both halves now
        self._enqueue(("analyze", i))
        self._enqueue(("passA", i))
        if i >= 1:
            self._enqueue(("passB", i - 1))
        self._save_state()

    def _finalize_schedule(self):
        """Called once recording has fully stopped: process the tail."""
        with self.lock:
            if self.status == "error":
                return
            self.status = "finalizing"
            self._enqueue(("finish", None))   # per-segment jobs were queued at close time
        self.broadcast({"type": "status", "status": "finalizing"})

    # ---------------- workers

    def _enqueue(self, job):
        with self.lock:
            self.jobs_pending += 1
        self.jobs.put(job)

    def _work_loop(self):
        while True:
            kind, arg = self.jobs.get()
            try:
                self._dispatch(kind, arg)
            except Exception:
                try:
                    self._dispatch(kind, arg)          # one retry
                except Exception as e:
                    self._mark_failed(kind, arg, e)    # never stall the pipeline
            finally:
                with self.lock:
                    self.jobs_pending -= 1
                self._gc_raw()
                if kind == "finish":
                    return
                self.unify_q.put("check")

    def _dispatch(self, kind, arg):
        if kind == "analyze":
            self._job_analyze(arg)
        elif kind == "passA":
            self._job_pass_a(arg)
        elif kind == "passB":
            self._job_pass_b(arg)
        elif kind == "finish":
            self._job_finish()

    def _mark_failed(self, kind, arg, e):
        self.broadcast({"type": "warn", "msg": f"{kind}({arg}) failed twice: {e}"})
        with self.lock:
            if kind in ("analyze", "passA") and arg < len(self.segments):
                seg = self.segments[arg]
                seg["state"] = "failed"
                if seg.get("passA") is None:
                    seg["passA"] = ""
                self.broadcast({"type": "segment", "segment": self._seg_public(seg)})
            elif kind == "passB" and arg < len(self.windows):
                w = self.windows[arg]
                w["state"], w["text"] = "failed", ""
                self.broadcast({"type": "window", "window": w})

    def _gc_raw(self):
        """Free in-memory raw audio once nothing can still need it (wav is on disk)."""
        term_s = ("done", "empty", "failed")
        term_w = ("done", "skipped", "failed")
        with self.lock:
            for i in list(self.raw):
                if self.segments[i]["state"] not in term_s:
                    continue
                left = self.windows[i - 1] if i >= 1 else None
                right = self.windows[i] if i < len(self.windows) else None
                if (left is None or left["state"] in term_w) and \
                   (right is not None and right["state"] in term_w):
                    del self.raw[i]

    def _predict(self, buf: np.ndarray, idx: int):
        """Cheap greedy STT on the in-progress buffer → live 'predicted' tail in the UI."""
        try:
            trimmed, _, sec, _m = trim_speech(buf, self.calib)
            if sec >= 0.3:
                text = self.trans.transcribe(trimmed, MODEL_NAME_B, 1)
                if text:
                    with self.lock:
                        self.prediction = {"i": idx, "text": text}
                    self.broadcast({"type": "predict", "i": idx, "text": text})
        except Exception:
            pass
        finally:
            self._predict_busy = False

    def _job_analyze(self, i):
        seg = self.segments[i]
        audio = self.raw[i]
        rms = frame_rms(audio)
        if not self.calib.calibrated or self.calib.low_signal:
            self.calib.calibrate(rms)      # retry calibration until real speech seen
            self.broadcast({"type": "calib", **self.calib.snapshot()})
        else:
            self.calib.adapt(rms)
        trimmed, kept, speech_sec, method = trim_speech(audio, self.calib)
        with self.lock:
            seg["kept"] = round(kept, 3)
            seg["speech_sec"] = round(speech_sec, 2)
            seg["vad"] = method
            if speech_sec < MIN_SPEECH_SEC:
                seg["state"] = "empty"
                seg["passA"] = ""
            else:
                seg["state"] = "analyzed"
                seg["_trimmed"] = trimmed
        self.broadcast({"type": "segment", "segment": self._seg_public(seg)})

    def _job_pass_a(self, i):
        seg = self.segments[i]
        with self.lock:
            if seg["state"] in ("empty", "failed", "done"):
                return
            seg["state"] = "transcribing"
            audio = seg.pop("_trimmed", None)
            if audio is None:
                audio = self.raw.get(i)
        if audio is None:
            with self.lock:
                seg["state"], seg["passA"] = "empty", ""
            return
        self.broadcast({"type": "segment", "segment": self._seg_public(seg)})
        text = self.trans.transcribe(audio)
        with self.lock:
            seg["passA"] = text
            seg["state"] = "done"
        self.broadcast({"type": "segment", "segment": self._seg_public(seg)})
        self._save_state()

    def _job_pass_b(self, wi):
        w = self.windows[wi]
        a, b = self.raw[wi], self.raw[wi + 1]
        combined = np.concatenate([a[len(a) // 2:], b[: len(b) // 2]])
        trimmed, _, speech_sec, _method = trim_speech(combined, self.calib)
        if speech_sec < MIN_SPEECH_SEC:
            with self.lock:
                w["state"], w["text"] = "skipped", ""
        else:
            text = self.trans.transcribe(trimmed, MODEL_NAME_B, BEAM_B)
            with self.lock:
                w["state"], w["text"] = "done", text
        self.broadcast({"type": "window", "window": w})
        self._save_state()

    def _job_finish(self):
        # stage 1: raw pass A is complete here — copy it now so a quick paste works
        # seconds after stop; the refined transcript replaces it when unify lands
        raw = " ".join(s["passA"] for s in self.segments if s.get("passA")).strip()
        if raw:
            try:
                subprocess.run(["wl-copy"], input=raw.encode(), timeout=5)
            except Exception:
                pass
            self.broadcast({"type": "raw_ready", "text": raw})
        done_ev = threading.Event()
        self.unify_q.put(("flush", done_ev))
        done_ev.wait(timeout=900)
        with self.lock:
            self.status = "done"
            self.raw.clear()
        full = self.unified_text()
        (self.dir / "unified.md").write_text(full)
        self._save_state()
        flat = full.replace("\t", " ").replace("\n", " / ")
        with open(DATA_DIR / "history.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{self.id}\t{self.elapsed()}\t{flat}\n")
        try:
            subprocess.run(["wl-copy"], input=full.encode(), timeout=5)
        except Exception:
            pass
        self.broadcast({"type": "status", "status": "done", "unified_full": full})

    def _unify_loop(self):
        while True:
            item = self.unify_q.get()
            if item == "check":
                self._maybe_unify(flush=False)
            else:
                _, ev = item
                self._maybe_unify(flush=True)
                ev.set()
                return

    # ---------------- unification

    def _ready_frontier(self):
        n = len(self.segments)
        f = 0
        while f < n:
            if self.segments[f]["state"] not in ("done", "empty", "failed"):
                break
            if f < n - 1 and self.windows[f]["state"] not in ("done", "skipped", "failed"):
                break
            f += 1
        return f

    def _maybe_unify(self, flush=False):
        while True:
            with self.lock:
                f = self._ready_frontier()
                n = len(self.segments)
                finished = flush
                u = self.unify_cursor
                avail = f - u
                if avail <= 0:
                    return
                if avail < UNIFY_MIN and not (finished and f == n):
                    return
                v = min(u + UNIFY_MAX, f)
                span = (u, v)
                a_parts, b_parts = [], []
                for i in range(u, v):
                    s = self.segments[i]
                    if s["passA"]:
                        a_parts.append(f"[{fmt_ts(s['t0'])}] {s['passA']}")
                for wi in range(max(0, u - 1), min(v - 1, len(self.windows))):
                    w = self.windows[wi]
                    if w["text"]:
                        mid = self.segments[wi]["t0"] + self.segments[wi]["dur"] / 2
                        b_parts.append(f"[{fmt_ts(mid)}] {w['text']}")
                context = self.unified_text()[-800:] or "(session start — nothing yet)"
            if not a_parts and not b_parts:
                with self.lock:
                    self.unify_cursor = v
                continue
            user = (f"PREVIOUS CONTEXT (already final, do not repeat):\n...{context}\n\n"
                    f"PASS A (aligned windows, timestamps are window starts):\n" + "\n".join(a_parts) +
                    "\n\nPASS B (half-offset windows, timestamps are window starts):\n" +
                    ("\n".join(b_parts) if b_parts else "(none)") +
                    "\n\nOutput the transcript continuation for the Pass A span only.")
            fallback = False
            for attempt in (1, 2):
                try:
                    text = call_minimax(UNIFY_SYSTEM, user)
                    break
                except Exception as e:
                    if attempt == 2:
                        text = " ".join(p.split("] ", 1)[-1] for p in a_parts)
                        fallback = True
                        self.broadcast({"type": "warn", "msg": f"unify LLM failed ({e}); used Pass A"})
                    else:
                        time.sleep(2)
            with self.lock:
                self.unify_cursor = v
                chunk = {"span": span, "text": text, "fallback": fallback}
                self.unified.append(chunk)
            self.broadcast({"type": "unified", "chunk": chunk, "cursor": v})
            self._save_state()

    # ---------------- controls & state

    def toggle_pause(self):
        with self.lock:
            if self.status == "recording":
                proc, self._rec_proc = self._rec_proc, None
                self.status = "paused"
                st = "paused"
            elif self.status == "paused":
                self._rec_proc = self._spawn_source()
                self.status = "recording"
                st = "recording"
                proc = None
            else:
                return
        if st == "paused":
            if proc:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            self._flush_buffer(final=True)     # close partial so it gets processed
        self.broadcast({"type": "status", "status": st})

    def stop(self):
        with self.lock:
            if self.status not in ("recording", "paused"):
                return
            self._stop_flag = True   # _finalize_schedule accumulates elapsed
        self._kill_source()
        # recorder thread sees _stop_flag, exits, and runs _finalize_schedule

    def elapsed(self):
        """Recorded audio time (samples-based, immune to pause/feed-rate)."""
        with self.lock:
            return round(sum(s["dur"] for s in self.segments) + self._buf_n / RATE, 1)

    def _seg_public(self, seg):
        return {k: v for k, v in seg.items() if not k.startswith("_")}

    def snapshot(self):
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "error": self.error,
                "elapsed": self.elapsed(),
                "calib": self.calib.snapshot(),
                "segments": [self._seg_public(s) for s in self.segments],
                "windows": self.windows,
                "unified": self.unified,
                "unify_cursor": self.unify_cursor,
                "prediction": self.prediction,
                "model": f"{MODEL_NAME}/b{BEAM} + {MODEL_NAME_B}/b{BEAM_B}",
                "seg_seconds": SEG_SECONDS,
            }

    def unified_text(self):
        return "\n\n".join(c["text"] for c in self.unified if c["text"]).strip()

    def _save_state(self):
        snap = self.snapshot()
        tmp = self.dir / "session.json.tmp"
        tmp.write_text(json.dumps(snap, indent=1))
        tmp.rename(self.dir / "session.json")
        (self.dir / "unified.md").write_text(self.unified_text())


# ---------------------------------------------------------------- flask app

app = Flask(__name__, static_folder=None)
STATIC = Path(__file__).parent / "static"
clients = []
clients_lock = threading.Lock()
session: Session | None = None
session_lock = threading.Lock()


def broadcast(event):
    msg = f"data: {json.dumps(event)}\n\n"
    with clients_lock:
        dead = []
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                q.dead = True      # generator will close so EventSource reconnects
                dead.append(q)
        for q in dead:
            clients.remove(q)


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC, p)


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/start")
def start():
    global session
    with session_lock:
        if session and session.status in ("recording", "paused", "finalizing"):
            return jsonify({"error": "session already active"}), 409
        session = Session(broadcast)
    broadcast({"type": "status", "status": "recording", "session": session.id})
    return jsonify({"ok": True, "id": session.id})


@app.post("/api/pause")
def pause():
    if session:
        session.toggle_pause()
    return jsonify({"ok": True})


@app.post("/api/stop")
def stop():
    if session:
        session.stop()
    return jsonify({"ok": True})


@app.get("/api/state")
def state():
    return jsonify(session.snapshot() if session else {"status": "idle"})


@app.post("/api/copy")
def copy():
    text = session.unified_text() if session else ""
    if request.json and request.json.get("text"):
        text = request.json["text"]
    try:
        subprocess.run(["wl-copy"], input=text.encode(), timeout=5)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/last")
def last():
    root = DATA_DIR / "sessions"
    if root.exists():
        for d in sorted(root.iterdir(), reverse=True):
            f = d / "unified.md"
            if f.exists() and f.read_text().strip():
                return Response(f.read_text(), mimetype="text/plain")
    return Response("", mimetype="text/plain")


@app.get("/api/sessions")
def sessions():
    root = DATA_DIR / "sessions"
    out = []
    if root.exists():
        for d in sorted(root.iterdir(), reverse=True)[:50]:
            f = d / "session.json"
            if f.exists():
                try:
                    j = json.loads(f.read_text())
                    out.append({"id": j["id"], "status": j.get("status"),
                                "elapsed": j.get("elapsed"), "segments": len(j.get("segments", [])),
                                "preview": " ".join(c["text"] for c in j.get("unified", []))[:120]})
                except Exception:
                    pass
    return jsonify(out)


@app.get("/api/audio/<sid>/<int:n>")
def audio(sid, n):
    if not re.fullmatch(r"[0-9_]+", sid):
        return jsonify({"error": "bad id"}), 400
    d = DATA_DIR / "sessions" / sid
    name = f"seg_{n:04d}.wav"
    if not (d / name).exists():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(d, name)


@app.get("/api/session/<sid>")
def get_session(sid):
    if not re.fullmatch(r"[0-9_]+", sid):
        return jsonify({"error": "bad id"}), 400
    f = DATA_DIR / "sessions" / sid / "session.json"
    if not f.exists():
        return jsonify({"error": "not found"}), 404
    return Response(f.read_text(), mimetype="application/json")


@app.get("/api/events")
def events():
    q = queue.Queue(maxsize=500)
    with clients_lock:
        clients.append(q)

    def gen():
        try:
            yield "data: " + json.dumps({"type": "hello"}) + "\n\n"
            while True:
                if getattr(q, "dead", False):
                    return
                try:
                    yield q.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            with clients_lock:
                if q in clients:
                    clients.remove(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    threading.Thread(target=TRANSCRIBER.warm, daemon=True).start()   # warm the GPU
    app.run(host="127.0.0.1", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
