# sttl — long-form dual-pass local dictation

Progressive speech-to-text: record for minutes or hours, watch a unified
transcript build as you talk. Audio is chunked (~8s, cut at quiet moments,
saved to disk immediately), Silero neural VAD drops noise-only spans (1s safety
buffer), then each region is transcribed twice — aligned windows (whisper
medium) and half-offset windows (whisper small) — and an LLM (MiniMax M2.5)
merges the two passes into clean text, chunk by chunk. On stop, the raw
transcript hits the clipboard in ~1s; the refined one replaces it seconds later.

Quick capture too: `sttl -q` (terminal, Enter to finish) or `sttl -w`
(auto-start window that closes itself) — same pipeline, open→talk→paste.

**Run**: `./sttl` (server on :7737 + Chrome app window) or `./start.sh -`.
**Deps**: python3 + faster-whisper, flask, numpy; arecord, ffmpeg, wl-copy,
google-chrome. LLM merge wants a MiniMax key at `~/.keys/.minimax2.5_tool_caller`
(without it, transcripts fall back to raw Pass A).

- Hotkeys: Space start/pause · S stop · C copy · D download · P passes · H history · ?
- Click a timeline segment to replay its audio; transcript is editable after stop
- Data: `~/.local/share/sttl/sessions/<id>/` (wavs, session.json, unified.md) + `history.log` TSV
- `sttl --last` prints newest transcript; `sttl --stop` kills the server
- Env: STTL_MODEL/BEAM (medium/5, pass A), STTL_MODEL_B/BEAM_B (small/8, pass B),
  STTL_SEG_SECONDS (8), STTL_PORT (7737),
  STTL_UNIFY_MIN/MAX (3/6), STTL_DATA; test hooks: STTL_FAKE_STT/FAKE_LLM/INPUT_WAV/SPEED/FAIL_ONCE
- Colors: `static/theme.css`
