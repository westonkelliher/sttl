#!/usr/bin/env bash
# start.sh — clean-slate launcher: kills existing instances, starts server, opens UI.
set -euo pipefail

C_ACC=$'\033[38;2;110;200;230m'
C_DIM=$'\033[38;2;138;138;138m'
C_ERR=$'\033[38;2;230;85;90m'
C_OK=$'\033[38;2;94;199;106m'
C_RST=$'\033[0m'

if [[ $# -lt 1 ]]; then
    echo "${C_ACC}usage:${C_RST} start.sh <port>   ${C_DIM}(\"-\" = default: 7737)${C_RST}"
    exit 1
fi

PORT="$1"
[[ "$PORT" == "-" ]] && PORT=7737
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "${C_ERR}error: port must be a number or '-'${C_RST}" >&2; exit 1; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HOME/.local/share/sttl"
URL="http://127.0.0.1:$PORT"
mkdir -p "$DATA"

# kill any existing instances (by script path and by port) so we never run two
pkill -f "python3 $DIR/server.py" 2>/dev/null && sleep 0.5 || true
fuser -k "$PORT/tcp" 2>/dev/null && sleep 0.5 || true

STTL_PORT="$PORT" nohup python3 "$DIR/server.py" >> "$DATA/server.log" 2>&1 &

# wait for the server before opening the browser
up=0
for _ in $(seq 1 60); do
    curl -sf "$URL/api/health" >/dev/null 2>&1 && { up=1; break; }
    sleep 0.1
done
if [[ "$up" -ne 1 ]]; then
    echo "${C_ERR}error: server failed to start; last log lines:${C_RST}" >&2
    tail -5 "$DATA/server.log" >&2
    exit 1
fi

echo "${C_OK}✓ sttl running at $URL${C_RST}"
nohup google-chrome --app="$URL" >/dev/null 2>&1 &
disown
