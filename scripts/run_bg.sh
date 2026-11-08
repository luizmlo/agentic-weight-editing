#!/usr/bin/env bash
# Roda um script do experimento numa unit própria do systemd --user (fora do cgroup do terminal):
# sobrevive ao fim do Claude Code/terminal, e o teto de memória faz um OOM matar só o experimento.
# (setsid/nohup não basta: o processo continua no scope do terminal e morre junto com ele.)
#
# uso: scripts/run_bg.sh <nome> <script.py|script.sh> [args...]    -> unit wsp-<nome>, log data/logs/<nome>.log
#      systemctl --user status wsp-<nome> | systemctl --user stop wsp-<nome>
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
mkdir -p "$ROOT/data/logs"
name=$1; shift
systemctl --user stop "wsp-$name" 2>/dev/null || true
systemctl --user reset-failed "wsp-$name" 2>/dev/null || true
systemd-run --user --unit "wsp-$name" --collect \
  -p WorkingDirectory="$ROOT" -p MemoryMax="${WSP_MEM_MAX:-6G}" -p MemorySwapMax=1G \
  -p StandardOutput="truncate:$ROOT/data/logs/$name.log" -p StandardError=inherit \
  -E PATH="$PATH" -E HOME="$HOME" -E HF_HUB_DISABLE_PROGRESS_BARS=1 \
  $( [[ $1 == *.sh ]] && echo /usr/bin/bash || echo "$(command -v uv) run --no-sync python -u" ) "$@"
echo "unit wsp-$name -> data/logs/$name.log"
