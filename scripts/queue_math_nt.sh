#!/usr/bin/env bash
# Janela de GPU até 09h00 de 12/11 (pré-registro: CADERNO 12/11 00h45). Espera a fila t400 acabar,
# depois roda no MATH-500[0:148]: original sem raciocínio, a2, a2 reduzida. Prazo duro: 08h50 (libera a GPU).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
DEADLINE=$(date -d '2026-11-12 08:50' +%s)
left() { echo $(( DEADLINE - $(date +%s) )); }
while systemctl --user is-active -q wsp-t400; do sleep 60; done
echo "== $(date '+%F %T') início (t400 terminou)"
run() {
  local s; s=$(left)
  if (( s < 600 )); then echo "== sem tempo para $*"; return; fi
  echo "== $(date '+%F %T') $* (limite ${s}s)"
  timeout --signal=INT --kill-after=60 "$s" uv run --no-sync python -u scripts/math_eval.py "$@"
}
run math148nt 148 8192 base --np 16 --nothink
run math148 148 12288 m2a_a2 --np 16
run math148 148 12288 m2a_a2_red --np 16
scripts/llama_serve.sh stop || true
echo "== $(date '+%F %T') fim"
