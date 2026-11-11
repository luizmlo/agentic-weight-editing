#!/usr/bin/env bash
# Segundo teste novo, GSM8K test[400:600] (pré-registro: CADERNO 11/11 19h00). Um braço por chamada, em ordem de
# prioridade; cada braço é salvo ao terminar.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
systemctl --user stop wsp-m2a 2>/dev/null || true
export WSP_UNLOCK_TEST=1
uv run --no-sync python -u scripts/gen_eval.py t400nt 200 8192 base --np 32 --offset 400 --nothink
for arm in m2a_a2 m2a_a1_062959_102+bf4096a1024 m2a_a2_red bf4096a1024 m2a_a1_062959_102; do
  echo "== $(date '+%F %T') $arm"
  uv run --no-sync python -u scripts/gen_eval.py t400 200 12288 "$arm" --np 16 --offset 400
done
scripts/llama_serve.sh stop || true
echo "== $(date '+%F %T') fim"
