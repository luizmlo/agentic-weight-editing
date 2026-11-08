#!/usr/bin/env bash
# Confirmação pré-registrada do segundo defeito (CADERNO 2026-11-08 20h): IFEval dev[48:200] (152 prompts), sem
# raciocínio, para o base, a escolha nº 1 do agente, o a1 (edição dos loops) e o braço P (system prompt).
# Depois retoma o t200.  uso: scripts/queue_if_confirm.sh <run> [<run2> ...]   (lê final.json de cada run)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
ARMS=base
for run in "$@"; do
  cid=$(py -c "import json;print(json.load(open('data/m2a_if/runs/$run/final.json'))['ranked'][0])")
  echo "$run nº1 = $cid"; echo "$cid" > "data/m2a_if/runs/$run/pick.txt"
  [[ -f data/gguf/m2aif_$run.gguf ]] || py scripts/poc7_build.py "m2aif_$run" "data/m2a_if/runs/$run/done/$cid.pt"
  ARMS="$ARMS,m2aif_$run"
done
py scripts/ifeval_eval.py dev152 "$ARMS,m2a_a1_062959_102,if_sys" --n 152 --offset 48 --np 16
scripts/llama_serve.sh stop || true
echo "=== retomando t200 ==="
scripts/queue_t200.sh
