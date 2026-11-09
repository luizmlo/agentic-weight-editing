#!/usr/bin/env bash
# Confirmação pré-registrada do b2 (CADERNO 2026-11-08 21h40): IFEval dev200 inteiro (a checagem do b2 não usou o
# IFEval), base reamostrado + escolha nº 1 do b2. Critério: ganho pareado > 2 EP.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
cid=$(py -c "import json;print(json.load(open('data/m2a_if/runs/b2/final.json'))['ranked'][0])")
echo "b2 nº1 = $cid"; echo "$cid" > data/m2a_if/runs/b2/pick.txt
[[ -f data/gguf/m2aif_b2.gguf ]] || py scripts/poc7_build.py m2aif_b2 "data/m2a_if/runs/b2/done/$cid.pt"
py scripts/ifeval_eval.py dev200b2 base,m2aif_b2 --n 200 --offset 0 --np 16
scripts/llama_serve.sh stop || true
