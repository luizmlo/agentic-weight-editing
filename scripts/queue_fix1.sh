#!/usr/bin/env bash
# Fila 1 de 10/11/2026 (pré-registro no CADERNO, 10h55): t200 sem raciocínio; nº 1 dos agentes a1 e a2 no t200.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
WSP_UNLOCK_TEST=1 uv run --no-sync python -u scripts/gen_eval.py t200nt 200 8192 base,m2a_a1_062959_102,astra \
  --np 32 --offset 200 --nothink
WSP_UNLOCK_TEST=1 uv run --no-sync python -u scripts/gen_eval.py t200 200 12288 m2a_a1,m2a_a2 --np 16 --offset 200
scripts/llama_serve.sh stop || true
echo "fim da fila 1: $(date)"
