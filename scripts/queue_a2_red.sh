#!/usr/bin/env bash
# Caminho da a2 por geração (pré-registro: CADERNO 11/11 15h15). Gera os GGUF e mede no dev48.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
systemctl --user stop wsp-m2a 2>/dev/null || true
uv run --no-sync python scripts/make_a2_variants.py
for v in a2_red a2_l16n998; do
  [[ -f data/gguf/m2a_$v.gguf ]] || uv run --no-sync python -u scripts/poc7_build.py m2a_$v data/m2a/variants/$v.pt
done
uv run --no-sync python -u scripts/gen_eval.py dev48 48 12288 base,m2a_a2,m2a_a2_red,m2a_a2_l16n998 --np 16 --split train
scripts/llama_serve.sh stop || true
