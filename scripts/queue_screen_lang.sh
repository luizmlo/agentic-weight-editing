#!/usr/bin/env bash
# Triagem 2: fidelidade de idioma (pedido em português), modelo base; depois retoma o t200.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
py scripts/struct_eval.py screen_lang lang 120 base --np 16 --sampling nothink
py scripts/struct_eval.py screen_lang_think lang 48 base --np 16 --think --max-new 3072 --sampling think35
echo "=== retomando t200 ==="
scripts/queue_t200.sh
