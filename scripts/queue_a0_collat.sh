#!/usr/bin/env bash
# Colaterais do m2a_a0 (painel da noite; base/astra já em cache por item).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
ARMS=base,astra,m2a_a0
py scripts/tool_eval.py nothink_dev300 300 $ARMS --np 32
py scripts/gen_eval.py nothink_n200 200 1024 $ARMS --np 32 --nothink
py scripts/tool_eval.py think_dev100 100 $ARMS --think --max-new 3072 --np 32
scripts/llama_serve.sh stop || true
