#!/usr/bin/env bash
# a1 (variante branda 062959_102, melhor no dev48): teste trancado t12k, MATH-500 e colaterais.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
A=m2a_a1_062959_102
WSP_UNLOCK_TEST=1 py scripts/gen_eval.py t12k 32 12288 base,astra,m2a_a0,$A --np 16
py scripts/math_eval.py math48 48 12288 base,astra,m2a_a0,$A --np 16
py scripts/tool_eval.py nothink_dev300 300 base,astra,$A --np 32
py scripts/gen_eval.py nothink_n200 200 1024 base,astra,$A --np 32 --nothink
py scripts/tool_eval.py think_dev100 100 base,astra,$A --think --max-new 3072 --np 32
scripts/llama_serve.sh stop || true
