#!/usr/bin/env bash
# a1 + budget forcing (bf4096, o mesmo ajustado no base; sem reajuste) no dev48, t12k e math48. Pré-registro: CADERNO 2026-11-06 tarde.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
C=m2a_a1_062959_102+bf4096
py scripts/gen_eval.py dev48 48 12288 base,$C --np 16 --split train
WSP_UNLOCK_TEST=1 py scripts/gen_eval.py t12k 32 12288 base,$C --np 16
py scripts/math_eval.py math48 48 12288 base,$C --np 16
scripts/llama_serve.sh stop || true
echo "fim $(date +%T)"
