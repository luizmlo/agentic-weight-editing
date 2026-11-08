#!/usr/bin/env bash
# Teste trancado do melhor candidato M→A confirmado no dev (a0) no mesmo conjunto do t12k (GSM8K test[0:32]).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
WSP_UNLOCK_TEST=1 uv run --no-sync python -u scripts/gen_eval.py t12k 32 12288 base,astra,m2a_a0 --np 16
scripts/llama_serve.sh stop || true
