#!/usr/bin/env bash
# Transferência para o MATH-500 (a tarefa do paper): base/astra/m2a_a0, 48 problemas, teto 12k.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
while systemctl --user is-active -q wsp-a0-collat; do sleep 30; done
uv run --no-sync python -u scripts/math_eval.py math48 48 12288 base,astra,m2a_a0 --np 16
scripts/llama_serve.sh stop || true
