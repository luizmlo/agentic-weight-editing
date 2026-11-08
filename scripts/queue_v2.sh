#!/usr/bin/env bash
# Espera o MATH-500 liberar a GPU e sobe o avaliador v2 (sonda + checagem por geração).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
while systemctl --user is-active -q wsp-math; do sleep 30; done
scripts/llama_serve.sh stop || true
exec uv run --no-sync python -u scripts/m2a_scorer.py
