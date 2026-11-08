#!/usr/bin/env bash
# E3 (busca ingênua com geração, mesmo orçamento do agente) -> dev48; depois MATH-500[48:148] para os braços principais.
# Pré-registro: CADERNO 2026-11-06 noite.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
scripts/llama_serve.sh stop || true
scripts/run_bg.sh m2a scripts/m2a_scorer.py
for i in $(seq 1 60); do grep -q "avaliador pronto" data/logs/m2a.log 2>/dev/null && break; sleep 10; done
echo "avaliador: $(tail -1 data/logs/m2a.log) $(date +%T)"
py scripts/m2a_naive.py naive
scripts/queue_confirm.sh naive
A=m2a_a1_062959_102
py scripts/math_eval.py math100b 100 12288 $A+bf4096,$A,bf4096,astra,base --np 16 --offset 48
scripts/llama_serve.sh stop || true
echo "fim $(date +%T)"
