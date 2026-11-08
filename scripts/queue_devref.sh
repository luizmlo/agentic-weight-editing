#!/usr/bin/env bash
# Referências do split de dev do M→A (GSM8K train[0:48], com raciocínio, teto 12k): base e astra.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
for pass in 1 2; do uv run --no-sync python -u scripts/gen_eval.py dev48 48 12288 base,astra --np 16 --split train; done
scripts/llama_serve.sh stop || true
