#!/usr/bin/env bash
# Checagem do sinal das ativações (precisa da GPU inteira) e depois a fila 1 (retoma por item).
cd "$(dirname "$(readlink -f "$0")")/.."
scripts/llama_serve.sh stop || true
uv run --no-sync python -u scripts/act_sign_check.py 12 1500 > data/logs/actsign.log 2>&1
exec scripts/queue_fix1.sh
