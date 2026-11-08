#!/usr/bin/env bash
# Pausa do t200 para a triagem do segundo defeito (curta) e retomada do t200 (retoma por item).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
scripts/queue_screen.sh
echo "=== retomando t200 ==="
scripts/queue_t200.sh
