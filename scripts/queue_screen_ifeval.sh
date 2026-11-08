#!/usr/bin/env bash
# Triagem 3: IFEval (dev 200, sem raciocínio), modelo base; depois retoma o t200.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv run --no-sync python -u scripts/ifeval_eval.py dev200 base --np 16
echo "=== retomando t200 ==="
scripts/queue_t200.sh
