#!/usr/bin/env bash
# Runs f1/f2 (pré-registro: CADERNO 10/11 10h50): confirma a nº 1 de cada no dev48; a melhor vai ao t200.
# Depois segue a fila 2.
set -uo pipefail
cd /mnt/dados/dev/agentic-weight-editing
echo "início: $(date)"
scripts/queue_confirm.sh f1 f2
W=$(uv run --no-sync python - <<'PY'
import json
d=json.load(open('data/gen_dev48.json'))['arms']
c=[(sum(bool(r['correct']) for r in d[a]['rows']), -sum(r['n_tokens'] for r in d[a]['rows']), a) for a in ('m2a_f1','m2a_f2')]
print(max(c)[2])
PY
)
echo "melhor no dev48: $W"
WSP_UNLOCK_TEST=1 uv run --no-sync python -u scripts/gen_eval.py t200 200 12288 "$W" --np 16 --offset 200
exec scripts/queue_fix2.sh
