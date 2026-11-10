#!/usr/bin/env bash
# Fila 2 (pré-registro: CADERNO 10/11 16h55): limite de raciocínio com teto para a resposta + cauda do t200.
set -uo pipefail
cd /mnt/dados/dev/agentic-weight-editing
echo "início da fila 2: $(date)"
# 1) varredura no dev48 (prática), modelo original
uv run --no-sync python -u scripts/gen_eval.py dev48 48 12288 bf1024a1024,bf2048a1024,bf4096a1024 --np 16
# 2) escolha pela regra pré-registrada: maior acerto no dev48; empate -> menos tokens
W=$(uv run --no-sync python - <<'PY'
import json
d=json.load(open('data/gen_dev48.json'))['arms']
c=[(sum(bool(r['correct']) for r in d[a]['rows']), -sum(r['n_tokens'] for r in d[a]['rows']), a) for a in ('bf1024a1024','bf2048a1024','bf4096a1024')]
print(max(c)[2])
PY
)
echo "vencedor no dev48: $W"
# 3) vencedor no t200, sozinho e com a a1
WSP_UNLOCK_TEST=1 uv run --no-sync python -u scripts/gen_eval.py t200 200 12288 "$W,m2a_a1_062959_102+$W" --np 16 --offset 200
# 4) cauda pré-registrada do t200
WSP_UNLOCK_TEST=1 uv run --no-sync python -u scripts/gen_eval.py t200 200 12288 lbx_w10_t1,ft_sft_ep2 --np 16 --offset 200
scripts/llama_serve.sh stop || true
echo "fim da fila 2: $(date)"
