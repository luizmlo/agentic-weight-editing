#!/usr/bin/env bash
# Teste trancado ampliado (pré-registro no CADERNO, 2026-11-08): GSM8K test[200:400], nunca tocado por nenhum braço
# (test[0:32] = t12k; test[0:200] só foi usado sem raciocínio, no painel de efeitos colaterais).
# Braços em ordem de prioridade; cada braço retoma por item (data/gen_t200/<braço>.jsonl), então a fila pode ser
# parada e relançada sem perder o que já terminou.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
ARMS=${1:-m2a_a1_062959_102,bf4096,m2a_a1_062959_102+bf4096,astra,base,lbx_w10_t1,ft_sft_ep2}
WSP_UNLOCK_TEST=1 uv run --no-sync python -u scripts/gen_eval.py t200 200 12288 "$ARMS" --np 16 --offset 200
scripts/llama_serve.sh stop || true
