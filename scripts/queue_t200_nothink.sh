#!/usr/bin/env bash
# Janela de GPU de 10/11/2026 (11h–14h). Pré-registro no CADERNO (2026-11-10 09h40).
# 1) t200 sem raciocínio (base, a1, astra), mesmos problemas do t200; 2) retoma o braço lbx_w10_t1 do t200 até o fim
# da janela (gen_eval retoma por item); 3) para o llama-server.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
START=${START:-11:00}; END=${END:-13:50}
now=$(date +%s); st=$(date -d "$START" +%s); [[ $st -gt $now ]] && { echo "esperando até $START"; sleep $((st - now)); }
WSP_UNLOCK_TEST=1 uv run --no-sync python -u scripts/gen_eval.py t200nt 200 8192 base,m2a_a1_062959_102,astra \
  --np 32 --offset 200 --nothink
left=$(( $(date -d "$END" +%s) - $(date +%s) ))
if [[ $left -gt 600 ]]; then
  echo "retomando lbx_w10_t1 por ${left}s"
  WSP_UNLOCK_TEST=1 timeout "$left" uv run --no-sync python -u scripts/gen_eval.py t200 200 12288 lbx_w10_t1 \
    --np 16 --offset 200
fi
scripts/llama_serve.sh stop || true
echo "fim da janela: $(date)"
