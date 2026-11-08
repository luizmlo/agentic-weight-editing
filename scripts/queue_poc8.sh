#!/usr/bin/env bash
# PoC-8 (2026-11-04): fatorial direção {astra, md, mdloop} × colunas {astra, topw, contrast}.
#   1. poc8_cells   células + métricas teacher-forced (previsão registrada ANTES da geração)    ~30 min GPU
#   2. poc7_build   GGUF das células novas                                                       ~1 min cada
#   3. gen_eval t12k  mesmos 32 problemas/teto/penalty do PoC-7b; base/astra/meandiff/random já feitos
#                     ordem de prioridade abaixo                                                 ~40 min por braço
# uso: scripts/run_bg.sh poc8 scripts/queue_poc8.sh     log: data/logs/poc8.log
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
log() { echo "[$(date '+%F %T')] $*"; }
py() { log ">> $*"; uv run --no-sync python -u "$@"; log "<< rc=$? $1"; }
ARMS=astra_contrast,mdloop_contrast,md_astra,astra_topw,md_contrast,mdloop_astra,mdloop_topw

scripts/llama_serve.sh stop || true
[[ -f data/poc8_cells.json ]] && grep -q random_topw data/poc8_cells.json || py scripts/poc8_cells.py
for m in ${ARMS//,/ }; do
  [[ -f data/gguf/$m.gguf ]] || py scripts/poc7_build.py "$m" poc8_cells.pt
done
for pass in 1 2; do py scripts/gen_eval.py t12k 32 12288 $ARMS --np 16; done
scripts/llama_serve.sh stop || true
log "PoC-8 concluído"
