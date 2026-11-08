#!/usr/bin/env bash
# Fila noturna (2026-11-04): espera o PoC-6 terminar e roda, em ordem de prioridade:
#   1. poc7_dirs   direção por diferença de médias -> edição no formato do Astra (+ controle aleatório),
#                  Δlogp nos textos em loop e em texto neutro (WikiText)                         ~15 min GPU (HF)
#   2. poc7_build  materializa meandiff e random como GGUF                                        ~5 min CPU
#   3. painel colateral, barato (respostas curtas, sem raciocínio):
#        tool_eval  tool calling sintético, 300 tarefas dev                                       ~15 min
#        gen_eval   GSM8K sem raciocínio, 200 problemas                                           ~25 min
#   4. gen_eval    GSM8K COM raciocínio, 32 problemas, teto 12k, penalty na saída inteira          ~5 h
#   5. tool_eval   tool calling com raciocínio, 100 tarefas, teto 3072                            ~1 h (se der tempo)
# Braços: base, astra, meandiff, random. Tudo retomável; um passo que falha não impede os seguintes.
# uso: scripts/run_bg.sh queue scripts/queue_night.sh     log: data/logs/queue.log
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
log() { echo "[$(date '+%F %T')] $*"; }
py() { log ">> $*"; uv run --no-sync python -u "$@"; log "<< rc=$? $1"; }
ARMS=base,astra,meandiff,random

while systemctl --user is-active -q wsp-poc6; do sleep 60; done
log "PoC-6 terminou"
scripts/llama_serve.sh stop || true   # GPU livre para o HF
sleep 5

[[ -f data/poc7_dirs.pt ]] || py scripts/poc7_dirs.py
for m in meandiff random; do
  [[ -f data/gguf/$m.gguf ]] || py scripts/poc7_build.py "$m"
done
# duas passadas em cada avaliação: a segunda só completa braços que caíram no meio
for pass in 1 2; do py scripts/tool_eval.py nothink_dev300 300 $ARMS --np 32; done
for pass in 1 2; do py scripts/gen_eval.py nothink_n200 200 1024 $ARMS --np 32 --nothink; done
for pass in 1 2; do py scripts/gen_eval.py t12k 32 12288 $ARMS --np 16; done
for pass in 1 2; do py scripts/tool_eval.py think_dev100 100 $ARMS --think --max-new 3072 --np 32; done
scripts/llama_serve.sh stop || true
log "fila concluída"
