#!/usr/bin/env bash
# Madrugada de 2026-11-08 (ordem pré-registrada no CADERNO, 21h40): espera os 5 braços principais do t200 terminarem,
# pausa o t200 (o checkpoint por item preserva o que já foi gerado), roda o baseline FT do IFEval e sai. O b2 e a cauda
# do t200 (lbx, LoRA) são lançados depois, à mão.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
n0=1   # linhas '^base {' no ifconfirm.log antes do t200 (do dev152)
until [[ $(grep -c '^base {' data/logs/ifconfirm.log) -gt $n0 ]] || ! systemctl --user is-active -q wsp-ifconfirm; do
  sleep 60
done
echo "t200 principal pronto $(date +%T)"; grep -E '^(m2a_a1_062959_102|bf4096|m2a_a1_062959_102\+bf4096|astra|base) \{' data/logs/ifconfirm.log | tail -5 | cut -c1-200
systemctl --user stop wsp-ifconfirm || true
scripts/llama_serve.sh stop || true
sleep 5
bash scripts/queue_if_ft.sh
echo "fila da madrugada terminou $(date +%T)"
