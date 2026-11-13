#!/usr/bin/env bash
# Depois das réplicas a3/a4 (pré-registro: CADERNO 13/11 03h20): confirma a nº 1 de cada no dev48 e retoma a fila do
# teto longo (os efeitos colaterais já estão em cache e são pulados).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
for r in a3 a4; do
  rsync -a /mnt/dados/dev/deep_traces/data/wsp/m2a/runs/$r/ data/m2a/runs/$r/
done
scripts/queue_confirm.sh a3 a4
exec scripts/queue_side_long.sh
