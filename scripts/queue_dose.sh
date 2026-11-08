#!/usr/bin/env bash
# Espera a confirmação, sobe o avaliador, varre a dose (m2a_dose.py), para o avaliador e gera no dev48.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
while systemctl --user is-active -q wsp-confirm; do sleep 30; done
scripts/llama_serve.sh stop || true
systemd-run --user --unit wsp-m2a --collect -p MemoryMax=6G -p MemorySwapMax=1G \
  --working-directory=$PWD -E PATH=$PATH /usr/bin/bash -c "uv run --no-sync python -u scripts/m2a_scorer.py >> data/logs/m2a.log 2>&1"
until grep -q "avaliador pronto" <(tail -n 50 data/logs/m2a.log); do sleep 10; done
uv run --no-sync python -u scripts/m2a_dose.py
systemctl --user stop wsp-m2a; sleep 5
args=()
while IFS=: read -r name a id; do args+=("dose:$id"); done < data/m2a/runs/dose/pick.txt
exec scripts/queue_confirm.sh "${args[@]}"
