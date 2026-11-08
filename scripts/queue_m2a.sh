#!/usr/bin/env bash
# M→A: espera as referências de dev liberarem a GPU, monta o atlas (se faltar) e vira o avaliador (daemon).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
while systemctl --user is-active -q wsp-devref; do sleep 30; done
scripts/llama_serve.sh stop || true
[[ -f data/m2a/atlas.pt ]] || uv run --no-sync python -u scripts/m2a_atlas.py
exec uv run --no-sync python -u scripts/m2a_scorer.py
