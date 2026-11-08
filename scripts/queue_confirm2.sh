#!/usr/bin/env bash
# Réplicas de H4: espera o final.json do r2 e confirma no dev48 (r2 nº1, a1 nº1, a1 variante branda).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
until [[ -f data/m2a/runs/r2/final.json ]]; do sleep 30; done
sleep 60
exec scripts/queue_confirm.sh a1 r2 a1:062959_102
