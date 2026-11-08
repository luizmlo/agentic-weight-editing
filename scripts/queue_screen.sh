#!/usr/bin/env bash
# Triagem do segundo defeito: modelo base, sem raciocínio, 3 famílias de saída estruturada (dev, 120 cada).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
for fam in json tools ifeval; do py scripts/struct_eval.py screen_$fam $fam 120 base,m2a_a1_062959_102 --np 16; done
