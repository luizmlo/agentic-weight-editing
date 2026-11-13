#!/usr/bin/env bash
# Sem prazo: roda até o fim (pré-registro: CADERNO 13/11 02h55; prazo das 08h50 retirado às 03h00). Efeitos colaterais da a2 reduzida e da a2;
# depois Astra (e, se der tempo, a2 e a1) com teto de 20.480 no t200, que já foi aberto.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
run() {
  echo "== $(date '+%F %T') $*"
  uv run --no-sync python -u "$@"
}
export WSP_UNLOCK_TEST=1
ARMS=base,m2a_a2_red,m2a_a2
run scripts/tool_eval.py nothink_dev300 300 $ARMS --np 32
run scripts/gen_eval.py nothink_n200 200 1024 $ARMS --np 32 --nothink
run scripts/tool_eval.py think_dev100 100 $ARMS --think --max-new 3072 --np 32
for arm in astra m2a_a2 m2a_a1_062959_102; do
  run scripts/gen_eval.py t200L 200 20480 "$arm" --np 16 --offset 200
done
scripts/llama_serve.sh stop || true
echo "== $(date '+%F %T') fim"
