#!/usr/bin/env bash
# Baseline de fine-tuning do segundo defeito: dados (amostragem de rejeição) -> LoRA-SFT (mesma receita do E4) ->
# GGUF por época -> escolha da época na checagem do agente (IFEval dev[0:48]) -> confirmação no dev152.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
[[ -f data/ft_if/sft.jsonl ]] || py scripts/ft_if_data.py 600 2
scripts/llama_serve.sh stop || true
py scripts/ft_train.py sft_if --data data/ft_if/sft.jsonl --epochs 3 --max-len 3072 || { echo "treino falhou"; exit 1; }
ARMS=base
for ep in 1 2 3; do
  [[ -d data/ft/sft_if/ep$ep ]] || continue
  py scripts/ft_export.py ft_if_ep$ep data/ft/sft_if/ep$ep && ARMS=$ARMS,ft_if_ep$ep
done
py scripts/ifeval_eval.py ftsel_if "$ARMS" --n 48 --offset 0 --np 16
PICK=$(py - <<'PY'
import json
a = json.load(open("data/ifeval_ftsel_if.json"))["arms"]
c = [(v["summary"]["prompt_strict"], v["summary"]["inst_strict"], int(k[-1]), k) for k, v in a.items() if k.startswith("ft_if_ep")]
print(max(c)[3] if c else "")
PY
)
echo "escolhido: $PICK $(date +%T)" | tee data/ft_if/pick.txt
[[ -n $PICK ]] && py scripts/ifeval_eval.py dev152 "$PICK" --n 152 --offset 48 --np 16
scripts/llama_serve.sh stop || true
