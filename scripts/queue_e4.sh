#!/usr/bin/env bash
# E4.0–4.2: viabilidade -> dados (autodestilação) -> LoRA-SFT -> GGUF por época -> escolha na checagem do agente
# (GSM8K train[200:216], teto 6144, a mesma informação que o agente teve) -> dev48. Pré-registro: CADERNO 2026-11-07.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
while systemctl --user is-active -q wsp-e3math; do sleep 60; done
scripts/llama_serve.sh stop || true
if [[ ! -f data/ft/spike.json ]]; then
  echo "== E4.0 viabilidade $(date +%T)"
  py scripts/ft_spike.py
fi
ML=$(py -c "import json;print(json.load(open('data/ft/spike.json')).get('recommended_max_len') or '')" 2>/dev/null)
[[ -n $ML ]] || { echo "LoRA não coube na GPU: parando (ver data/ft/spike.json)"; exit 1; }
echo "== E4.1 dados (spike recomendou $ML; treino usa 8192, ver CADERNO) $(date +%T)"
py scripts/ft_data.py 600
scripts/llama_serve.sh stop || true
echo "== E4.2 treino $(date +%T)"
py scripts/ft_train.py sft --epochs 3 --max-len 8192 || { echo "treino falhou"; exit 1; }
ARMS=base
for ep in 1 2 3; do
  [[ -d data/ft/sft/ep$ep ]] || continue
  py scripts/ft_export.py ft_sft_ep$ep data/ft/sft/ep$ep && ARMS=$ARMS,ft_sft_ep$ep
done
echo "== escolha da época $(date +%T)"
py scripts/gen_eval.py ftsel 16 6144 $ARMS --np 16 --split train --offset 200
PICK=$(py - <<'PY'
import json
a = json.load(open("data/gen_ftsel.json"))["arms"]
c = [(v["summary"]["acc"], -v["summary"]["cut"], int(k[-1]), k) for k, v in a.items() if k.startswith("ft_sft_ep")]
print(max(c)[3] if c else "")
PY
)
echo "escolhido: $PICK $(date +%T)" | tee data/ft/pick.txt
[[ -n $PICK ]] && py scripts/gen_eval.py dev48 48 12288 base,$PICK --np 16 --split train
scripts/llama_serve.sh stop || true
echo "fim $(date +%T)"
