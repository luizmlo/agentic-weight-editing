#!/usr/bin/env bash
# M→A, confirmação pelo supervisor: para o avaliador, gera o GGUF de candidatos e mede no dev48
# (GSM8K train[0:48], teto 12k), pareado com base/astra (já em cache).
# args: <run> (usa ranked[0] do final.json) ou <run>:<id>
# uso: scripts/run_bg.sh confirm scripts/queue_confirm.sh r1 a0 r1:192156_139
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
systemctl --user stop wsp-m2a 2>/dev/null || true
sleep 5
for arg in "$@"; do
  run=${arg%%:*}
  if [[ $arg == *:* ]]; then id=${arg#*:}; name=m2a_${run}_${id}
  else
    f=data/m2a/runs/$run/final.json
    [[ -f $f ]] || { echo "sem $f"; continue; }
    id=$(uv run --no-sync python -c "import json;print(json.load(open('$f'))['ranked'][0])")
    name=m2a_${run}
  fi
  echo "== $arg -> $id ($name)"
  [[ -f data/gguf/$name.gguf ]] || uv run --no-sync python -u scripts/poc7_build.py $name data/m2a/runs/$run/done/$id.pt
  uv run --no-sync python -u scripts/gen_eval.py dev48 48 12288 base,astra,$name --np 16 --split train
done
scripts/llama_serve.sh stop || true
