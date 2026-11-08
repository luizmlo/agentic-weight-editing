#!/usr/bin/env bash
# E1 (baselines de decodificação) + confirmação das réplicas a2/r3. Pré-registro: CADERNO 2026-11-06.
#  1) espera as réplicas do agente terminarem (final.json; no máximo 5 h) e para o avaliador
#  2) varredura no dev48 (GSM8K train[0:48], teto 12k): logit bias e budget forcing sobre o base
#  3) escolhe o melhor de cada família no dev48 (acc; empate -> menos tokens)
#  4) escolhidos + a1 no teste trancado t12k e no MATH-500[0:48]
#  5) confirma a2 e r3 no dev48
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
py() { uv run --no-sync python -u "$@"; }
R=data/m2a/runs
for i in $(seq 1 300); do [[ -f $R/a2/final.json && -f $R/r3/final.json ]] && break; sleep 60; done
echo "réplicas: a2=$([[ -f $R/a2/final.json ]] && echo ok || echo FALTA) r3=$([[ -f $R/r3/final.json ]] && echo ok || echo FALTA) $(date +%T)"
systemctl --user stop wsp-m2a 2>/dev/null || true
sleep 5
scripts/llama_serve.sh stop || true
py scripts/gen_eval.py dev48 48 12288 base,astra,lb_w10_t0,lbx_w10_t1,lb_w3_t1,bf4096,bf8192 --np 16 --split train
PICKS=$(py - <<'PY'
import json, re
a = json.load(open("data/gen_dev48.json"))["arms"]
def best(pat):
    c = [(v["summary"]["acc"], -v["summary"]["mean_tokens"], k) for k, v in a.items() if re.fullmatch(pat, k)]
    return max(c)[2] if c else None
print(",".join(x for x in (best(r"lbx?_w[\d.]+_t[\d.]+"), best(r"bf\d+")) if x))
PY
)
echo "escolhidos no dev48: $PICKS $(date +%T)" | tee data/e1_picks.txt
A1=m2a_a1_062959_102
WSP_UNLOCK_TEST=1 py scripts/gen_eval.py t12k 32 12288 base,astra,$A1,$PICKS --np 16
py scripts/math_eval.py math48 48 12288 base,astra,$A1,$PICKS --np 16
scripts/queue_confirm.sh a2 r3
scripts/llama_serve.sh stop || true
echo "fim $(date +%T)"
