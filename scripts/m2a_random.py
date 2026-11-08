"""M→A, braço R (controle): busca aleatória com o MESMO avaliador e orçamento, sem atlas e sem agente.

Cada candidato: 1–2 edições rank-1, camada e tipo de matriz uniformes, u ~ N(0, I) normalizado,
cols = None ou 64 colunas aleatórias, alpha uniforme em [-10, 10]. Guarda o melhor por branch_rep
com os limites do brief (branch_prod >= -0.02, close <= 1, neutral >= -0.01).

uso: scripts/run_bg.sh m2a-rnd scripts/m2a_random.py rnd 60
"""
import json
import random
import sys

import torch

from wsp import m2a

run, n = sys.argv[1], int(sys.argv[2])
rng = random.Random(0)
torch.manual_seed(0)
kinds = [(L, "mlp") for L in range(24)] + [(L, "self_attn" if L in m2a.FULL_ATTN else "linear_attn") for L in range(24)]
best = None
for i in range(n):
    edits = []
    for _ in range(rng.choice([1, 2])):
        L, k = rng.choice(kinds)
        dim = m2a.IN_DIM[m2a.KIND[k]]
        cols = None if rng.random() < 0.5 else rng.sample(range(dim), 64)
        edits.append({"tensor": m2a.tensor_name(L, k), "u": torch.randn(1024), "cols": cols,
                      "alpha": round(rng.uniform(-10, 10), 2)})
    r = m2a.submit(run, edits, note=f"random {i}")
    if "error" in r:
        print(i, r["error"], flush=True)
        break
    ok = r["branch_prod"] >= -0.02 and r["close"] <= 1 and r["neutral"] >= -0.01
    if ok and (best is None or r["branch_rep"] < best["branch_rep"]):
        best = r
    print(i, r["id"], round(r["branch_rep"], 4), round(r["branch_prod"], 4), round(r["neutral"], 4), ok, flush=True)
top = sorted([h for h in m2a.history(run) if "branch_rep" in h and h["branch_prod"] >= -0.02 and h["close"] <= 1
              and h["neutral"] >= -0.01], key=lambda h: h["branch_rep"])[:3]
(m2a.RUNS / run / "final.json").write_text(json.dumps({"ranked": [h["id"] for h in top], "rationale": "random search"}))
print("melhor", json.dumps(best)[:500])
