"""E3 — controle de protocolo: busca INGÊNUA com as mesmas ferramentas e o mesmo orçamento do agente v2
(40 sondas + 6 checagens por geração), sem raciocínio.

Gerador de candidatos (as ferramentas que os agentes vencedores usaram, mas sorteadas):
  token t sorteado entre os 300 mais frequentes nas saídas do base (dev48);
  u = direção de leitura do logit lens, (1 + w_norm) ⊙ E[t];
  camada L uniforme em 12..23, matriz mlp ou atenção (out_proj/o_proj);
  cols = as m colunas com maior |cos(W[:, j], u)|, m ∈ {1, 4, 16, 64, todas};
  alpha uniforme em [-10, 10]; 1–2 edições por candidato.
Fase 1: 40 sondas. Fase 2: as 6 de branch_rep mais negativo dentro dos limites do brief vão para a checagem
por geração. final.json ranqueia pela geração (acc, depois menos cortes).

uso: scripts/run_bg.sh m2a-naive scripts/m2a_naive.py naive
"""
import json
import random
import sys
from collections import Counter

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from wsp import m2a
from wsp.paths import BASE_REPO, BASE_REV, DATA, safetensors_file, snapshot

run = sys.argv[1]
rng = random.Random(1)
F = safetensors_file(BASE_REPO, BASE_REV)
tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))


def T(name):
    with safe_open(F, "pt") as s:
        return s.get_tensor(name).float()


cnt = Counter()
for line in (DATA / "gen_dev48" / "base.jsonl").read_text().splitlines():
    cnt.update(tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"])
pool = [t for t, _ in cnt.most_common(300)]
E = T("model.language_model.embed_tokens.weight")
g = 1 + T("model.language_model.norm.weight")
R = {t: E[t] * g for t in pool}
del E

kinds = [(L, "mlp") for L in range(12, 24)] + [(L, "self_attn" if L in m2a.FULL_ATTN else "linear_attn")
                                               for L in range(12, 24)]


def edit():
    t = rng.choice(pool)
    L, k = rng.choice(kinds)
    u = R[t] / R[t].norm()
    m = rng.choice([1, 4, 16, 64, None])
    cols = None
    if m is not None:
        W = T(m2a.tensor_name(L, k))
        cos = (u @ W) / W.norm(dim=0)
        cols = cos.abs().topk(m).indices.tolist()
    return {"tensor": m2a.tensor_name(L, k), "u": u, "cols": cols, "alpha": round(rng.uniform(-10, 10), 2)}, \
        f"tok={tok.decode([t])!r} L{L} {k} m={m}"


for i in range(40):
    es, desc = zip(*[edit() for _ in range(rng.choice([1, 2]))])
    r = m2a.submit(run, list(es), note="naive " + " | ".join(desc))
    if "error" in r:
        print(i, r["error"], flush=True)
        break
    print(i, r["id"], round(r["branch_rep"], 4), round(r["branch_prod"], 4), round(r["close"], 3),
          round(r["neutral"], 4), "|", " | ".join(desc), flush=True)

ok = [h for h in m2a.history(run) if "branch_rep" in h and h["branch_prod"] >= -0.02 and h["close"] <= 1
      and h["neutral"] >= -0.01]
top = sorted(ok, key=lambda h: h["branch_rep"])[:6]
gens = []
for h in top:
    gr = m2a.submit_gen(run, h["id"])
    print("gen", h["id"], round(h["branch_rep"], 4), {k: gr.get(k) for k in ("acc", "cut", "rescues", "losses", "error")},
          flush=True)
    if "acc" in gr:
        gens.append((gr["acc"], -gr["cut"], h["id"]))
ranked = [c for _, _, c in sorted(gens, reverse=True)][:3] or [h["id"] for h in top[:3]]
(m2a.RUNS / run / "final.json").write_text(json.dumps({"ranked": ranked, "rationale": "naive logit-lens search",
                                                        "gens": gens}))
print("final", ranked, flush=True)
