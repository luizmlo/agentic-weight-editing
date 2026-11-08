"""PoC-3/4/5: medições teacher-forced sobre os textos publicados (MATH-500, base em loop).

Rótulo por token (mecânico, sem LLM): "repetitivo" se o 8-grama que termina nele já apareceu antes
no texto; "produtivo" caso contrário. Só tokens do assistente, numa janela [W0, W1) para reduzir o
confundimento por posição.

PoC-3  dose-resposta: Δlogp(editado α - base) em tokens repetitivos vs produtivos, α ∈ {1,2,4,6}.
       Assinatura do Astra (relatório): ≈ -0.10 nats/token repetitivo vs ≈ -0.01 produtivo em α=6.
PoC-4  método trivial: direção = média(residual | repetitivo) - média(residual | produtivo) nas
       camadas 16/20 do residual. Cosseno com u15/u19 (edição do Astra). Acaso ≈ 1/sqrt(1024) ≈ 0.03.
PoC-5  ideia do prompt: mesmo texto lido sob system prompts diferentes (nenhum / anti-repetição /
       placebo). Mede Δlogp nos repetitivos e se o delta de ativação causado pelo prompt aponta na
       direção u.
"""

import json
import sys

import torch
from datasets import load_dataset

from wsp import modelio, tf
from wsp.paths import DATA, EDITED_REPO, snapshot

W0, W1 = 512, int(sys.argv[1]) if len(sys.argv) > 1 else 4096
HID = (16, 20)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SYSTEMS = {
    "none": None,
    "anti_rep": "Think efficiently. Never repeat a sentence or a calculation you have already written. "
    "If you notice that you are repeating yourself, stop and write the final answer.",
    "placebo": "You are a helpful assistant. Read the user's question with care and answer it in a "
    "clear, friendly and well organized way, using plain language.",
}

probs = {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
ev = snapshot(EDITED_REPO) / "eval"
texts = [json.loads(l) for l in open(ev / "samples_base_math500.jsonl")]

model, tok = modelio.load()
dirs = torch.load(DATA / "edit_dirs.pt")
u = {16: dirs[15]["u"].float(), 20: dirs[19]["u"].float()}


def msgs(problem, system):
    m = [{"role": "user", "content": problem + SUFFIX}]
    return ([{"role": "system", "content": system}] if system else []) + m


def masks(ids, n_prompt):
    """máscaras sobre posições de logp (alvo = token t+1) e sobre posições de residual."""
    rep = tf.dup_ngram_mask(ids.tolist())
    win = torch.zeros(len(ids), dtype=torch.bool)
    win[n_prompt + W0 : n_prompt + W1] = True
    return rep & win, (~rep) & win


rows = []
for i, r in enumerate(texts):
    p = probs[r["id"]]
    rec = {"id": r["id"]}
    base_ids, npb = tf.build_ids(tok, msgs(p, None), r["text"], W1)
    rep, prod = masks(base_ids, npb)
    rec["n_rep"], rec["n_prod"] = int(rep.sum()), int(prod.sum())
    if rec["n_rep"] < 50 or rec["n_prod"] < 50:
        print(i, "pulado", rec)
        continue
    # logp: posição t do vetor logp prevê token t+1 -> máscara deslocada
    rep_lp, prod_lp = rep[1:], prod[1:]

    base = tf.read(model, base_ids, HID)
    rec["base"] = {"rep": base["logp"][rep_lp].mean().item(), "prod": base["logp"][prod_lp].mean().item()}

    # PoC-4: direção por diferença de médias
    for l in HID:
        h = base["hidden"][l]
        d = h[rep].mean(0) - h[prod].mean(0)
        rec[f"cos_meandiff_u{l}"] = torch.nn.functional.cosine_similarity(d, u[l], dim=0).item()
        rec[f"dmean_{l}"] = d  # agregado depois

    # PoC-3: dose-resposta
    rec["dose"] = {}
    for a in (1, 2, 4, 6):
        undo = [modelio.apply_rank1(model, L, dirs[L]["u"], a, dirs[L]["cols"]) for L in (15, 19)]
        e = tf.read(model, base_ids)
        for f in reversed(undo):
            f()
        dl = e["logp"] - base["logp"]
        rec["dose"][a] = {"rep": dl[rep_lp].mean().item(), "prod": dl[prod_lp].mean().item()}

    # PoC-5: system prompts (mesmo texto do assistente)
    rec["sys"] = {}
    for name, sysmsg in SYSTEMS.items():
        if sysmsg is None:
            continue
        ids_s, nps = tf.build_ids(tok, msgs(p, sysmsg), r["text"], W1)
        off = nps - npb  # mesmo texto, prompt mais longo
        o = tf.read(model, ids_s, HID)
        lp_s = o["logp"][off:]  # alinhado com base["logp"]
        dl = lp_s - base["logp"]
        s = {"rep": dl[rep_lp].mean().item(), "prod": dl[prod_lp].mean().item()}
        for l in HID:
            # delta de ativação causado pelo prompt nas posições do assistente (janela)
            win = rep | prod
            dh = (o["hidden"][l][off:][win] - base["hidden"][l][win]).mean(0)
            s[f"cos_u{l}"] = torch.nn.functional.cosine_similarity(dh, u[l], dim=0).item()
            s[f"proj_u{l}"] = (dh @ u[l]).item()
            rec[f"dsys_{name}_{l}"] = dh
        rec["sys"][name] = s
    rows.append(rec)
    print(i, rec["n_rep"], rec["n_prod"], {k: round(v, 3) for k, v in rec.items() if k.startswith("cos")},
          {a: round(v["rep"], 3) for a, v in rec["dose"].items()}, flush=True)

# --- agregados
def mean(xs):
    return sum(xs) / len(xs)

agg = {"n_texts": len(rows), "window": [W0, W1]}
agg["base_logp"] = {k: mean([r["base"][k] for r in rows]) for k in ("rep", "prod")}
agg["dose"] = {a: {k: mean([r["dose"][a][k] for r in rows]) for k in ("rep", "prod")} for a in (1, 2, 4, 6)}
for l in HID:
    agg[f"cos_meandiff_u{l}_per_text_mean"] = mean([r[f"cos_meandiff_u{l}"] for r in rows])
    pooled = torch.stack([r[f"dmean_{l}"] for r in rows]).mean(0)
    agg[f"cos_meandiff_u{l}_pooled"] = torch.nn.functional.cosine_similarity(pooled, u[l], dim=0).item()
    rnd = torch.nn.functional.normalize(torch.randn(2000, pooled.numel()), dim=1)
    agg[f"cos_random_abs_mean"] = (rnd @ u[l]).abs().mean().item()
agg["sys"] = {}
for name in ("anti_rep", "placebo"):
    s = {k: mean([r["sys"][name][k] for r in rows]) for k in rows[0]["sys"][name]}
    for l in HID:
        pooled = torch.stack([r[f"dsys_{name}_{l}"] for r in rows]).mean(0)
        s[f"cos_u{l}_pooled"] = torch.nn.functional.cosine_similarity(pooled, u[l], dim=0).item()
    agg["sys"][name] = s
# o delta específico do anti-repetição (descontando o placebo) aponta para u?
for l in HID:
    a = torch.stack([r[f"dsys_anti_rep_{l}"] - r[f"dsys_placebo_{l}"] for r in rows]).mean(0)
    agg[f"cos_antirep_minus_placebo_u{l}"] = torch.nn.functional.cosine_similarity(a, u[l], dim=0).item()

print(json.dumps(agg, indent=1))
keep = [{k: v for k, v in r.items() if not torch.is_tensor(v)} for r in rows]
(DATA / f"poc3_tf_w{W1}.json").write_text(json.dumps({"agg": agg, "rows": keep}, indent=1))
