"""PoC-8a: fatorial direção × regra de colunas, e as métricas teacher-forced de cada célula.

Direções (unitárias, residual 16/20 = saída das camadas 15/19):
  astra   u publicado (PoC-1)
  md      diferença de médias do PoC-4/7 (8-gramas repetidos vs não, janela 512–4096: em geral ANTES do loop)
  mdloop  diferença de médias "região do loop" [onset, onset+256) vs "produtivo pré-loop" [onset-1024, onset-64)
Regras de colunas (64 neurônios do down_proj por camada):
  astra     as colunas do Astra
  topw      top-64 de |u^T W|                       (regra do PoC-7)
  contrast  top-64 de |Δa ⊙ u^T W|, Δa = ativação média do neurônio no loop - no pré-loop (regra do Astra,
            segundo os logs: "~91%/98% of repetitive-minus-productive writing contrast")
Edição: W <- W - 6 u (mask ⊙ u^T W), como o Astra publicado. Controle: random/topw (PoC-7).

Métricas (antes de gerar — funcionam como previsão registrada para o PoC-8b):
  branch_astra / branch_own   pares casados (wsp.probes): Δlogp repetitivo, produtivo
  onset                       textos naturais: Δlogp em [onset, onset+64) e em controle produtivo pré-loop
  neutral                     WikiText-2, 8×1024 tokens
Saídas: data/poc8_cells.pt, data/poc8_cells.json
"""

import json

import torch
from datasets import load_dataset

from wsp import modelio, probes, tf
from wsp.paths import DATA, EDITED_REPO, snapshot

K, ALPHA, N_ONSET = 64, 6.0, 8
LAYER_HID = {15: 16, 19: 20}
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

probs = {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
texts = [json.loads(l) for l in open(snapshot(EDITED_REPO) / "eval" / "samples_base_math500.jsonl")]
model, tok = modelio.load()
astra = torch.load(DATA / "edit_dirs.pt")
p7 = torch.load(DATA / "poc7_dirs.pt")
report = {"k": K, "alpha": ALPHA}

# --- textos naturais: onset do loop, recorte até onset+256
items = []
for r in texts:
    ids, npr = tf.build_ids(tok, [{"role": "user", "content": probs[r["id"]] + SUFFIX}], r["text"])
    o = probes.onset(ids[npr:].tolist())
    if o is None or o < 1100 or o > 10000:  # precisa de trecho pré-loop; >10k não cabe bem na 1660
        continue
    o += npr
    items.append({"ids": ids[: o + 256], "o": o, "npr": npr})
items.sort(key=lambda x: x["o"])
report["onsets"] = [x["o"] - x["npr"] for x in items]

# --- passo 1: modelo base, ativações do MLP (entrada do down_proj) e residual em loop vs pré-loop
acts, hooks = {}, []
for L in LAYER_HID:
    hooks.append(modelio.down_proj(model, L).register_forward_hook(
        lambda m, inp, out, L=L: acts.__setitem__(L, inp[0][0].float().cpu())))
da = {L: [] for L in LAYER_HID}
dh = {L: [] for L in LAYER_HID}
for x in items:
    ids, o = x["ids"], x["o"]
    rep_m = tf.dup_ngram_mask(ids.tolist())
    loop = torch.zeros(len(ids), dtype=torch.bool)
    loop[o:o + 256] = True
    prod = torch.zeros(len(ids), dtype=torch.bool)
    prod[max(x["npr"], o - 1024):o - 64] = True
    prod &= ~rep_m
    x["loop"], x["prod"] = loop, prod
    out = tf.read(model, ids, tuple(LAYER_HID.values()))
    x["base_lp"] = out["logp"]
    for L, h in LAYER_HID.items():
        da[L].append(acts[L][loop].mean(0) - acts[L][prod].mean(0))
        dh[L].append(out["hidden"][h][loop].mean(0) - out["hidden"][h][prod].mean(0))
    print("onset", x["o"] - x["npr"], int(prod.sum()), flush=True)
for h in hooks:
    h.remove()
da = {L: torch.stack(v).mean(0) for L, v in da.items()}

dirs = {"astra": {L: astra[L]["u"].float() for L in LAYER_HID},
        "md": {L: p7["meandiff"][L]["u"].float() for L in LAYER_HID},
        "mdloop": {L: torch.stack(v).mean(0) for L, v in dh.items()}}
dirs = {k: {L: u / u.norm() for L, u in v.items()} for k, v in dirs.items()}


def cols_for(rule, u, L):
    if rule == "astra":
        return astra[L]["cols"].sort().values
    W = modelio.down_proj(model, L).weight.float().cpu()
    s = u @ W
    if rule == "contrast":
        s = s * da[L]
    return s.abs().topk(K).indices.sort().values


cells = {}
for d, us in dirs.items():
    for rule in ("astra", "topw", "contrast"):
        cells[f"{d}_{rule}"] = {L: {"u": us[L], "cols": cols_for(rule, us[L], L), "alpha": ALPHA,
                                    "tensor": astra[L]["tensor"]} for L in LAYER_HID}
cells["random_topw"] = p7["random"]

report["dir_cos"] = {f"{a}~{b}_L{L}": abs(torch.dot(dirs[a][L], dirs[b][L]).item())
                     for a, b in (("astra", "md"), ("astra", "mdloop"), ("md", "mdloop")) for L in LAYER_HID}
report["cols_overlap_with_astra"] = {
    c: {L: len(set(e[L]["cols"].tolist()) & set(astra[L]["cols"].tolist())) / K for L in LAYER_HID}
    for c, e in cells.items()}
print(json.dumps({k: report[k] for k in ("dir_cos", "cols_overlap_with_astra")}, indent=1), flush=True)
torch.save(cells, DATA / "poc8_cells.pt")

# --- passo 2: métricas por célula
br = {"astra": probes.branch_items(tok, probes.ASTRA_CASES), "own": probes.branch_items(tok, probes.OWN_CASES)}
for v in br.values():
    for it in v:
        it["base_lp"] = tf.read(model, it["ids"])["logp"]
wt = "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
wt_ids = tok(wt, add_special_tokens=False)["input_ids"]
neutral = [torch.tensor(wt_ids[j * 1024:(j + 1) * 1024]) for j in range(8)]
neutral_base = [tf.read(model, x)["logp"] for x in neutral]
on_items = items[:N_ONSET]


def mean(xs):
    return sum(xs) / len(xs)


report["cells"] = {}
for name, e in cells.items():
    undo = modelio.apply_published_edit(model, e)
    m = {}
    for key, its in br.items():
        for cond in ("repetitive", "productive"):
            ds = [(tf.read(model, it["ids"])["logp"] - it["base_lp"])[it["mask"]].mean().item()
                  for it in its if it["cond"] == cond]
            m[f"branch_{key}_{cond[:3]}"] = mean(ds)
        m[f"branch_{key}_sel"] = m[f"branch_{key}_rep"] - m[f"branch_{key}_pro"]
    on, pre = [], []
    for x in on_items:
        dl = tf.read(model, x["ids"])["logp"] - x["base_lp"]
        o = x["o"]
        on.append(dl[o - 1:o + 63].mean().item())   # logp[t-1] prevê o token t
        pre.append(dl[x["prod"][1:]].mean().item())
    m["onset64"], m["preloop_prod"] = mean(on), mean(pre)
    m["onset_sel"] = m["onset64"] - m["preloop_prod"]
    m["neutral"] = mean([(tf.read(model, x)["logp"] - b).mean().item() for x, b in zip(neutral, neutral_base)])
    undo()
    report["cells"][name] = m
    print(name, json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
    (DATA / "poc8_cells.json").write_text(json.dumps(report, indent=1))
