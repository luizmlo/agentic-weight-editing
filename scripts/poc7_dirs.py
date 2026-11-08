"""PoC-7a: transformar a direção recuperada no PoC-4 (diferença de médias) numa edição no formato do Astra.

Método "M" sem acesso ao u do Astra:
  d_l   = média sobre textos de [média(residual | token repetitivo) - média(residual | produtivo)]
          no residual hidden[16] / hidden[20] (saída das camadas 15 / 19), janela [512, 4096)
  cols  = 64 colunas do down_proj da camada com maior |d^T W| (quem mais escreve em d)
  edição W <- W - 6 d (mask ⊙ d^T W)   (mesmo α e mesmo nº de colunas do Astra)
Controle: direção aleatória (semente 0) com o mesmo procedimento de seleção de colunas.

Também: (0) Δlogp em texto neutro (WikiText-2) como medida de dano colateral; (1) checa se a regra "top-64 por |u^T W|" reproduz as colunas que o Astra escolheu;
(2) Δlogp teacher-forced (repetitivo vs produtivo) das três edições nos mesmos textos.
Saída: data/poc7_dirs.pt  {método: {camada: {u, cols, alpha, tensor}}}  e poc7_dirs.json
"""

import json

import torch
from datasets import load_dataset

from wsp import modelio, tf
from wsp.paths import DATA, EDITED_REPO, snapshot

W0, W1, K, ALPHA = 512, 4096, 64, 6.0
LAYER_HID = {15: 16, 19: 20}
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

probs = {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
texts = [json.loads(l) for l in open(snapshot(EDITED_REPO) / "eval" / "samples_base_math500.jsonl")]
model, tok = modelio.load()
astra = torch.load(DATA / "edit_dirs.pt")

# --- passo 1: leitura do modelo base, direção por diferença de médias
cache, dsum = [], {l: [] for l in LAYER_HID}
for i, r in enumerate(texts):
    ids, npr = tf.build_ids(tok, [{"role": "user", "content": probs[r["id"]] + SUFFIX}], r["text"], W1)
    rep = tf.dup_ngram_mask(ids.tolist())
    win = torch.zeros(len(ids), dtype=torch.bool)
    win[npr + W0: npr + W1] = True
    rep, prod = rep & win, (~rep) & win
    if rep.sum() < 50 or prod.sum() < 50:
        continue
    o = tf.read(model, ids, tuple(LAYER_HID.values()))
    for L, h in LAYER_HID.items():
        H = o["hidden"][h]
        dsum[L].append(H[rep].mean(0) - H[prod].mean(0))
    cache.append((ids, rep[1:], prod[1:], o["logp"]))
    print("leitura", i, int(rep.sum()), int(prod.sum()), flush=True)

g = torch.Generator().manual_seed(0)
raw = {"meandiff": {L: torch.stack(v).mean(0) for L, v in dsum.items()},
       "random": {L: torch.randn(1024, generator=g) for L in LAYER_HID}}

report = {"n_texts": len(cache), "window": [W0, W1], "k": K, "alpha": ALPHA, "methods": {}}
edits = {"astra": {L: astra[L] for L in LAYER_HID}}
for name, vs in raw.items():
    edits[name] = {}
    for L, v in vs.items():
        u = v / v.norm()
        W = modelio.down_proj(model, L).weight.float().cpu()
        cols = (u @ W).abs().topk(K).indices.sort().values
        edits[name][L] = {"u": u, "cols": cols, "alpha": ALPHA, "tensor": astra[L]["tensor"]}

# regra de seleção do Astra: top-K por |u^T W| reproduz as colunas dele?
for L in LAYER_HID:
    W = modelio.down_proj(model, L).weight.float().cpu()
    ua = astra[L]["u"].float() / astra[L]["u"].float().norm()
    top = set((ua @ W).abs().topk(K).indices.tolist())
    report[f"astra_cols_top{K}_rule_overlap_L{L}"] = len(top & set(astra[L]["cols"].tolist())) / K

for name, e in edits.items():
    m = {}
    for L in LAYER_HID:
        ua = astra[L]["u"].float()
        m[f"cos_u_astra_L{L}"] = torch.nn.functional.cosine_similarity(e[L]["u"].float(), ua, dim=0).item()
        m[f"cols_overlap_astra_L{L}"] = len(set(e[L]["cols"].tolist()) & set(astra[L]["cols"].tolist())) / K
    report["methods"][name] = m

# texto neutro (efeito colateral fora do domínio): 16 blocos de 1024 tokens do WikiText-2 test, sem template
wt = "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
wt_ids = tok(wt, add_special_tokens=False)["input_ids"]
neutral = [torch.tensor(wt_ids[j * 1024:(j + 1) * 1024]) for j in range(16)]
neutral_base = [tf.read(model, x)["logp"] for x in neutral]
report["neutral_base_nll"] = -sum(x.mean().item() for x in neutral_base) / len(neutral_base)

# --- passo 2: Δlogp teacher-forced de cada edição (α do próprio dict)
for name, e in edits.items():
    undo = modelio.apply_published_edit(model, e)
    dr, dp = [], []
    for ids, rep_lp, prod_lp, base_lp in cache:
        dl = tf.read(model, ids)["logp"] - base_lp
        dr.append(dl[rep_lp].mean().item())
        dp.append(dl[prod_lp].mean().item())
    dn = [(tf.read(model, x)["logp"] - b).mean().item() for x, b in zip(neutral, neutral_base)]
    undo()
    report["methods"][name]["dlogp_rep"] = sum(dr) / len(dr)
    report["methods"][name]["dlogp_prod"] = sum(dp) / len(dp)
    report["methods"][name]["dlogp_neutral_wikitext"] = sum(dn) / len(dn)
    print(name, json.dumps(report["methods"][name]), flush=True)

torch.save({k: v for k, v in edits.items() if k != "astra"}, DATA / "poc7_dirs.pt")
(DATA / "poc7_dirs.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
