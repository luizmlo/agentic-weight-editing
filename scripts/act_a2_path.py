"""Caminho da a2: efeito de cada neurônio (sozinho / todos menos um) em log p("Wait") e ativações dos neurônios dentro
do modelo editado (pré-registro: CADERNO 11/11 11h45). GPU fp16.
uso: uv run --no-sync python scripts/act_a2_path.py [N_TEXTOS=12] [MAX_TOK=1500]  -> data/act_a2_path.json
"""
import json
import sys

import torch
from datasets import load_dataset

from wsp import m2a, modelio, tf
from wsp.evalrun import ENDTHINK_ID, WAIT_IDS, WAIT_LC_IDS
from wsp.paths import DATA

NT, MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 12, int(sys.argv[2]) if len(sys.argv) > 2 else 1500
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
model, tok = modelio.load(dtype=torch.float16, device="cuda")

cid = json.loads((DATA / "m2a/runs/a2/final.json").read_text())["ranked"][0]
e = torch.load(DATA / "m2a" / "runs" / "a2" / "done" / f"{cid}.pt", weights_only=False)
A2 = e["edits"] if isinstance(e, dict) and "edits" in e else e
layer = lambda ed: int(ed["tensor"].split("layers.")[1].split(".")[0])
SINGLE = {}  # (L, c) -> edição de um neurônio só
for ed in A2:
    if ed["cols"] is not None:
        for c in ed["cols"].tolist():
            SINGLE[(layer(ed), c)] = {**ed, "cols": torch.tensor([c])}
REST = [ed for ed in A2 if ed["cols"] is None]
name = lambda k: f"L{k[0]}n{k[1]}"
VARIANTS = {"a2": A2, "neurons": list(SINGLE.values())}
for k in SINGLE:
    VARIANTS[f"only_{name(k)}"] = [SINGLE[k]]
    VARIANTS[f"without_{name(k)}"] = [v for kk, v in SINGLE.items() if kk != k]
VARIANTS["only_L16"] = [v for k, v in SINGLE.items() if k[0] == 16]
VARIANTS["only_L16_L17"] = [v for k, v in SINGLE.items() if k[0] in (16, 17)]
ACT_IN = ["base", "a2", "only_L16", "only_L16_L17"]

LAYERS = sorted({L for L, _ in SINGLE})
acts = {}
for L in LAYERS:
    modelio.down_proj(model, L).register_forward_pre_hook(lambda m, inp, L=L: acts.__setitem__(L, inp[0][0].detach().float().cpu()))

ds = load_dataset("openai/gsm8k", "main", split="train")
rows = [json.loads(l) for l in open(DATA / "gen_dev48" / "base.jsonl")]
rows = [r for r in rows if r["cut"] and "</think>" not in r["text"]][:NT]
wait = torch.tensor(WAIT_IDS + WAIT_LC_IDS)

@torch.no_grad()
def logps(ids, chunk=96):
    h = model.model(input_ids=ids[None].to(model.device), use_cache=False).last_hidden_state[0]
    W, C = [], []
    for s_ in range(0, h.shape[0], chunk):
        lg = model.lm_head(h[s_:s_ + chunk]).float().log_softmax(-1)
        W.append(torch.logsumexp(lg[:, wait.to(lg.device)], -1).cpu()); C.append(lg[:, ENDTHINK_ID].cpu())
    return torch.cat(W), torch.cat(C)

D = {v: [] for v in VARIANTS}
ACT = {m: {name(k): [] for k in SINGLE} for m in ACT_IN}
npre = 0
for i, r in enumerate(rows):
    ids, npr = tf.build_ids(tok, [{"role": "user", "content": ds[r["i"]]["question"] + SUFFIX}], r["text"], MAXT)
    pos = torch.arange(npr, len(ids) - 1)
    pre = pos[torch.isin(ids[pos + 1], wait)]
    npre += len(pre)
    bw, _ = logps(ids)
    for k in SINGLE:
        ACT["base"][name(k)] += acts[k[0]][:, k[1]][pre].tolist()
    for v, eds in VARIANTS.items():
        undo, _ = m2a.apply(model, eds)
        w, _ = logps(ids); undo()
        D[v] += (w - bw)[pre].tolist()
        if v in ACT:
            for k in SINGLE:
                ACT[v][name(k)] += acts[k[0]][:, k[1]][pre].tolist()
    print(f"texto {i + 1}/{len(rows)}: {len(pre)} antes de Wait", flush=True)

mean = lambda x: sum(x) / max(len(x), 1)
out = {"n_texts": len(rows), "max_tok": MAXT, "n_prewait": npre, "cid": cid,
       "dwait_pre": {v: mean(x) for v, x in D.items()}, "neurons": {}}
for k, ed in SINGLE.items():
    u = ed["u"].float(); u = u / u.norm()
    cu = float(u @ modelio.down_proj(model, k[0]).weight[:, k[1]].float().cpu())
    a = float(ed["alpha"])
    o = {"alpha": a, "u_dot_w": cu}
    for m in ACT_IN:
        x = ACT[m][name(k)]
        o[f"act_{m}"] = mean(x); o[f"frac_neg_{m}"] = mean([t < 0 for t in x])
    o["write_base"] = o["act_base"] * cu
    o["write_in_a2"] = o["act_a2"] * cu * (1 - a)
    out["neurons"][name(k)] = o
(DATA / "act_a2_path.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
