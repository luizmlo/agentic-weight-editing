"""Sinal das ativações dos neurônios de "Wait" e efeito de a1 e a2 em p("Wait") (GPU fp16, os kernels DeltaNet não rodam em CPU; pré-registro CADERNO 10/11 10h55).

Para textos em loop do modelo base (dev48, cortados no teto), mede:
  1) a ativação (entrada do down_proj) de cada neurônio editado, em todas as posições do raciocínio e nas posições
     imediatamente antes de um token "Wait";
  2) a escrita direta do neurônio na direção u da edição: ativação × (uᵀw), no base e depois da edição (× (1−α));
  3) teacher-forced: Δ log p("Wait") e Δ log p("</think>") de cada variante de edição contra o base, nas posições antes
     de "Wait" e em todas as posições do raciocínio.
uso: uv run --no-sync python scripts/act_sign_check.py [N_TEXTOS=8] [MAX_TOK=1200]  -> data/act_sign_check.json
"""
import json
import sys

import torch
from datasets import load_dataset

from wsp import m2a, modelio, tf
from wsp.evalrun import ENDTHINK_ID, WAIT_IDS, WAIT_LC_IDS
from wsp.paths import DATA

NT, MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 8, int(sys.argv[2]) if len(sys.argv) > 2 else 1200
torch.set_num_threads(6)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
model, tok = modelio.load(dtype=torch.float16, device="cuda")

def load_edits(run, cid):
    e = torch.load(DATA / "m2a" / "runs" / run / "done" / f"{cid}.pt", weights_only=False)
    return e["edits"] if isinstance(e, dict) and "edits" in e else e

A1 = load_edits("a1", "062959_102")
A1P = load_edits("a1", "074537_659")
A2 = load_edits("a2", json.loads((DATA / "m2a/runs/a2/final.json").read_text())["ranked"][0])
neur = lambda eds: [e for e in eds if e["cols"] is not None]
rest = lambda eds: [e for e in eds if e["cols"] is None]
VARIANTS = {"a1": A1, "a1_neurons": neur(A1), "a1_rest": rest(A1), "a1_pick1": A1P,
            "a2": A2, "a2_neurons": neur(A2), "a2_rest": rest(A2)}

# neurônios (camada, coluna) -> lista de (rótulo da edição, u, alpha)
NEURONS = {}
for lab, eds in (("a1", A1), ("a2", A2)):
    for e in neur(eds):
        L = int(e["tensor"].split("layers.")[1].split(".")[0])
        for c in e["cols"].tolist():
            NEURONS.setdefault((L, c), []).append((lab, e["u"].float(), float(e["alpha"])))
LAYERS = sorted({L for L, _ in NEURONS})
acts = {}
hooks = [modelio.down_proj(model, L).register_forward_pre_hook(lambda m, inp, L=L: acts.__setitem__(L, inp[0][0].detach().float().cpu()))
         for L in LAYERS]

ds = load_dataset("openai/gsm8k", "main", split="train")
rows = [json.loads(l) for l in open(DATA / "gen_dev48" / "base.jsonl")]
rows = [r for r in rows if r["cut"] and "</think>" not in r["text"]][:NT]
wait = torch.tensor(WAIT_IDS + WAIT_LC_IDS)

@torch.no_grad()
def logps(ids, chunk=96):
    h = model.model(input_ids=ids[None].to(model.device), use_cache=False).last_hidden_state[0]
    W, C = [], []
    for s_ in range(0, h.shape[0], chunk):  # logits em blocos: a GPU é dividida com a fila de avaliação
        lg = model.lm_head(h[s_:s_ + chunk]).float().log_softmax(-1)
        W.append(torch.logsumexp(lg[:, wait.to(lg.device)], -1).cpu()); C.append(lg[:, ENDTHINK_ID].cpu())
    return torch.cat(W), torch.cat(C)

S = {"neurons": {f"L{L}n{c}": {"act_all": [], "act_prewait": []} for L, c in NEURONS}, "variants": {v: {"dwait_pre": [], "dwait_all": [], "dclose_all": []} for v in VARIANTS}, "n_prewait": 0, "n_pos": 0}
for k, r in enumerate(rows):
    ids, npr = tf.build_ids(tok, [{"role": "user", "content": ds[r["i"]]["question"] + SUFFIX}], r["text"], MAXT)
    pos = torch.arange(npr, len(ids) - 1)                      # posições do raciocínio (prevê o token seguinte)
    pre = pos[torch.isin(ids[pos + 1], wait)]                  # posições cujo próximo token é "Wait"
    bw, bc = logps(ids)
    for (L, c) in NEURONS:
        a = acts[L][:, c]
        S["neurons"][f"L{L}n{c}"]["act_all"] += a[pos].tolist(); S["neurons"][f"L{L}n{c}"]["act_prewait"] += a[pre].tolist()
    S["n_prewait"] += len(pre); S["n_pos"] += len(pos)
    for v, eds in VARIANTS.items():
        undo, _ = m2a.apply(model, eds)
        w, c_ = logps(ids); undo()
        S["variants"][v]["dwait_pre"] += (w - bw)[pre].tolist(); S["variants"][v]["dwait_all"] += (w - bw)[pos].tolist()
        S["variants"][v]["dclose_all"] += (c_ - bc)[pos].tolist()
    print(f"texto {k + 1}/{len(rows)}: {len(pos)} posições, {len(pre)} antes de Wait", flush=True)

mean = lambda x: sum(x) / max(len(x), 1)
out = {"n_texts": len(rows), "max_tok": MAXT, "n_pos": S["n_pos"], "n_prewait": S["n_prewait"], "neurons": {}, "variants": {}}
for (L, c), lst in NEURONS.items():
    d = S["neurons"][f"L{L}n{c}"]; w = modelio.down_proj(model, L).weight[:, c].float().cpu()
    o = {"act_mean_all": mean(d["act_all"]), "act_mean_prewait": mean(d["act_prewait"]),
         "frac_neg_all": mean([x < 0 for x in d["act_all"]]), "frac_neg_prewait": mean([x < 0 for x in d["act_prewait"]])}
    for lab, u, alpha in lst:
        u = u / u.norm(); cu = float(u @ w)
        o[lab] = {"alpha": alpha, "u_dot_w": cu, "write_on_u_prewait_base": o["act_mean_prewait"] * cu,
                  "write_on_u_prewait_edited": o["act_mean_prewait"] * cu * (1 - alpha)}
    out["neurons"][f"L{L}n{c}"] = o
for v, d in S["variants"].items():
    out["variants"][v] = {k: mean(x) for k, x in d.items()}
(DATA / "act_sign_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
