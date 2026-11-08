"""M→A, fase M: atlas de ativações do modelo BASE para o agente (data/m2a/atlas.pt).

Médias (fp32) por camada L = 0..23 de três sinais:
  resid    saída da camada L (residual depois da camada L; = hidden_states[L+1])
  mlp_in   entrada do mlp.down_proj da camada L (3584 neurônios)
  attn_in  entrada do out_proj/o_proj da atenção da camada L (2048)
em duas famílias de contextos:
  natural  18 textos do base em loop (MATH-500, amostras publicadas). Regiões por texto:
           loop = [onset, onset+256), preloop = [onset-1024, onset-64) sem 8-gramas repetidos.
  branch   12 pares casados (busca): continuação repetitiva vs produtiva, sob 4 system prompts
           (none / anti_rep / placebo / concise) — a parte de "mapeamento por prompt" (H4).
"""

import json

import torch
from datasets import load_dataset

from wsp import m2a, modelio, probes, tf
from wsp.paths import EDITED_REPO, snapshot

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SYSTEMS = {
    "none": None,
    "anti_rep": "Think efficiently. Never repeat a sentence or a calculation you have already written. "
    "If you notice that you are repeating yourself, stop and write the final answer.",
    "placebo": "You are a helpful assistant. Read the user's question with care and answer it in a "
    "clear, friendly and well organized way, using plain language.",
    "concise": "Be concise. Solve the problem directly, check it once, and stop as soon as you have the answer.",
}
CASES = probes.ASTRA_CASES + probes.OWN_CASES

model, tok = modelio.load()
layers = modelio.layers(model)
MASKS, STORE = {}, {}


def mk(kind, L, from_input):
    def hook(mod, inp, out):
        x = inp[0] if from_input else (out[0] if isinstance(out, tuple) else out)
        x = x[0]
        for name, m in MASKS.items():
            STORE[(kind, L, name)] = x[m.to(x.device)].float().mean(0).cpu()
    return hook


for L, layer in enumerate(layers):
    layer.register_forward_hook(mk("resid", L, False))
    layer.mlp.down_proj.register_forward_hook(mk("mlp_in", L, True))
    attn_out = layer.self_attn.o_proj if L in m2a.FULL_ATTN else layer.linear_attn.out_proj
    attn_out.register_forward_hook(mk("attn_in", L, True))


@torch.no_grad()
def run(ids, masks):
    MASKS.clear()
    MASKS.update(masks)
    STORE.clear()
    model.model(input_ids=ids.to(model.device)[None], use_cache=False)
    return {k: {n: torch.stack([STORE[(k, L, n)] for L in range(24)]) for n in masks}
            for k in ("resid", "mlp_in", "attn_in")}


def stack(rows):
    return {k: {n: torch.stack([r[k][n] for r in rows]) for n in rows[0][k]} for k in rows[0]}


# --- natural
probs = {r["unique_id"]: r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
texts = [json.loads(l) for l in open(snapshot(EDITED_REPO) / "eval" / "samples_base_math500.jsonl")]
nat, onsets, nat_ids = [], [], []
for r in texts:
    ids, npr = tf.build_ids(tok, [{"role": "user", "content": probs[r["id"]] + SUFFIX}], r["text"])
    o = probes.onset(ids[npr:].tolist())
    if o is None or o < 1100 or o > 10000:
        continue
    o += npr
    ids = ids[: o + 256]
    rep = tf.dup_ngram_mask(ids.tolist())
    loop = torch.zeros(len(ids), dtype=torch.bool)
    loop[o:] = True
    pre = torch.zeros(len(ids), dtype=torch.bool)
    pre[max(npr, o - 1024):o - 64] = True
    pre &= ~rep
    nat.append(run(ids, {"loop": loop, "preloop": pre}))
    onsets.append(o - npr)
    nat_ids.append(r["id"])
    print("natural", r["id"], o - npr, flush=True)

# --- branch, sob cada system prompt
branch = {}
for sname, sysmsg in SYSTEMS.items():
    rows = {"repetitive": [], "productive": []}
    for it in probes.branch_items(tok, CASES, system=sysmsg):
        tmask = torch.cat([torch.zeros(1, dtype=torch.bool), it["mask"]])  # posições de token da continuação
        rows[it["cond"]].append(run(it["ids"], {"x": tmask}))
    branch[sname] = {k: {"rep": torch.stack([r[k]["x"] for r in rows["repetitive"]]),
                         "prod": torch.stack([r[k]["x"] for r in rows["productive"]])}
                     for k in ("resid", "mlp_in", "attn_in")}
    print("branch", sname, flush=True)

atlas = {
    "README": __doc__,
    "natural": {**stack(nat), "onsets": onsets, "ids": nat_ids},
    "branch": branch,
    "branch_cases": CASES,
    "systems": SYSTEMS,
    "shapes": "natural[sinal][região] = [n_textos, 24, dim]; branch[system][sinal][rep|prod] = [12 casos, 24, dim]",
}
m2a.M2A.mkdir(parents=True, exist_ok=True)
torch.save(atlas, m2a.ATLAS)
print("atlas ok", {k: tuple(v["loop"].shape) for k, v in atlas["natural"].items() if isinstance(v, dict)})
