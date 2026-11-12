"""Monta um repositório pronto para o Hugging Face com o modelo editado (pesos inteiros, GGUF e as edições).

uso: hf_package.py <nome> <edições.pt> <gguf>     -> data/hf/<nome>/
ex.: hf_package.py Qwen3.5-0.8B-LoopFix data/m2a/variants/a2_red.pt data/gguf/m2a_a2_red.gguf

Conteúdo:
  model.safetensors (+ config, tokenizer, chat template)  modelo inteiro editado, bf16, carrega no transformers
  <nome>-F16.gguf                                         o mesmo modelo para llama.cpp / Ollama / LM Studio
  edits/edits.safetensors + edits/edits.json + edits/apply_edits.py
                                                          as edições rank-1, sem pickle, para aplicar ao original
  LICENSE (Apache 2.0, do Qwen) + NOTICE (o que mudou)    exigência da seção 4(b) da licença
O model card (README.md) é escrito à parte (docs/hf/<nome>.md) e copiado para cá.

A conta é a mesma de poc7_build.py: W <- W - alpha * u (mask * u^T W), em fp32, gravada no dtype original.
Depois de montar, confere que os tensores editados batem com o GGUF usado em todas as avaliações.
"""

import json
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from wsp.paths import BASE_REPO, BASE_REV, DATA, ROOT, safetensors_file, snapshot

name, edits_pt, gguf = sys.argv[1], sys.argv[2], sys.argv[3]
edits = torch.load(edits_pt, weights_only=False)["edits"]
out = DATA / "hf" / name
shutil.rmtree(out, ignore_errors=True)
(out / "edits").mkdir(parents=True)

src = snapshot(BASE_REPO, BASE_REV)
shard = safetensors_file(BASE_REPO, BASE_REV)
for f in src.iterdir():
    if f.name in (shard.name, "model.safetensors.index.json", "README.md"):
        continue
    shutil.copy(f, out / f.name)

with safe_open(shard, "pt") as fh:
    meta = dict(fh.metadata() or {})
sd = load_file(shard)
tensors, info = {}, []
for i, e in enumerate(edits):
    key = e["tensor"]
    W = sd[key].float()
    u = e["u"].float()
    u = u / u.norm()
    proj = u @ W
    mask = torch.zeros_like(proj)
    cols = None if e["cols"] is None else torch.as_tensor(e["cols"]).long()
    mask[cols if cols is not None else slice(None)] = 1.0
    W2 = W - float(e["alpha"]) * torch.outer(u, proj * mask)
    sd[key] = W2.to(sd[key].dtype).contiguous()
    tensors[f"edit{i}.u"] = u.contiguous()
    if cols is not None:
        tensors[f"edit{i}.cols"] = cols.contiguous()
    info.append({"index": i, "tensor": key, "alpha": float(e["alpha"]),
                 "cols": None if cols is None else cols.tolist(),
                 "rel_change": ((W2 - W).norm() / W.norm()).item()})
meta["modified"] = f"rank-1 weight edits ({len(edits)}) applied to {BASE_REPO}@{BASE_REV}; see NOTICE"
save_file(sd, out / "model.safetensors", metadata=meta)
save_file(tensors, out / "edits" / "edits.safetensors")
(out / "edits" / "edits.json").write_text(json.dumps(
    {"base_model": BASE_REPO, "base_revision": BASE_REV, "source": str(edits_pt),
     "formula": "W <- W - alpha * outer(u, mask * (u @ W)), in fp32, cast back to the original dtype",
     "edits": info}, indent=1))
shutil.copy(ROOT / "scripts" / "hf_apply_edits.py", out / "edits" / "apply_edits.py")
shutil.copy(gguf, out / f"{name}-F16.gguf")
(out / "NOTICE").write_text(
    f"{name}\n\n"
    "This model is a modified version of Qwen3.5-0.8B (https://huggingface.co/Qwen/Qwen3.5-0.8B),\n"
    "Copyright 2026 Alibaba Cloud, licensed under the Apache License, Version 2.0 (see LICENSE).\n\n"
    f"Modification (2026, Luiz Melo): {len(edits)} rank-1 edits to the following weight matrices of\n"
    "model.safetensors; every other tensor and every other file is unchanged.\n"
    + "".join(f"  - {e['tensor']} (alpha = {e['alpha']:+g}, "
              f"{'all columns' if e['cols'] is None else 'columns ' + ', '.join(map(str, e['cols']))})\n" for e in info)
    + "The edits are listed in edits/edits.json and can be re-applied with edits/apply_edits.py.\n")

# conferência: os tensores editados devem bater com os do GGUF avaliado (f16)
from gguf import GGUFReader  # noqa: E402

r = GGUFReader(str(gguf))
gg = {t.name: t for t in r.tensors}
for e in info:
    L = int(e["tensor"].split("layers.")[1].split(".")[0])
    gname = f"blk.{L}.ffn_down.weight" if "down_proj" in e["tensor"] else None
    if gname is None:
        cands = [n for n in gg if n.startswith(f"blk.{L}.") and ("ssm_out" in n or "attn_output" in n)]
        gname = cands[0] if cands else None
    if gname is None:
        print("aviso: tensor do GGUF não encontrado para", e["tensor"])
        continue
    a = torch.from_numpy(gg[gname].data.copy()).float().reshape(sd[e["tensor"]].shape)
    d = (a - sd[e["tensor"]].float()).abs().max().item()
    print(f"{e['tensor']} vs GGUF {gname}: max |diff| = {d:.2e}")
    assert d < 1e-2, d
print("ok:", out)
