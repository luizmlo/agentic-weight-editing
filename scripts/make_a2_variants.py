"""Variantes reduzidas da a2 para o teste por geração (pré-registro: CADERNO 11/11 15h15).
Mesmos u e alpha da escolha da a2 (015739_897); só muda quais edições/colunas entram.
  a2_red      = L16n998 (alpha=+10) + L22 linear_attn.out_proj na direção de </think> (alpha=-4)
  a2_l16n998  = só L16n998
uso: uv run --no-sync python scripts/make_a2_variants.py  -> data/m2a/variants/{a2_red,a2_l16n998}.pt
"""
import torch

from wsp.paths import DATA

E = torch.load(DATA / "m2a/runs/a2/done/015739_897.pt", weights_only=False)["edits"]
assert "layers.16.mlp" in E[0]["tensor"] and 998 in E[0]["cols"].tolist()
assert E[3]["cols"] is None and "layers.22.linear_attn" in E[3]["tensor"]
l16 = {**E[0], "cols": torch.tensor([998])}
out = DATA / "m2a/variants"
out.mkdir(parents=True, exist_ok=True)
torch.save({"edits": [l16, E[3]], "note": "a2 reduzida: L16n998 + L22 </think>"}, out / "a2_red.pt")
torch.save({"edits": [l16], "note": "só L16n998 da a2"}, out / "a2_l16n998.pt")
