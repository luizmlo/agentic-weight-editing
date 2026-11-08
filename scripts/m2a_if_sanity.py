"""Sanidade do avaliador IF (run 'sanity'): edição aleatória, a edição a1 (do defeito de loops) e um teste por geração.

Espera o daemon wsp-m2a-if. Imprime as métricas de cada candidato.
"""
import torch

from wsp import m2a_if as m
from wsp.paths import DATA

g = torch.Generator().manual_seed(0)
rand = [{"tensor": m.tensor_name(12, "mlp"), "u": torch.randn(1024, generator=g),
         "cols": torch.randperm(3584, generator=g)[:64].tolist(), "alpha": 6.0}]
a1 = torch.load(DATA / "m2a" / "runs" / "a1" / "done" / "062959_102.pt")["edits"]
a1 = [{"tensor": e["tensor"], "u": e["u"], "cols": None if e["cols"] is None else e["cols"].tolist(), "alpha": e["alpha"]}
      for e in a1]
for name, ed in (("random", rand), ("a1_loops", a1)):
    r = m.submit("sanity", ed, note=f"sanity: {name}")
    print(name, {k: r.get(k) for k in ("if_margin", "ctrl_margin", "if_sel", "neutral", "per_constraint", "secs", "error")}, flush=True)
    last = r
print("base margin:", last.get("base_margin"))
r = m.submit_gen("sanity", last["id"])
print("gen a1:", {k: r.get(k) for k in ("prompt_strict", "inst_strict", "cut", "base_prompt_strict", "base_cut",
                                       "rescues", "losses", "secs", "error")}, flush=True)
print("exemplo de falha:", (r.get("fail_examples") or [None])[0])
