"""Braço M→A: ferramenta do agente (atlas de ativações + avaliador de edições com orçamento).

O agente NÃO roda o modelo diretamente: lê o atlas (data/m2a/atlas.pt), pode ler os pesos do modelo
base na CPU, e submete edições ao avaliador (daemon scripts/m2a_scorer.py), que debita do orçamento
da run.

Espaço de edições (regras do PLANO, como no paper):
  até MAX_EDITS atualizações rank-1 do lado da SAÍDA em matrizes que escrevem no residual
      W <- W - alpha * u (mask ⊙ u^T W)          u ∈ R^1024 (normalizado aqui), |alpha| <= MAX_ALPHA
  tensores permitidos: model.language_model.layers.{L}.{mlp.down_proj | linear_attn.out_proj |
  self_attn.o_proj}.weight  (L = 0..23; self_attn só nas camadas 3, 7, 11, 15, 19, 23)
  cols: None (todas as colunas) ou lista de índices de coluna (dimensão de entrada da matriz)

Uso pelo agente:
    from wsp import m2a
    r = m2a.submit("run1", [{"tensor": m2a.tensor_name(15, "mlp"), "u": u, "cols": cols, "alpha": 6.0}],
                   note="hipótese ...")
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

from wsp.paths import DATA

M2A = DATA / "m2a"
RUNS = M2A / "runs"
ATLAS = M2A / "atlas.pt"
MAX_EDITS, MAX_ALPHA, D = 4, 10.0, 1024
ALLOWED = re.compile(r"model\.language_model\.layers\.(\d+)\.(mlp\.down_proj|linear_attn\.out_proj|self_attn\.o_proj)\.weight")
FULL_ATTN = (3, 7, 11, 15, 19, 23)
KIND = {"mlp": "mlp.down_proj", "linear_attn": "linear_attn.out_proj", "self_attn": "self_attn.o_proj"}
IN_DIM = {"mlp.down_proj": 3584, "linear_attn.out_proj": 2048, "self_attn.o_proj": 2048}


def tensor_name(layer: int, kind: str) -> str:
    """kind: 'mlp' | 'linear_attn' | 'self_attn'."""
    return f"model.language_model.layers.{layer}.{KIND[kind]}.weight"


def validate(edits: list[dict]) -> list[dict]:
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_EDITS:
        raise ValueError(f"1..{MAX_EDITS} edições")
    out = []
    for e in edits:
        m = ALLOWED.fullmatch(e["tensor"])
        if not m:
            raise ValueError(f"tensor não permitido: {e['tensor']}")
        L, kind = int(m.group(1)), m.group(2)
        bad_attn = (kind == "self_attn.o_proj" and L not in FULL_ATTN) or (kind == "linear_attn.out_proj" and L in FULL_ATTN)
        if not 0 <= L < 24 or bad_attn:
            raise ValueError(f"camada/tipo inexistente: {e['tensor']}")
        u = torch.as_tensor(e["u"], dtype=torch.float32).flatten()
        if u.numel() != D or not torch.isfinite(u).all() or u.norm() == 0:
            raise ValueError("u precisa ser um vetor finito, não nulo, de dimensão 1024")
        a = float(e["alpha"])
        if abs(a) > MAX_ALPHA:
            raise ValueError(f"|alpha| <= {MAX_ALPHA}")
        cols = e.get("cols")
        if cols is not None:
            cols = torch.as_tensor(cols, dtype=torch.long).flatten().unique()
            if cols.numel() == 0 or cols.min() < 0 or cols.max() >= IN_DIM[kind]:
                raise ValueError(f"cols fora de [0, {IN_DIM[kind]})")
        out.append({"tensor": e["tensor"], "u": u / u.norm(), "cols": cols, "alpha": a})
    return out


@torch.no_grad()
def apply(model, edits: list[dict]):
    """Aplica edições validadas no modelo HF (em memória). Devolve (undo, rel_change por edição)."""
    saved, rel = [], []
    for e in edits:
        W = model.model.get_parameter(e["tensor"].replace("model.language_model.", ""))
        saved.append((W, W.detach().clone()))
        Wf = W.float()
        u = e["u"].to(Wf.device)
        proj = u @ Wf
        if e["cols"] is not None:
            mask = torch.zeros_like(proj)
            mask[e["cols"].to(proj.device)] = 1.0
            proj = proj * mask
        delta = e["alpha"] * torch.outer(u, proj)
        rel.append((delta.norm() / Wf.norm()).item())
        W.copy_((Wf - delta).to(W.dtype))

    def undo():
        for W, old in reversed(saved):
            W.copy_(old)

    return undo, rel


def submit(run: str, edits: list[dict], note: str = "", timeout: float = 900) -> dict:
    """Envia um candidato ao avaliador e espera o resultado (debita 1 do orçamento da run)."""
    d = RUNS / run
    if not (d / "config.json").exists():
        raise RuntimeError(f"run {run} não existe (o supervisor cria com config.json)")
    validate(edits)  # falha cedo, sem gastar orçamento
    cid = time.strftime("%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    tmp = d / "inbox" / f".{cid}.tmp"
    torch.save({"edits": edits, "note": note}, tmp)
    tmp.rename(d / "inbox" / f"{cid}.pt")
    out = d / "outbox" / f"{cid}.json"
    t = time.time()
    while not out.exists():
        if time.time() - t > timeout:
            raise TimeoutError(f"sem resposta do avaliador para {cid} (daemon wsp-m2a-scorer está ativo?)")
        time.sleep(1)
    return json.loads(out.read_text())


def submit_gen(run: str, cid: str, timeout: float = 3600) -> dict:
    """Checagem por GERAÇÃO de um candidato já avaliado (id devolvido por submit). Debita 1 do
    orçamento de geração (config.json "gen_budget"). Leva ~5–15 min."""
    d = RUNS / run
    if not (d / "done" / f"{cid}.pt").exists():
        raise RuntimeError(f"candidato {cid} não foi avaliado nesta run (use submit primeiro)")
    gid = "gen_" + time.strftime("%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    tmp = d / "inbox" / f".{gid}.tmp"
    torch.save({"kind": "gen", "cid": cid}, tmp)
    tmp.rename(d / "inbox" / f"{gid}.pt")
    out = d / "outbox" / f"{gid}.json"
    t = time.time()
    while not out.exists():
        if time.time() - t > timeout:
            raise TimeoutError(f"sem resposta do avaliador para {gid}")
        time.sleep(2)
    return json.loads(out.read_text())


def history(run: str) -> list[dict]:
    """Todos os resultados da run, em ordem."""
    return [json.loads(p.read_text()) for p in sorted((RUNS / run / "outbox").glob("*.json"))]


def load_atlas() -> dict:
    return torch.load(ATLAS)
