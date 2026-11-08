"""Carregar o Qwen3.5-0.8B (só a parte de texto) e aplicar edições de pesos em memória."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wsp.paths import BASE_REPO, BASE_REV, snapshot


def load(dtype=torch.float16, device="cuda", repo=BASE_REPO, revision=BASE_REV):
    path = snapshot(repo, revision)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def layers(model):
    return model.model.layers


def down_proj(model, layer: int) -> torch.nn.Linear:
    return layers(model)[layer].mlp.down_proj


@torch.no_grad()
def apply_rank1(model, layer: int, u: torch.Tensor, alpha: float, cols=None):
    """W <- W - alpha * u (u^T W) [só nas colunas `cols`]. Devolve uma função que desfaz."""
    lin = down_proj(model, layer)
    W = lin.weight
    old = W.detach().clone()
    Wf = W.float()
    u = u.to(Wf.device, torch.float32)
    u = u / u.norm()
    proj = u @ Wf  # [in]
    if cols is not None:
        mask = torch.zeros_like(proj)
        mask[torch.as_tensor(cols, device=proj.device)] = 1.0
        proj = proj * mask
    W.copy_((Wf - alpha * torch.outer(u, proj)).to(W.dtype))

    def undo():
        W.copy_(old)

    return undo


def apply_published_edit(model, dirs: dict):
    """Aplica a edição do Astra recuperada pelo poc1_diff (dict layer -> {u, cols, alpha})."""
    undos = [apply_rank1(model, l, d["u"], d["alpha"], d["cols"]) for l, d in dirs.items()]

    def undo():
        for f in reversed(undos):
            f()

    return undo
