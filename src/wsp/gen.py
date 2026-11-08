"""Geração em lote no HF com o protocolo de amostragem do paper (think35) e métricas de loop."""

from __future__ import annotations

import re

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from wsp.tf import dup_ngram_mask

THINK35 = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
PRESENCE_PENALTY = 1.5


class PresencePenalty(LogitsProcessor):
    """Como no vLLM: subtrai `penalty` do logit de todo token que já apareceu na saída gerada."""

    def __init__(self, penalty: float, prompt_len: int):
        self.penalty, self.prompt_len = penalty, prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len :]
        if gen.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, gen, True)
        return scores - self.penalty * seen.to(scores.dtype)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int, seed: int = 0, batch_size: int = 64):
    """prompts já com o chat template aplicado. Devolve lista de dicts com texto e ids gerados."""
    tok.padding_side = "left"
    out = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s : s + batch_size]
        b = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seed + s)
        plen = b["input_ids"].shape[1]
        g = model.generate(
            **b,
            max_new_tokens=max_new_tokens,
            logits_processor=LogitsProcessorList([PresencePenalty(PRESENCE_PENALTY, plen)]),
            pad_token_id=tok.pad_token_id,
            **THINK35,
        )
        for row in g[:, plen:].tolist():
            ids = [t for t in row if t != tok.pad_token_id]
            out.append({"ids": ids, "text": tok.decode(ids, skip_special_tokens=False)})
    return out


BOX = re.compile(r"\\boxed\{")


def last_boxed(text: str) -> str | None:
    starts = [m.end() for m in BOX.finditer(text)]
    if not starts:
        return None
    i, depth, j = starts[-1], 1, starts[-1]
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i : j - 1] if depth == 0 else None


def num(s: str | None):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s).strip()
    try:
        return float(s)
    except ValueError:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        return float(m[-1]) if len(m) == 1 else None


def loop_stats(ids: list[int], max_new_tokens: int, eos_ids: set[int], tail=512):
    """cut = bateu no teto sem EOS; dup_tail = fração de 8-gramas repetidos nos últimos `tail` tokens."""
    m = dup_ngram_mask(ids)
    t = m[-tail:] if len(ids) >= tail else m
    finished = bool(ids) and ids[-1] in eos_ids
    return {"n_tokens": len(ids), "cut": (not finished) and len(ids) >= max_new_tokens,
            "dup_tail": float(t.float().mean()) if len(t) else 0.0}
