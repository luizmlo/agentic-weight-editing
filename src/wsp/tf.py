"""Leitura teacher-forced: log-prob por token e residual stream em camadas escolhidas.

Os logits são calculados em blocos (vocabulário de 248k não cabe inteiro na 1660 para textos longos).
Convenção de camadas como no paper: hidden[0] = embeddings, hidden[l] = saída da camada l-1 (0-based),
ou seja, hidden[16] é o residual depois da camada 15.
"""

from __future__ import annotations

import torch


def build_ids(tok, messages, assistant_text: str, max_text_tokens: int | None = None, enable_thinking=True):
    """ids = template(messages, generation prompt) + tokens do texto do assistente. Devolve (ids, n_prompt)."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(assistant_text, add_special_tokens=False)["input_ids"]
    if max_text_tokens:
        t_ids = t_ids[:max_text_tokens]
    return torch.tensor(p_ids + t_ids), len(p_ids)


@torch.no_grad()
def read(model, ids: torch.Tensor, hidden_layers=(), chunk=1024):
    """Um texto -> {'logp': [T-1] log p(token_t | <t), 'hidden': {l: [T, d] fp32 na CPU}}."""
    ids = ids.to(model.device)[None]
    want = sorted(set(hidden_layers))
    out = model.model(input_ids=ids, output_hidden_states=bool(want), use_cache=False)
    h_last = out.last_hidden_state[0]  # já passou pela norma final
    lp = torch.empty(ids.shape[1] - 1, device="cpu")
    for s in range(0, ids.shape[1] - 1, chunk):
        e = min(s + chunk, ids.shape[1] - 1)
        logits = model.lm_head(h_last[s:e]).float()
        lp[s:e] = logits.log_softmax(-1).gather(-1, ids[0, s + 1 : e + 1, None]).squeeze(-1).cpu()
    hidden = {l: out.hidden_states[l][0].float().cpu() for l in want}
    return {"logp": lp, "hidden": hidden}


def dup_ngram_mask(ids: list[int], n=8) -> torch.Tensor:
    """mask[t] = True se o n-grama que termina no token t já apareceu antes no mesmo texto."""
    seen, mask = set(), torch.zeros(len(ids), dtype=torch.bool)
    for t in range(n - 1, len(ids)):
        g = tuple(ids[t - n + 1 : t + 1])
        if g in seen:
            mask[t] = True
        seen.add(g)
    return mask
