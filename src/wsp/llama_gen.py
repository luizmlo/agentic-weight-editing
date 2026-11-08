"""Geração em paralelo pelo llama-server do experimento (scripts/llama_serve.sh, slots -np).

Mesma interface do ollama_gen. O servidor carrega um modelo por vez: `ensure(model)` troca se preciso.
Prompt cru (já com o chat template do HF) e protocolo de amostragem do paper.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import httpx

from wsp.paths import ROOT

URL = os.environ.get("WSP_LLAMA_URL", "http://127.0.0.1:8090")
THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "min_p": 0.0}
# modo sem raciocínio (recomendação da Qwen para non-thinking)
NOTHINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "min_p": 0.0}
# tool calling: sem presence penalty, porque o llama-server põe o prompt no sampler e a penalidade puniria
# copiar os valores dos argumentos que estão na mensagem do usuário
TOOLS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0, "min_p": 0.0}
SERVE = ROOT / "scripts" / "llama_serve.sh"
STATE = Path(os.environ.get("WSP_LLAMA_DIR", ROOT / "data" / "llama")) / "model"


def ensure(model: str, np: int = 16, ctx_per_slot: int = 9216):
    """Sobe/troca o servidor se o modelo OU a configuração de slots for diferente da pedida."""
    want = f"{model} {np} {ctx_per_slot}"
    cur = STATE.read_text().strip() if STATE.exists() else None
    try:
        up = httpx.get(f"{URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        up = False
    if not (up and cur == want):
        subprocess.run([str(SERVE), "start", model, str(np), str(ctx_per_slot)], check=True)
        if httpx.get(f"{URL}/health", timeout=5).status_code != 200:
            raise RuntimeError(f"llama-server não subiu: {want}")


async def _one(client, sem, i, prompt, max_new, seed, penalty_last_n, on_done, sampling):
    async with sem:
        # repeat_last_n: janela da presence penalty no llama.cpp (padrão 64). O vLLM do paper penaliza
        # a saída inteira -> passar >= max_new. (O llama-server também alimenta o prompt no sampler.)
        sampling = dict(sampling)
        budget = sampling.pop("think_budget", None)  # budget forcing (s1): fecha o raciocínio em `budget` tokens
        t = time.time()

        async def call(p, n):
            body = {"prompt": p, "n_predict": n, "seed": seed, "cache_prompt": False,
                    "repeat_last_n": penalty_last_n, **sampling}
            r = await client.post(f"{URL}/completion", json=body)
            r.raise_for_status()
            d = r.json()
            return d["content"], d.get("tokens_predicted", 0), d.get("stop_type")

        if budget is None or budget >= max_new:
            text, n, stop = await call(prompt, max_new)
            forced = False
        else:
            text, n, stop = await call(prompt, budget)
            forced = False
            if stop == "limit":
                if "</think>" not in text:  # força o fechamento e deixa responder com o resto do teto
                    text += "\n</think>\n\n"
                    forced = True
                t2, n2, stop = await call(prompt + text, max_new - budget)
                text, n = text + t2, n + n2
        out = {"text": text, "n_tokens": n, "done_reason": stop, "cut": stop == "limit", "forced": forced,
               "secs": time.time() - t}
        if on_done:
            on_done(i, out)
        return out


async def _many(prompts, max_new, seeds, concurrency, penalty_last_n, on_done, sampling):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=None) as client:
        return await asyncio.gather(*[_one(client, sem, i, p, max_new, s, penalty_last_n, on_done, sampling)
                                      for i, (p, s) in enumerate(zip(prompts, seeds))])


def generate(model: str, prompts: list[str], max_new: int, seeds: list[int] | None = None,
             concurrency: int = 16, ctx_per_slot: int | None = None, penalty_last_n: int = 64,
             on_done=None, sampling: dict | None = None) -> list[dict]:
    """on_done(i, out) é chamado a cada resposta concluída (checkpoint por problema)."""
    ensure(model, concurrency, ctx_per_slot or max_new + 1024)
    seeds = seeds if seeds is not None else list(range(len(prompts)))
    return asyncio.run(_many(prompts, max_new, seeds, concurrency, penalty_last_n, on_done, sampling or THINK35))
