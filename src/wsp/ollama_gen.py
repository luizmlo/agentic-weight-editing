"""Geração em paralelo pelo serviço Ollama (systemd, OLLAMA_NUM_PARALLEL=16 via drop-in wsp.conf).

Usa `raw: true` com o prompt já montado pelo tokenizer do HF, para o template ser idêntico ao das
leituras teacher-forced, e o protocolo de amostragem do paper (think35 + presence_penalty 1.5).
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx

URL = os.environ.get("WSP_OLLAMA_URL", "http://127.0.0.1:11434")
THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


async def _one(client, sem, model, prompt, max_new, seed, num_ctx):
    async with sem:
        body = {
            "model": model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "30m",
            "options": {**THINK35, "seed": seed, "num_predict": max_new, "num_ctx": num_ctx},
        }
        t = time.time()
        r = await client.post(f"{URL}/api/generate", json=body)
        r.raise_for_status()
        d = r.json()
        text = (f"{d['thinking']}" if d.get("thinking") else "") + d.get("response", "")
        return {"text": text, "n_tokens": d.get("eval_count", 0), "done_reason": d.get("done_reason"),
                "cut": d.get("done_reason") == "length", "secs": time.time() - t}


async def _many(model, prompts, max_new, seeds, concurrency, num_ctx):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=None) as client:
        return await asyncio.gather(*[_one(client, sem, model, p, max_new, s, num_ctx) for p, s in zip(prompts, seeds)])


def generate(model: str, prompts: list[str], max_new: int, seeds: list[int] | None = None,
             concurrency: int = 16, num_ctx: int | None = None) -> list[dict]:
    seeds = seeds if seeds is not None else list(range(len(prompts)))
    num_ctx = num_ctx or max_new + 1024
    return asyncio.run(_many(model, prompts, max_new, seeds, concurrency, num_ctx))
