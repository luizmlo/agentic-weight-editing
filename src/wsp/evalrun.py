"""Execução de um braço de avaliação por geração, comum a gen_eval e tool_eval.

- braço -> modelo: nomes fixos (Ollama) ou data/gguf/<braço>.gguf
- checkpoint POR ITEM em data/<tag>/<braço>.jsonl (retoma de onde parou)
- se o servidor não subir (VRAM) ou cair, tenta de novo com metade dos slots
- ledger: toda avaliação registra uma linha em data/ledger.jsonl (orçamento e uso do split de teste)
"""

from __future__ import annotations

import json
import re
import subprocess
import time

import httpx

from wsp import llama_gen
from wsp.paths import DATA

NAMED = {"base": "wsp-base", "astra": "wsp-astra", "anti_rep": "wsp-base", "placebo": "wsp-base", "if_sys": "wsp-base"}
LEDGER = DATA / "ledger.jsonl"


# Braços de DECODIFICAÇÃO sobre o modelo base (E1: baselines contra a edição de pesos)
#   lb_w<A>_t<B>   logit bias -A em "Wait"/" Wait" (13784, 13428) e +B em "</think>" (248069); ex.: lb_w5_t1
#   lbx_w<A>_t<B>  idem, incluindo "wait"/" wait" minúsculos (11158, 3655)
#   bf<N>          budget forcing: se o raciocínio não fechou em N tokens, injeta "</think>" e segue até o teto
WAIT_IDS, WAIT_LC_IDS, ENDTHINK_ID = (13784, 13428), (11158, 3655), 248069
_DEC = re.compile(r"(lbx?)_w([\d.]+)_t([\d.]+)|bf(\d+)")


def decode_extra(arm: str) -> dict:
    if "+" in arm:  # combinação <braço>+<decodificação>, ex.: m2a_a1+bf4096
        return decode_extra(arm.split("+", 1)[1])
    m = _DEC.fullmatch(arm)
    if not m:
        return {}
    if m.group(4):
        return {"think_budget": int(m.group(4))}
    a, b = float(m.group(2)), float(m.group(3))
    ids = WAIT_IDS + (WAIT_LC_IDS if m.group(1) == "lbx" else ())
    return {"logit_bias": [[i, -a] for i in ids] + ([[ENDTHINK_ID, b]] if b else [])}


def model_of(arm: str) -> str:
    if "+" in arm:
        return model_of(arm.split("+", 1)[0])
    if _DEC.fullmatch(arm):
        return NAMED["base"]
    if arm in NAMED:
        return NAMED[arm]
    g = DATA / "gguf" / f"{arm}.gguf"
    if not g.exists():
        raise FileNotFoundError(g)
    return str(g)


def ledger(**kw):
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%F %T"), **kw}) + "\n")


def run(tag: str, arm: str, prompts: list[str], max_new: int, np: int = 16, penalty_last_n: int | None = None,
        sampling: dict | None = None, ctx_per_slot: int | None = None) -> tuple[list[dict], dict]:
    """Gera prompts[i] com semente i. Devolve (saídas na ordem, estatísticas desta execução)."""
    part = DATA / tag / f"{arm}.jsonl"
    part.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if part.exists():
        for line in part.read_text().splitlines():
            d = json.loads(line)
            done[d["i"]] = d
    n = len(prompts)
    todo = [i for i in range(n) if i not in done]
    print(tag, arm, f"{len(done)} já feitos, {len(todo)} a gerar", flush=True)
    t, new_tok = time.time(), 0
    if todo:
        f = part.open("a")

        def on_done(k, out):
            nonlocal new_tok
            d = {"i": cur[k], **out}
            done[d["i"]] = d
            new_tok += out["n_tokens"]
            f.write(json.dumps(d) + "\n")
            f.flush()
            if len(done) % 8 == 0 or len(done) == n:
                print(f"  {arm} {len(done)}/{n} ({new_tok / (time.time() - t):.0f} tok/s)", flush=True)

        np_ = np
        while True:
            cur = [i for i in range(n) if i not in done]
            if not cur:
                break
            try:
                llama_gen.generate(model_of(arm), [prompts[i] for i in cur], max_new, seeds=cur, concurrency=np_,
                                   ctx_per_slot=ctx_per_slot or max_new + 1024,
                                   penalty_last_n=penalty_last_n if penalty_last_n is not None else max_new + 512,
                                   on_done=on_done, sampling={**(sampling or llama_gen.THINK35), **decode_extra(arm)})
                break
            except (RuntimeError, OSError, subprocess.CalledProcessError, httpx.HTTPError) as e:
                if np_ <= 4:
                    raise
                np_ //= 2
                print(arm, "servidor falhou:", repr(e), "-> np", np_, flush=True)
        f.close()
    secs = time.time() - t
    return [done[i] for i in range(n)], {"secs_this_run": round(secs),
                                         "tok_s_this_run": round(new_tok / secs, 1) if todo else None}


def paired(base_ok: list[bool], arm_ok: list[bool]) -> dict:
    d = [int(y) - int(x) for x, y in zip(base_ok, arm_ok)]
    n = len(d)
    mu = sum(d) / n
    se = (sum((x - mu) ** 2 for x in d) / (n - 1) / n) ** 0.5 if n > 1 else 0.0
    return {"gain": mu, "se": se, "rescues": d.count(1), "losses": d.count(-1)}
