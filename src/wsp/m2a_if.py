"""Ferramenta do agente para o SEGUNDO defeito (seguir instruções de formato, IFEval, sem raciocínio).

Mesmo espaço de edições e mesmas regras de wsp.m2a (até 4 edições rank-1 do lado da saída, |alpha| <= 10); muda só
o que o avaliador mede (daemon scripts/m2a_scorer_if.py, runs em data/m2a_if/runs/).

    from wsp import m2a_if as m
    r = m.submit(RUN, [{"tensor": m.tensor_name(15, "mlp"), "u": u, "cols": None, "alpha": 2.0}], note="hipótese")
    g = m.submit_gen(RUN, r["id"])     # checagem por geração (IFEval, 48 prompts de prática)
    m.history(RUN)
"""

from __future__ import annotations

from wsp import m2a
from wsp.m2a import ALLOWED, D, FULL_ATTN, IN_DIM, KIND, MAX_ALPHA, MAX_EDITS, apply, tensor_name, validate  # noqa: F401
from wsp.paths import DATA

M2A = DATA / "m2a_if"
RUNS = M2A / "runs"


def submit(run: str, edits: list[dict], note: str = "", timeout: float = 900) -> dict:
    return m2a.submit(run, edits, note, timeout, runs=RUNS)


def submit_gen(run: str, cid: str, timeout: float = 3600) -> dict:
    return m2a.submit_gen(run, cid, timeout, runs=RUNS)


def history(run: str) -> list[dict]:
    return m2a.history(run, runs=RUNS)
