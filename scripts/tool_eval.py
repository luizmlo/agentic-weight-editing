"""Avaliação de tool calling (benchmark sintético wsp.toolbench) por braço.

Mede se as edições de pesos mexem em uma capacidade não-alvo (efeito colateral) e serve de linha de
base para o braço de tool calling (F3). Split dev por padrão; --test exige WSP_UNLOCK_TEST=1 e vai
para o ledger.

uso: tool_eval.py TAG N BRAÇOS [--think] [--max-new 512] [--np 32] [--test]
"""

import argparse
import json
import os
import sys

from transformers import AutoTokenizer

from wsp import evalrun, llama_gen, toolbench
from wsp.paths import BASE_REPO, BASE_REV, DATA, snapshot

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("n", type=int)
ap.add_argument("arms")
ap.add_argument("--think", action="store_true")
ap.add_argument("--max-new", type=int, default=None)
ap.add_argument("--np", type=int, default=32)
ap.add_argument("--test", action="store_true")
A = ap.parse_args()
SPLIT = "test" if A.test else "dev"
if A.test and os.environ.get("WSP_UNLOCK_TEST") != "1":
    sys.exit("split de teste trancado: defina WSP_UNLOCK_TEST=1 (fica registrado no ledger)")
MAXN = A.max_new or (3072 if A.think else 512)
tasks = toolbench.split(SPLIT, A.n)
tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))
prompts = [tok.apply_chat_template([{"role": "user", "content": t["user"]}], tools=t["tools"], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=A.think) for t in tasks]

OUT = DATA / f"tool_{A.tag}.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {
    "tag": A.tag, "split": SPLIT, "N": A.n, "think": A.think, "max_new": MAXN, "sampling": llama_gen.TOOLS, "arms": {}}
evalrun.ledger(kind="tool_eval", tag=A.tag, split=SPLIT, n=A.n, arms=A.arms.split(","), think=A.think)
for arm in A.arms.split(","):
    if arm in res["arms"]:
        print(arm, "(já feito)", json.dumps(res["arms"][arm]["summary"]), flush=True)
        continue
    try:
        outs, stats = evalrun.run(f"tool_{A.tag}", arm, prompts, MAXN, np=A.np, sampling=llama_gen.TOOLS,
                                  penalty_last_n=64)
    except Exception as e:
        print(arm, "ERRO:", repr(e), flush=True)
        continue
    rows = [{**toolbench.score(t, o["text"], o["cut"]), "n_tokens": o["n_tokens"], "text": o["text"][-400:]}
            for t, o in zip(tasks, outs)]
    summ = {**stats, "exact": sum(r["exact"] for r in rows) / A.n,
            "format_ok": sum(r["format_ok"] for r in rows) / A.n,
            "cut": sum(o["cut"] for o in outs) / A.n,
            "mean_tokens": sum(r["n_tokens"] for r in rows) / A.n}
    for k in ("single", "parallel", "irrelevance"):
        sub = [r for r in rows if r["kind"] == k]
        summ[f"exact_{k}"] = sum(r["exact"] for r in sub) / len(sub) if sub else None
    res["arms"][arm] = {"model": evalrun.model_of(arm), "summary": summ, "rows": rows}
    print(arm, json.dumps(summ), flush=True)
    OUT.write_text(json.dumps(res, indent=1))

if "base" in res["arms"]:
    b = [r["exact"] for r in res["arms"]["base"]["rows"]]
    for arm, a in res["arms"].items():
        if arm != "base":
            a["paired_vs_base"] = evalrun.paired(b, [r["exact"] for r in a["rows"]])
            print(arm, "vs base", a["paired_vs_base"], flush=True)
OUT.write_text(json.dumps(res, indent=1))
