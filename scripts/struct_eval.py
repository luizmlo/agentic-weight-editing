"""Triagem/avaliação do segundo defeito (saída estruturada, wsp.structbench) por braço.

uso: struct_eval.py TAG FAMÍLIA N BRAÇOS [--think] [--max-new 768] [--np 16] [--offset 0] [--test]
     FAMÍLIA: json | tools | ifeval.  Split dev por padrão; --test exige WSP_UNLOCK_TEST=1 e vai para o ledger.
"""

import argparse
import json
import os
import sys

from transformers import AutoTokenizer

from wsp import evalrun, llama_gen, structbench
from wsp.paths import BASE_REPO, BASE_REV, DATA, snapshot

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("family", choices=list(structbench.FAMILIES))
ap.add_argument("n", type=int)
ap.add_argument("arms")
ap.add_argument("--think", action="store_true")
ap.add_argument("--max-new", type=int, default=None)
ap.add_argument("--np", type=int, default=16)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--test", action="store_true")
A = ap.parse_args()
SPLIT = "test" if A.test else "dev"
if A.test and os.environ.get("WSP_UNLOCK_TEST") != "1":
    sys.exit("split de teste trancado: defina WSP_UNLOCK_TEST=1 (fica registrado no ledger)")
MAXN = A.max_new or (4096 if A.think else 768)
tasks = structbench.split(A.family, SPLIT, A.n, A.offset)
tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))
prompts = [tok.apply_chat_template([{"role": "user", "content": t["user"]}], tools=t.get("tools"), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=A.think) for t in tasks]
SAMPLING = llama_gen.TOOLS  # sem presence penalty: o llama-server põe o prompt no sampler e puniria copiar valores

OUT = DATA / f"struct_{A.tag}.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {
    "tag": A.tag, "family": A.family, "split": f"{SPLIT}[{A.offset}:{A.offset + A.n}]", "N": A.n, "think": A.think,
    "max_new": MAXN, "sampling": SAMPLING, "arms": {}}
evalrun.ledger(kind="struct_eval", tag=A.tag, family=A.family, split=res["split"], arms=A.arms.split(","), think=A.think)
for arm in A.arms.split(","):
    if arm in res["arms"]:
        print(arm, "(já feito)", json.dumps(res["arms"][arm]["summary"]), flush=True)
        continue
    try:
        outs, stats = evalrun.run(f"struct_{A.tag}", arm, prompts, MAXN, np=A.np, sampling=SAMPLING, penalty_last_n=64,
                                  ctx_per_slot=MAXN + 1536)
    except Exception as e:
        print(arm, "ERRO:", repr(e), flush=True)
        continue
    rows = [{**structbench.score(t, o["text"], o["cut"]), "n_tokens": o["n_tokens"], "cut": o["cut"],
             "text": o["text"][-1200:]} for t, o in zip(tasks, outs)]
    n = len(rows)
    summ = {**stats, "exact": sum(r["exact"] for r in rows) / n, "cut": sum(r["cut"] for r in rows) / n,
            "mean_tokens": sum(r["n_tokens"] for r in rows) / n}
    for k in ("json_ok", "schema_ok", "format_ok", "name_ok", "type_ok"):
        if k in rows[0]:
            summ[k] = sum(r[k] for r in rows) / n
    if A.family == "json":
        summ["field_acc"] = sum(r["field_acc"] for r in rows) / n
        summ["strict_parse"] = sum(r["parse"] == "strict" for r in rows) / n
    if A.family == "tools":
        summ["extra_optional_rate"] = sum(r["extra_optional"] > 0 for r in rows) / n
        summ["missing_required_rate"] = sum(r["missing_required"] > 0 for r in rows) / n
    if A.family == "ifeval":
        per = {}
        for r in rows:
            for c, ok in r["per"].items():
                per.setdefault(c, []).append(ok)
        summ["per_constraint"] = {c: round(sum(v) / len(v), 3) for c, v in per.items()}
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
