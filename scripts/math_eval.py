"""Avaliação por geração no MATH-500 (a tarefa do paper) com braços arbitrários, com raciocínio.

Corretor leve (sem math_verify): normaliza o último \\boxed{} e o gabarito (espaços, \\left/\\right,
dfrac/tfrac, \\text{}, graus, $) e compara como string; se ambos forem numéricos, compara o número.
Os textos brutos ficam em data/gen_<TAG>/<braço>.jsonl, para recorrigir depois com um corretor melhor.

uso: math_eval.py TAG N MAX_NEW BRAÇOS [--np 16] [--offset 0]
"""

import argparse
import json
import re

from datasets import load_dataset
from transformers import AutoTokenizer

from wsp import evalrun, gen, llama_gen
from wsp.paths import BASE_REPO, BASE_REV, DATA, snapshot

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("n", type=int)
ap.add_argument("max_new", type=int)
ap.add_argument("arms")
ap.add_argument("--np", type=int, default=16)
ap.add_argument("--offset", type=int, default=0)
A = ap.parse_args()
TAG, N, MAXN, ARMS = A.tag, A.n, A.max_new, A.arms.split(",")
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(A.offset, A.offset + N))
tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))


def norm(s):
    if s is None:
        return None
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(mathrm|mbox|textbf)\{([^}]*)\}", r"\2", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""),
                 ("%", ""), ("\\$", ""), (" ", "")):
        s = s.replace(a, b)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"^[a-zA-Z]\s*=", "", s)  # "x=5" -> "5"
    return s.rstrip(".")


def same(pred, gold):
    p, g = norm(pred), norm(gold)
    if p is None:
        return False
    if p == g:
        return True
    try:
        return abs(float(p.replace(",", "")) - float(g.replace(",", ""))) < 1e-6
    except ValueError:
        return False


prompts = [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) for q in ds["problem"]]


def run_arm(arm):
    outs, stats = evalrun.run(f"gen_{TAG}", arm, prompts, MAXN, np=A.np, sampling=llama_gen.THINK35)
    rows = []
    for o, g, lvl in zip(outs, ds["answer"], ds["level"]):
        pred = gen.last_boxed(o["text"])
        rows.append({"n_tokens": o["n_tokens"], "cut": o["cut"], "correct": same(pred, g), "pred": pred,
                     "gold": g, "level": lvl, "closed_think": "</think>" in o["text"]})
    return {"model": evalrun.model_of(arm), "rows": rows, "summary": {
        **stats, "acc": sum(r["correct"] for r in rows) / N, "cut": sum(r["cut"] for r in rows) / N,
        "closed_think": sum(r["closed_think"] for r in rows) / N,
        "mean_tokens": sum(r["n_tokens"] for r in rows) / N}}


OUT = DATA / f"gen_{TAG}.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {
    "tag": TAG, "N": N, "max_new": MAXN, "split": f"math500[{A.offset}:{A.offset + N}]", "thinking": True,
    "engine": "llama-server", "sampling": llama_gen.THINK35, "arms": {}}
evalrun.ledger(kind="math_eval", tag=TAG, split=f"math500[{A.offset}:{A.offset + N}]", arms=ARMS, max_new=MAXN)
for arm in ARMS:
    if arm in res["arms"]:
        print(arm, "(já feito)", json.dumps(res["arms"][arm]["summary"]), flush=True)
        continue
    try:
        res["arms"][arm] = run_arm(arm)
    except Exception as e:
        print(arm, "ERRO:", repr(e), flush=True)
        continue
    print(arm, json.dumps(res["arms"][arm]["summary"]), flush=True)
    OUT.write_text(json.dumps(res, indent=1))

if "base" in res["arms"]:
    b = [r["correct"] for r in res["arms"]["base"]["rows"]]
    for arm, a in res["arms"].items():
        if arm != "base":
            a["paired_vs_base"] = evalrun.paired(b, [r["correct"] for r in a["rows"]])
            print(arm, "vs base", a["paired_vs_base"], flush=True)
OUT.write_text(json.dumps(res, indent=1))
