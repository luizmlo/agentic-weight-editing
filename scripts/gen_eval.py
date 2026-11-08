"""Avaliação por geração no GSM8K test com braços arbitrários (sucessor do poc6_gen).

Braços: base | astra | anti_rep | placebo | <nome> (data/gguf/<nome>.gguf, sem system prompt).
Mesma semente por problema em todos os braços (pareamento aproximado: o batching do llama.cpp não é
determinístico). Checkpoint por problema: ver wsp.evalrun.

uso: gen_eval.py TAG N MAX_NEW BRAÇOS [--np 16] [--plast MAX_NEW+512] [--nothink]
"""

import argparse
import json

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
ap.add_argument("--plast", type=int, default=None)
ap.add_argument("--nothink", action="store_true", help="enable_thinking=False e amostragem non-thinking")
ap.add_argument("--split", default="test", help="test (avaliação) ou train (dev do agente)")
ap.add_argument("--offset", type=int, default=0)
A = ap.parse_args()
TAG, N, MAXN, ARMS = A.tag, A.n, A.max_new, A.arms.split(",")
SAMPLING = llama_gen.NOTHINK if A.nothink else llama_gen.THINK35
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SYSTEMS = {
    "anti_rep": "Think efficiently. Never repeat a sentence or a calculation you have already written. "
    "If you notice that you are repeating yourself, stop and write the final answer.",
    "placebo": "You are a helpful assistant. Read the user's question with care and answer it in a "
    "clear, friendly and well organized way, using plain language.",
}

ds = load_dataset("openai/gsm8k", "main", split=A.split).select(range(A.offset, A.offset + N))
gold = [float(a.split("####")[-1].replace(",", "").strip()) for a in ds["answer"]]
tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))


def prompts(system=None):
    out = []
    for q in ds["question"]:
        m = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q + SUFFIX}]
        out.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=not A.nothink))
    return out


def run_arm(arm):
    outs, stats = evalrun.run(f"gen_{TAG}", arm, prompts(SYSTEMS.get(arm)), MAXN, np=A.np,
                              penalty_last_n=A.plast, sampling=SAMPLING)
    rows = []
    for o, g in zip(outs, gold):
        ids = tok(o["text"], add_special_tokens=False)["input_ids"]
        tail = gen.dup_ngram_mask(ids)[-512:]
        ans = gen.num(gen.last_boxed(o["text"]))
        rows.append({
            "n_tokens": o["n_tokens"], "cut": o["cut"], "correct": ans is not None and abs(ans - g) < 1e-6,
            "closed_think": "</think>" in o["text"], "dup_tail": float(tail.float().mean()) if len(tail) else 0.0,
            "tail": o["text"][-200:],
        })
    fin = sorted(r["n_tokens"] for r in rows if not r["cut"])
    summ = {**stats,
            "acc": sum(r["correct"] for r in rows) / N,
            "cut": sum(r["cut"] for r in rows) / N,
            "closed_think": sum(r["closed_think"] for r in rows) / N,
            "loop_at_cap": sum(r["cut"] and r["dup_tail"] > 0.5 for r in rows) / N,
            "mean_tokens": sum(r["n_tokens"] for r in rows) / N,
            "median_tokens_finished": fin[len(fin) // 2] if fin else None}
    return {"model": evalrun.model_of(arm), "summary": summ, "rows": rows}


OUT = DATA / f"gen_{TAG}.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {
    "tag": TAG, "N": N, "max_new": MAXN, "np": A.np, "penalty_last_n": A.plast or MAXN + 512,
    "split": f"{A.split}[{A.offset}:{A.offset + N}]", "thinking": not A.nothink, "engine": "llama-server", "sampling": SAMPLING, "arms": {}}
evalrun.ledger(kind="gen_eval", tag=TAG, split=f"gsm8k_{A.split}[{A.offset}:{A.offset + N}]", arms=ARMS, max_new=MAXN)
for arm in ARMS:
    if arm in res["arms"]:
        print(arm, "(já feito)", json.dumps(res["arms"][arm]["summary"]), flush=True)
        continue
    try:
        res["arms"][arm] = run_arm(arm)
    except Exception as e:  # um braço quebrado não derruba a fila
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
