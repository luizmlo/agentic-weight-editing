"""PoC-6: confirmação comportamental por geração (llama-server do Ollama com slots paralelos).

GSM8K test, N problemas, braços com a mesma semente por problema:
  base      wsp-base, sem system prompt
  astra     wsp-astra (checkpoint editado publicado)
  anti_rep  wsp-base + system prompt anti-repetição        (braço P da ideia do prompt)
  placebo   wsp-base + system prompt neutro de mesmo tamanho (controle do anterior)

uso: poc6_gen.py N MAX_NEW [braços separados por vírgula] [concorrência]
"""

import json
import sys
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from wsp import gen, llama_gen
from wsp.paths import BASE_REPO, BASE_REV, DATA, snapshot

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
MAXN = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "base,anti_rep,placebo,astra").split(",")
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SYSTEMS = {
    "anti_rep": "Think efficiently. Never repeat a sentence or a calculation you have already written. "
    "If you notice that you are repeating yourself, stop and write the final answer.",
    "placebo": "You are a helpful assistant. Read the user's question with care and answer it in a "
    "clear, friendly and well organized way, using plain language.",
}
MODEL = {"base": "wsp-base", "astra": "wsp-astra", "anti_rep": "wsp-base", "placebo": "wsp-base"}

ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N))
gold = [float(a.split("####")[-1].replace(",", "").strip()) for a in ds["answer"]]
tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))


def prompts(system=None):
    out = []
    for q in ds["question"]:
        m = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q + SUFFIX}]
        out.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True))
    return out


OUT = DATA / f"poc6_gen_n{N}_t{MAXN}.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {"N": N, "max_new": MAXN, "engine": "llama-server",
                                                       "concurrency": CONC, "arms": {}}
for arm in ARMS:
    if arm in res["arms"]:  # retomável: braço já concluído numa execução anterior
        print(arm, "(já feito)", json.dumps(res["arms"][arm]["summary"]), flush=True)
        continue
    t = time.time()
    outs = llama_gen.generate(MODEL[arm], prompts(SYSTEMS.get(arm)), MAXN, seeds=list(range(N)), concurrency=CONC)
    secs = time.time() - t
    rows = []
    for o, g in zip(outs, gold):
        ids = tok(o["text"], add_special_tokens=False)["input_ids"]
        m = gen.dup_ngram_mask(ids)
        tail = m[-512:]
        ans = gen.num(gen.last_boxed(o["text"]))
        rows.append({
            "n_tokens": o["n_tokens"], "cut": o["cut"], "correct": ans is not None and abs(ans - g) < 1e-6,
            "closed_think": "</think>" in o["text"], "dup_tail": float(tail.float().mean()) if len(tail) else 0.0,
            "tail": o["text"][-200:],
        })
    n = len(rows)
    toks = sum(r["n_tokens"] for r in rows)
    summ = {
        "secs": round(secs), "tok_s": round(toks / secs, 1),
        "acc": sum(r["correct"] for r in rows) / n,
        "cut": sum(r["cut"] for r in rows) / n,
        "closed_think": sum(r["closed_think"] for r in rows) / n,
        "loop_at_cap": sum(r["cut"] and r["dup_tail"] > 0.5 for r in rows) / n,
        "mean_tokens": toks / n,
    }
    res["arms"][arm] = {"summary": summ, "rows": rows}
    print(arm, json.dumps(summ), flush=True)
    OUT.write_text(json.dumps(res, indent=1))

if "base" in res["arms"]:
    b = res["arms"]["base"]["rows"]
    for arm in ARMS:
        if arm == "base":
            continue
        e = res["arms"][arm]["rows"]
        n = len(e)
        d = [int(y["correct"]) - int(x["correct"]) for x, y in zip(b, e)]
        mu = sum(d) / n
        se = (sum((x - mu) ** 2 for x in d) / (n - 1) / n) ** 0.5
        res["arms"][arm]["paired_vs_base"] = {"gain": mu, "se": se, "rescues": d.count(1), "losses": d.count(-1)}
        print(arm, "vs base", res["arms"][arm]["paired_vs_base"], flush=True)
OUT.write_text(json.dumps(res, indent=1))
