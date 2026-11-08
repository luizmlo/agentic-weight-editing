"""IFEval (Zhou et al., 2023) por braço, com o verificador oficial (wsp.ifeval). Candidato a segundo defeito.

Split fixo dos 541 prompts (embaralhamento com semente 0): dev = 200 primeiros, test = 341 restantes (TRANCADO:
--test exige WSP_UNLOCK_TEST=1 e vai para o ledger). --offset/--n escolhem uma fatia do split.
Métricas: prompt-level strict/loose (todas as instruções do prompt seguidas) e instruction-level strict/loose.

uso: ifeval_eval.py TAG BRAÇOS [--n 200] [--offset 0] [--think] [--max-new 2048] [--np 16] [--test]
"""

import argparse
import collections
import json
import os
import random
import sys

from transformers import AutoTokenizer

from wsp import evalrun, llama_gen
from wsp.ifeval import evaluation_lib as ev
from wsp.paths import BASE_REPO, BASE_REV, DATA, ROOT, snapshot

ALL = ev.read_prompt_list(str(ROOT / "data" / "ifeval" / "input_data.jsonl"))
PERM = random.Random(0).sample(range(len(ALL)), len(ALL))
SPLITS = {"dev": PERM[:200], "test": PERM[200:]}
SYSTEMS = {  # braço P (pré-registrado, CADERNO 2026-11-08 20h): o base com um system prompt pedindo para seguir as instruções
    "if_sys": "You are a careful assistant. Before answering, read every instruction in the request and follow all of "
              "them exactly: format, length, letter case, punctuation, and required or forbidden words. Your response "
              "must satisfy every constraint in the request.",
}


def split(name, n=None, offset=0):
    idx = SPLITS[name][offset:offset + n if n else None]
    return [ALL[i] for i in idx]


def grade(inp, text: str, cut: bool) -> dict:
    body = text.split("</think>", 1)[-1].strip()
    s = ev.test_instruction_following_strict(inp, {inp.prompt: body})
    lo = ev.test_instruction_following_loose(inp, {inp.prompt: body})
    return {"key": inp.key, "ids": inp.instruction_id_list, "strict": s.follow_instruction_list,
            "loose": lo.follow_instruction_list, "prompt_strict": s.follow_all_instructions and not cut,
            "prompt_loose": lo.follow_all_instructions and not cut}


def summarize(rows):
    n = len(rows)
    inst = [x for r in rows for x in r["strict"]]
    inst_l = [x for r in rows for x in r["loose"]]
    per = collections.defaultdict(list)
    for r in rows:
        for i, ok in zip(r["ids"], r["strict"]):
            per[i.split(":")[0]].append(ok)
    return {"prompt_strict": sum(r["prompt_strict"] for r in rows) / n,
            "prompt_loose": sum(r["prompt_loose"] for r in rows) / n,
            "inst_strict": sum(inst) / len(inst), "inst_loose": sum(inst_l) / len(inst_l),
            "per_group_strict": {k: round(sum(v) / len(v), 3) for k, v in sorted(per.items())}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("arms")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--np", type=int, default=16)
    ap.add_argument("--test", action="store_true")
    A = ap.parse_args()
    SPLIT = "test" if A.test else "dev"
    if A.test and os.environ.get("WSP_UNLOCK_TEST") != "1":
        sys.exit("split de teste trancado: defina WSP_UNLOCK_TEST=1 (fica registrado no ledger)")
    MAXN = A.max_new or (6144 if A.think else 2048)
    items = split(SPLIT, A.n, A.offset)
    tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))

    def prompts_for(arm):
        sysm = [{"role": "system", "content": SYSTEMS[arm]}] if arm in SYSTEMS else []
        return [tok.apply_chat_template(sysm + [{"role": "user", "content": x.prompt}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=A.think) for x in items]
    # sem presence penalty: o llama-server põe o prompt no sampler e puniria repetir palavras pedidas no prompt
    SAMPLING = llama_gen.THINK35 if A.think else llama_gen.TOOLS
    PLAST = MAXN + 512 if A.think else 64
    sl = f"{SPLIT}[{A.offset}:{A.offset + len(items)}]"
    OUT = DATA / f"ifeval_{A.tag}.json"
    res = json.loads(OUT.read_text()) if OUT.exists() else {
        "tag": A.tag, "split": sl, "N": len(items), "think": A.think, "max_new": MAXN, "sampling": SAMPLING, "arms": {}}
    evalrun.ledger(kind="ifeval_eval", tag=A.tag, split=sl, arms=A.arms.split(","), think=A.think)
    for arm in A.arms.split(","):
        if arm in res["arms"]:
            print(arm, "(já feito)", json.dumps(res["arms"][arm]["summary"]), flush=True)
            continue
        try:
            outs, stats = evalrun.run(f"ifeval_{A.tag}", arm, prompts_for(arm), MAXN, np=A.np, sampling=SAMPLING,
                                      penalty_last_n=PLAST, ctx_per_slot=MAXN + 1024)
        except Exception as e:
            print(arm, "ERRO:", repr(e), flush=True)
            continue
        rows = [{**grade(x, o["text"], o["cut"]), "n_tokens": o["n_tokens"], "cut": o["cut"], "text": o["text"][-1500:]}
                for x, o in zip(items, outs)]
        summ = {**stats, **summarize(rows), "cut": sum(r["cut"] for r in rows) / len(rows),
                "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows)}
        res["arms"][arm] = {"model": evalrun.model_of(arm), "summary": summ, "rows": rows}
        print(arm, json.dumps(summ), flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    if "base" in res["arms"]:
        b = [r["prompt_strict"] for r in res["arms"]["base"]["rows"]]
        for arm, a in res["arms"].items():
            if arm != "base":
                a["paired_vs_base"] = evalrun.paired(b, [r["prompt_strict"] for r in a["rows"]])
                print(arm, "vs base", a["paired_vs_base"], flush=True)
    OUT.write_text(json.dumps(res, indent=1))
