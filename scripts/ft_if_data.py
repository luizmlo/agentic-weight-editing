"""Baseline de fine-tuning do segundo defeito (análogo ao E4.1): dados por amostragem de rejeição.

O próprio base responde K vezes (sem raciocínio, teto 2048, amostragem TOOLS) a N prompts sintéticos no estilo
IFEval (wsp.ifsynth: tarefas próprias + instruções sorteadas pelos geradores do verificador oficial; nenhum prompt do
IFEval). Fica a primeira resposta que segue TODAS as instruções e não foi cortada (no máximo 1 por prompt, como no E4).
Saída: data/ft_if/sft.jsonl ({prompt, completion, kind}) e data/ft_if/sft_stats.json.

uso: ft_if_data.py [N=600] [K=2]
"""

import collections
import json
import sys

from transformers import AutoTokenizer

from wsp import evalrun, ifsynth, llama_gen
from wsp.paths import BASE_REPO, BASE_REV, DATA, snapshot

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OUT = DATA / "ft_if"
OUT.mkdir(parents=True, exist_ok=True)
items = [ifsynth.make(10000 + i) for i in range(N)]
tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))
P = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False, add_generation_prompt=True,
                             enable_thinking=False) for x in items]
evalrun.ledger(kind="ft_if_data", split=f"ifsynth[10000:{10000 + N}]", k=K)
outs, stats = evalrun.run("ft_if_gen", "base", [p for p in P for _ in range(K)], 2048, np=16, penalty_last_n=64,
                          sampling=llama_gen.TOOLS, ctx_per_slot=3072)
rows, per_inst = [], collections.defaultdict(list)
n_ok_samples = 0
for i, x in enumerate(items):
    for k in range(K):
        o = outs[i * K + k]
        ok = ifsynth.check(x, o["text"]) if not o["cut"] else [False] * len(x["ids"])
        for iid, v in zip(x["ids"], ok):
            per_inst[iid.split(":")[0]].append(v)
        if all(ok):
            n_ok_samples += 1
            if not any(r["i"] == i for r in rows):
                rows.append({"i": i, "kind": "ifsynth", "prompt": P[i], "completion": o["text"].strip(), "ids": x["ids"]})
(OUT / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
st = {"N": N, "K": K, "kept_prompts": len(rows), "sample_pass_rate": n_ok_samples / (N * K),
      "cut_rate": sum(o["cut"] for o in outs) / len(outs), **stats,
      "per_group_pass": {g: round(sum(v) / len(v), 3) for g, v in sorted(per_inst.items())}}
(OUT / "sft_stats.json").write_text(json.dumps(st, indent=1))
print(json.dumps(st), flush=True)
