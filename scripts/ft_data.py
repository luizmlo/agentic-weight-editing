"""E4.1 — dados de SFT por autodestilação do base (rejection sampling), GSM8K train[300:300+N].

Fase A: o base gera 1 amostra por problema (raciocínio ligado, teto 6144, mesma amostragem da avaliação).
Fase B (excisão do loop): nas amostras com loop duro (probes.onset no raciocínio), corta no início do loop,
  fecha com "\\n</think>\\n\\n" e deixa o base responder (teto 1024).
Mantém só as respostas CORRETAS:
  natural  raciocínio fechado pelo próprio modelo, sem cortar no teto e sem loop duro
  excised  prefixo pré-loop + fechamento + resposta
Não toca o dev48 (train[0:48]) nem o conjunto de checagem do agente (train[200:216]).
Saída: data/ft/sft.jsonl  {i, kind, prompt, completion, n_completion_tokens}

uso: scripts/run_bg.sh ft-data scripts/ft_data.py 600
"""

import json
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

from wsp import evalrun, gen, llama_gen, probes
from wsp.paths import BASE_REPO, BASE_REV, DATA, snapshot

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
OFF, CAP, NP = 300, 6144, 32
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CLOSE = "\n</think>\n\n"
tok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))
ds = load_dataset("openai/gsm8k", "main", split="train").select(range(OFF, OFF + N))
gold = [float(a.split("####")[-1].replace(",", "").strip()) for a in ds["answer"]]
prompts = [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) for q in ds["question"]]
evalrun.ledger(kind="ft_data", split=f"gsm8k_train[{OFF}:{OFF + N}]", cap=CAP)


def ok(text, g):
    a = gen.num(gen.last_boxed(text))
    return a is not None and abs(a - g) < 1e-6


outs, _ = evalrun.run("ft_data_a", "base", prompts, CAP, np=NP, sampling=llama_gen.THINK35)
rows, todo = [], []
stats = {"n": N, "natural": 0, "loop": 0, "other": 0}
for i, (o, g) in enumerate(zip(outs, gold)):
    text = o["text"]
    think = text.split("</think>")[0]
    ids = tok(think, add_special_tokens=False)["input_ids"]
    on = probes.onset(ids)
    if on is None and not o["cut"] and "</think>" in text:
        if ok(text, g):
            rows.append({"i": OFF + i, "kind": "natural", "prompt": prompts[i], "completion": text})
            stats["natural"] += 1
        else:
            stats["other"] += 1
    elif on is not None and on >= 128:
        stats["loop"] += 1
        todo.append((i, tok.decode(ids[:on]).rstrip() + CLOSE))
    else:
        stats["other"] += 1
print("fase A", stats, flush=True)

b_prompts = [prompts[i] + pre for i, pre in todo]
b_outs, _ = evalrun.run("ft_data_b", "base", b_prompts, 1024, np=NP, sampling=llama_gen.THINK35,
                        ctx_per_slot=CAP + 2048)
stats["excised"] = 0
for (i, pre), o in zip(todo, b_outs):
    if not o["cut"] and ok(o["text"], gold[i]):
        rows.append({"i": OFF + i, "kind": "excised", "prompt": prompts[i], "completion": pre + o["text"]})
        stats["excised"] += 1
for r in rows:
    r["n_completion_tokens"] = len(tok(r["completion"], add_special_tokens=False)["input_ids"])
out = DATA / "ft" / "sft.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["i"])))
(DATA / "ft" / "sft_stats.json").write_text(json.dumps(stats, indent=1))
print("fim", stats, "->", out, flush=True)
