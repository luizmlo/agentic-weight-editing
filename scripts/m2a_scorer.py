"""M→A, fase A: avaliador de edições (daemon). Um modelo base fp16 na GPU, sempre carregado.

Fila em arquivo: data/m2a/runs/<run>/inbox/<id>.pt  ->  outbox/<id>.json (o agente lê)
                                                        ->  sealed/<id>.json (holdout; só o supervisor)
Orçamento: runs/<run>/config.json {"budget": N}; cada avaliação válida debita 1 (runs/<run>/ledger.jsonl).
Inválidas não debitam.

Métricas (Δlogp editado − base, nats/token, nos tokens da continuação):
  branch_rep / branch_prod   12 pares casados de busca (6 do relatório do Astra + 6 próprios)
  close                      Δlogp de "</think>" logo após o prefixo (positivo = incentiva parar cedo demais)
  neutral                    WikiText-2, 8 blocos de 1024 tokens
  holdout_rep / holdout_prod 12 pares casados novos, LACRADOS (sealed/)
Validação externa (PoC-8): branch_rep tem Spearman 0,89 com a acurácia e 0,96 com (−loop no teto)
em 10 edições geradas no GSM8K com teto de 12k.

v2 (2026-11-05): checagem por GERAÇÃO (m2a.submit_gen, orçamento separado "gen_budget").
  O modelo HF sai da GPU; o candidato vira GGUF (poc7_build) e o llama-server (porta 8091, estado
  próprio) gera GSM8K train[200:216] com raciocínio, teto 6k, presence 1.5 no output inteiro.
  Devolve acc, cut (fração cortada no teto), closed_think, tokens e os finais de 3 respostas cortadas;
  a referência do base no mesmo conjunto vem junto (calculada uma vez e guardada em m2a/gen_ref_base.json).
Motivo: a sonda branch sofreu Goodhart entre famílias de edição (CADERNO, 2026-11-04 noite).

uso: scripts/run_bg.sh m2a scripts/m2a_scorer.py
"""

import json
import os
import subprocess
import time
import traceback

from wsp.paths import DATA

# servidor de geração próprio (não colide com as filas de avaliação na porta 8090); setar antes de importar llama_gen
os.environ.setdefault("WSP_LLAMA_URL", "http://127.0.0.1:8091")
os.environ.setdefault("WSP_LLAMA_PORT", "8091")
os.environ.setdefault("WSP_LLAMA_DIR", str(DATA / "llama_m2a"))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from wsp import evalrun, gen, llama_gen, m2a, modelio, probes, tf
from wsp.paths import BASE_REPO, BASE_REV, ROOT, snapshot

model, tok = modelio.load()
CASES = probes.ASTRA_CASES + probes.OWN_CASES
HOLD = json.loads((m2a.M2A / "sealed" / "holdout_cases.json").read_text())


def prep(items):
    for it in items:
        it["base"] = tf.read(model, it["ids"])["logp"]
    return items


search = prep(probes.branch_items(tok, CASES))
close = prep(probes.branch_items(tok, CASES[:4], conds=("close",)))
hold = prep(probes.branch_items(tok, HOLD))
wt = "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
wt_ids = tok(wt, add_special_tokens=False)["input_ids"]
neutral = prep([{"ids": torch.tensor(wt_ids[j * 1024:(j + 1) * 1024]), "mask": None} for j in range(8)])

GEN_MAX, GEN_NP = 6144, 16
_gds = load_dataset("openai/gsm8k", "main", split="train").select(range(200, 216))
_htok = AutoTokenizer.from_pretrained(snapshot(BASE_REPO, BASE_REV))
GEN_PROMPTS = [_htok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your "
                                           "final answer within \\boxed{}."}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True) for q in _gds["question"]]
GEN_GOLD = [float(a.split("####")[-1].replace(",", "").strip()) for a in _gds["answer"]]
print("avaliador pronto", flush=True)


def gen_check(gguf):
    """Gera com o GGUF dado (HF fora da GPU durante a geração)."""
    model.to("cpu")
    torch.cuda.empty_cache()
    try:
        outs = llama_gen.generate(gguf, GEN_PROMPTS, GEN_MAX, concurrency=GEN_NP, penalty_last_n=GEN_MAX + 512,
                                  sampling=llama_gen.THINK35)
    finally:
        subprocess.run([str(ROOT / "scripts/llama_serve.sh"), "stop"], check=False)
        model.to("cuda")
    rows = []
    for o, g in zip(outs, GEN_GOLD):
        a = gen.num(gen.last_boxed(o["text"]))
        rows.append({"correct": a is not None and abs(a - g) < 1e-6, "cut": o["cut"], "n": o["n_tokens"],
                     "closed": "</think>" in o["text"], "tail": o["text"][-300:]})
    n = len(rows)
    return {"acc": sum(r["correct"] for r in rows) / n, "cut": sum(r["cut"] for r in rows) / n,
            "closed_think": sum(r["closed"] for r in rows) / n, "mean_tokens": sum(r["n"] for r in rows) / n,
            "correct": [int(r["correct"]) for r in rows],
            "cut_tails": [r["tail"] for r in rows if r["cut"]][:3]}


def gen_ref():
    f = m2a.M2A / "gen_ref_base.json"
    if not f.exists():
        f.write_text(json.dumps(gen_check(evalrun.model_of("base"))))
    return json.loads(f.read_text())


def handle_gen(rd, req):
    cfg = json.loads((rd / "config.json").read_text())
    led = rd / "gen_ledger.jsonl"
    used = len(led.read_text().splitlines()) if led.exists() else 0
    if used >= cfg.get("gen_budget", 0):
        raise RuntimeError(f"orçamento de geração esgotado ({used}/{cfg.get('gen_budget', 0)})")
    cid = req["cid"]
    ref = gen_ref()
    name = f"m2a_tmp_{rd.name}_{cid}"
    subprocess.run(["uv", "run", "--no-sync", "python", "-u", str(ROOT / "scripts/poc7_build.py"), name,
                    str(rd / "done" / f"{cid}.pt")], check=True, cwd=ROOT)
    gguf = m2a.DATA / "gguf" / f"{name}.gguf"
    try:
        r = gen_check(str(gguf))
    finally:
        gguf.unlink(missing_ok=True)
        (m2a.DATA / "gguf" / f"{name}.json").unlink(missing_ok=True)
    both = [(b, a) for b, a in zip(ref["correct"], r["correct"])]
    r.update({"cid": cid, "base_acc": ref["acc"], "base_cut": ref["cut"],
              "rescues": sum(a and not b for b, a in both), "losses": sum(b and not a for b, a in both),
              "set": "GSM8K train[200:216], thinking, cap 6144", "gen_used": used + 1,
              "gen_budget": cfg.get("gen_budget", 0)})
    with led.open("a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%F %T"), "cid": cid, "acc": r["acc"], "cut": r["cut"]}) + "\n")
    return r


def delta(it):
    d = tf.read(model, it["ids"])["logp"] - it["base"]
    return (d if it["mask"] is None else d[it["mask"]]).mean().item()


def mean(xs):
    return sum(xs) / len(xs)


def score(edits):
    undo, rel = m2a.apply(model, edits)
    try:
        rep = [delta(it) for it in search if it["cond"] == "repetitive"]
        prod = [delta(it) for it in search if it["cond"] == "productive"]
        r = {"branch_rep": mean(rep), "branch_prod": mean(prod), "branch_sel": mean(rep) - mean(prod),
             "per_case_rep": [round(x, 4) for x in rep], "close": mean([delta(it) for it in close]),
             "neutral": mean([delta(it) for it in neutral])}
        s = {"holdout_rep": mean([delta(it) for it in hold if it["cond"] == "repetitive"]),
             "holdout_prod": mean([delta(it) for it in hold if it["cond"] == "productive"])}
    finally:
        undo()
    return r, s, rel


while True:
    (m2a.M2A / "scorer.alive").write_text(time.strftime("%F %T"))
    for f in sorted(m2a.RUNS.glob("*/inbox/*.pt")):
        rd = f.parent.parent
        cid = f.stem
        t = time.time()
        out = {"id": cid}
        try:
            cfg = json.loads((rd / "config.json").read_text())
            led = rd / "ledger.jsonl"
            used = len(led.read_text().splitlines()) if led.exists() else 0
            req = torch.load(f)
            if req.get("kind") == "gen":
                out.update(handle_gen(rd, req))
                out["secs"] = round(time.time() - t, 1)
                raise StopIteration
            out["note"] = req.get("note", "")
            edits = m2a.validate(req["edits"])
            if used >= cfg["budget"]:
                raise RuntimeError(f"orçamento esgotado ({used}/{cfg['budget']})")
            r, s, rel = score(edits)
            out.update(r)
            out["edits"] = [{"tensor": e["tensor"], "alpha": e["alpha"],
                             "n_cols": None if e["cols"] is None else int(e["cols"].numel()), "rel_change": round(x, 4)}
                            for e, x in zip(edits, rel)]
            out["n_used"], out["budget"] = used + 1, cfg["budget"]
            out["secs"] = round(time.time() - t, 1)
            (rd / "sealed").mkdir(exist_ok=True)
            (rd / "sealed" / f"{cid}.json").write_text(json.dumps({"id": cid, **s}))
            with led.open("a") as fh:
                fh.write(json.dumps({"ts": time.strftime("%F %T"), "id": cid, "branch_rep": r["branch_rep"]}) + "\n")
            (rd / "done").mkdir(exist_ok=True)
            torch.save({"edits": edits, "note": out["note"]}, rd / "done" / f"{cid}.pt")
        except StopIteration:
            pass
        except Exception as e:
            out["error"] = repr(e)
            traceback.print_exc()
        (rd / "outbox" / f"{cid}.json").write_text(json.dumps(out, indent=1))
        f.unlink()
        print(rd.name, cid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()
                             if k in ("branch_rep", "branch_prod", "close", "neutral", "error", "acc", "cut")}, flush=True)
    time.sleep(1)
