"""Avaliador de edições para o SEGUNDO defeito (seguir instruções de formato, IFEval, sem raciocínio). Daemon.

Mesmo protocolo do avaliador v2 (scripts/m2a_scorer.py): fila em arquivo, orçamento por run, holdout lacrado,
sondas teacher-forced baratas + checagens por geração caras. Runs em data/m2a_if/runs/<run>/ (config.json com
"budget" e "gen_budget"). Ferramenta do agente: wsp.m2a_if.

Sondas (wsp.ifprobes; Δ = editado − base, nats por token, média por par):
  if_margin       Δ(logp da versão que cumpre − logp da que não cumpre), prompt COM a instrução (8 tópicos × 8
                  instruções). Positivo = a edição passa a preferir cumprir.
  per_constraint  if_margin por instrução
  ctrl_margin     o mesmo, com o prompt SEM a instrução (3 tópicos × 8). Deve ficar ~0 (positivo = aplica o formato
                  sempre)
  neutral         Δlogp em WikiText-2 (4 blocos de 1024 tokens)
  holdout_margin  if_margin em 3 tópicos novos, LACRADO (sealed/)
Checagem por geração (submit_gen): conjunto escolhido por "gen_set" no config.json do run, sem raciocínio, teto 2048,
amostragem TOOLS, verificador oficial:
  ifeval48  IFEval dev[0:48] (split fixo de scripts/ifeval_eval.py); padrão, usado em sanity e b1
  synth128  128 prompts sintéticos de wsp.ifsynth (sementes 50000..50127; disjuntas das do FT, 10000+); b2 em diante.
            Não toca o IFEval, então a confirmação pode usar o dev200 inteiro. Sem versão "loose" (loose = strict). O HF sai da GPU, o candidato vira GGUF (poc7_build) e o llama-server
(porta 8091) gera.

uso: scripts/run_bg.sh m2a-if scripts/m2a_scorer_if.py
"""

import importlib.util
import json
import os
import subprocess
import time
import traceback

from wsp.paths import DATA

os.environ.setdefault("WSP_LLAMA_URL", "http://127.0.0.1:8091")
os.environ.setdefault("WSP_LLAMA_PORT", "8091")
os.environ.setdefault("WSP_LLAMA_DIR", str(DATA / "llama_m2a"))

import torch
from datasets import load_dataset

from wsp import evalrun, ifprobes, ifsynth, llama_gen, m2a, m2a_if, modelio, tf
from wsp.paths import ROOT

_spec = importlib.util.spec_from_file_location("ifeval_eval", ROOT / "scripts" / "ifeval_eval.py")
ife = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ife)

model, tok = modelio.load()
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0


@torch.no_grad()
def batch_mean_logp(seqs, bs=16):
    """[(ids, mask)] -> [logp médio por token nas posições do mask], em lotes com padding à DIREITA (modelo causal:
    as posições reais não enxergam o padding, então o resultado é o de cada sequência sozinha)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out = [0.0] * len(seqs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(seqs[i][0]) for i in idx)
        x = torch.full((len(idx), L), PAD, dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, :len(seqs[i][0])] = seqs[i][0]
        h = model.model(input_ids=x.to(model.device), use_cache=False).last_hidden_state
        for j, i in enumerate(idx):
            ids, m = seqs[i]
            pos = m.nonzero().squeeze(-1).to(model.device)
            lp = model.lm_head(h[j, pos]).float().log_softmax(-1)
            out[i] = lp.gather(-1, ids.to(model.device)[pos + 1, None]).mean().item()
    return out


def pair_margins(items):
    lp = batch_mean_logp([it[k] for it in items for k in ("ok", "bad")])
    return [lp[2 * i] - lp[2 * i + 1] for i in range(len(items))]


def prep_pairs(items):
    for it, mg in zip(items, pair_margins(items)):
        it["m_base"] = mg
    return items


search = prep_pairs(ifprobes.pairs(tok, ifprobes.SEARCH_TOPICS, True))
control = prep_pairs(ifprobes.pairs(tok, ifprobes.SEARCH_TOPICS[:3], False))  # 3 tópicos bastam e o forward é lento (~600 tok/s)
hold = prep_pairs(ifprobes.pairs(tok, ifprobes.HOLDOUT_TOPICS, True))
wt = "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
wt_ids = tok(wt, add_special_tokens=False)["input_ids"]
neutral = [{"ids": torch.tensor(wt_ids[j * 1024:(j + 1) * 1024])} for j in range(4)]
for it in neutral:
    it["base"] = tf.read(model, it["ids"])["logp"]
BASE_MARGIN = {}
for it in search:
    BASE_MARGIN.setdefault(it["constraint"], []).append(it["m_base"])
BASE_MARGIN = {k: round(sum(v) / len(v), 4) for k, v in BASE_MARGIN.items()}

GEN_MAX, GEN_NP = 2048, 16
GEN_DESC = {"ifeval48": "IFEval practice prompts (48), thinking off, cap 2048",
            "synth128": "IFEval-style practice prompts (128; our own writing tasks + the official instruction "
                        "generators), thinking off, cap 2048"}
_GEN = {}


def gen_set(name):
    """-> (itens, prompts com chat template, texto de cada prompt)"""
    if name not in _GEN:
        if name == "synth128":
            items = [ifsynth.make(50000 + i) for i in range(128)]
            texts = [it["prompt"] for it in items]
        elif name == "ifeval48":
            items = ife.split("dev", 48, 0)
            texts = [x.prompt for x in items]
        else:
            raise ValueError(f"gen_set desconhecido: {name}")
        prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for t in texts]
        _GEN[name] = (items, prompts, texts)
    return _GEN[name]


def grade_any(x, text, cut):
    if isinstance(x, dict):  # wsp.ifsynth
        st = ifsynth.check(x, text.split("</think>", 1)[-1].strip())
        ok = all(st) and not cut
        return {"key": x["seed"], "ids": x["ids"], "strict": st, "loose": st, "prompt_strict": ok, "prompt_loose": ok}
    return ife.grade(x, text, cut)
print("avaliador IF pronto; margem base por instrução:", BASE_MARGIN, flush=True)


def gen_check(gguf, set_name="ifeval48"):
    items, prompts, texts = gen_set(set_name)
    model.to("cpu")
    torch.cuda.empty_cache()
    try:
        outs = llama_gen.generate(gguf, prompts, GEN_MAX, concurrency=GEN_NP, ctx_per_slot=GEN_MAX + 1024,
                                  penalty_last_n=64, sampling=llama_gen.TOOLS)
    finally:
        subprocess.run([str(ROOT / "scripts/llama_serve.sh"), "stop"], check=False)
        model.to("cuda")
    rows = [{**grade_any(x, o["text"], o["cut"]), "cut": o["cut"], "n": o["n_tokens"], "text": o["text"]}
            for x, o in zip(items, outs)]
    s = ife.summarize(rows)
    fails = [r for r in rows if not r["prompt_strict"]]
    ex = []
    for r, t in [(r, t) for r, t in zip(rows, texts) if not r["prompt_strict"]][:4]:
        ex.append({"prompt": t[:400], "failed": [i for i, ok in zip(r["ids"], r["strict"]) if not ok],
                   "cut": r["cut"], "response_start": r["text"][:300], "response_end": r["text"][-200:]})
    return {"prompt_strict": s["prompt_strict"], "inst_strict": s["inst_strict"], "prompt_loose": s["prompt_loose"],
            "per_group_strict": s["per_group_strict"], "cut": sum(r["cut"] for r in rows) / len(rows),
            "mean_tokens": sum(r["n"] for r in rows) / len(rows), "correct": [int(r["prompt_strict"]) for r in rows],
            "n_fail": len(fails), "fail_examples": ex}


def gen_ref(set_name="ifeval48"):
    f = m2a_if.M2A / ("gen_ref_base.json" if set_name == "ifeval48" else f"gen_ref_base_{set_name}.json")
    if not f.exists():
        f.write_text(json.dumps(gen_check(evalrun.model_of("base"), set_name)))
    return json.loads(f.read_text())


def handle_gen(rd, req):
    cfg = json.loads((rd / "config.json").read_text())
    led = rd / "gen_ledger.jsonl"
    used = len(led.read_text().splitlines()) if led.exists() else 0
    if used >= cfg.get("gen_budget", 0):
        raise RuntimeError(f"orçamento de geração esgotado ({used}/{cfg.get('gen_budget', 0)})")
    cid = req["cid"]
    set_name = cfg.get("gen_set", "ifeval48")
    ref = gen_ref(set_name)
    name = f"m2aif_tmp_{rd.name}_{cid}"
    subprocess.run(["uv", "run", "--no-sync", "python", "-u", str(ROOT / "scripts/poc7_build.py"), name,
                    str(rd / "done" / f"{cid}.pt")], check=True, cwd=ROOT)
    gguf = DATA / "gguf" / f"{name}.gguf"
    try:
        r = gen_check(str(gguf), set_name)
    finally:
        gguf.unlink(missing_ok=True)
        (DATA / "gguf" / f"{name}.json").unlink(missing_ok=True)
    both = list(zip(ref["correct"], r["correct"]))
    r.update({"cid": cid, "base_prompt_strict": ref["prompt_strict"], "base_inst_strict": ref["inst_strict"],
              "base_cut": ref["cut"], "rescues": sum(a and not b for b, a in both),
              "losses": sum(b and not a for b, a in both),
              "set": GEN_DESC[set_name], "gen_used": used + 1,
              "gen_budget": cfg.get("gen_budget", 0)})
    with led.open("a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%F %T"), "cid": cid, "prompt_strict": r["prompt_strict"],
                             "cut": r["cut"]}) + "\n")
    return r


def margin_delta(items):
    return [(it["constraint"], mg - it["m_base"]) for it, mg in zip(items, pair_margins(items))]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def score(edits):
    undo, rel = m2a.apply(model, edits)
    try:
        sd = margin_delta(search)
        per = {}
        for c, x in sd:
            per.setdefault(c, []).append(x)
        r = {"if_margin": mean(x for _, x in sd), "per_constraint": {c: round(mean(v), 4) for c, v in per.items()},
             "ctrl_margin": mean(x for _, x in margin_delta(control)),
             "neutral": mean((tf.read(model, it["ids"])["logp"] - it["base"]).mean().item() for it in neutral),
             "base_margin": BASE_MARGIN}
        r["if_sel"] = r["if_margin"] - r["ctrl_margin"]
        s = {"holdout_margin": mean(x for _, x in margin_delta(hold))}
    finally:
        undo()
    return r, s, rel


while True:
    (m2a_if.M2A / "scorer.alive").write_text(time.strftime("%F %T"))
    for f in sorted(m2a_if.RUNS.glob("*/inbox/*.pt")):
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
                fh.write(json.dumps({"ts": time.strftime("%F %T"), "id": cid, "if_margin": r["if_margin"]}) + "\n")
            (rd / "done").mkdir(exist_ok=True)
            torch.save({"edits": edits, "note": out["note"]}, rd / "done" / f"{cid}.pt")
        except StopIteration:
            pass
        except Exception as e:
            out["error"] = repr(e)
            traceback.print_exc()
        (rd / "outbox").mkdir(exist_ok=True)
        (rd / "outbox" / f"{cid}.json").write_text(json.dumps(out, indent=1))
        f.unlink()
        print(rd.name, cid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()
                             if k in ("if_margin", "ctrl_margin", "neutral", "error", "prompt_strict", "cut")}, flush=True)
    time.sleep(1)
