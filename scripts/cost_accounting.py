"""Cost accounting: agent tokens + GPU-hours for every arm, written to data/cost.json.

Sources:
  - agent transcripts (Claude Code subagent JSONL, local, not versioned): per-message `usage`
  - scorer ledgers (data/m2a/runs/<run>/{ledger,gen_ledger}.jsonl): number of probes and generation checks
  - generation summaries (data/gen_*.json): wall time of each evaluation arm, mean generated tokens
  - E4 timings (from the E4 logs; see CADERNO 2026-11-08)
  - Astra (paper logs, REFS/runs/<run>/turns/*/events.jsonl): per-turn `usage`

usage: uv run python scripts/cost_accounting.py <claude-session-dir-with-subagents/>
"""

import glob
import json
import sys
from pathlib import Path

from wsp.paths import DATA, REFS

SESSION = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# Claude Code subagent id -> run. "killed" = first launch killed by systemd-oomd after ~2 min (2026-11-06 01:17).
AGENTS = {
    "a34e95627e872eda1": "a0", "a3b6cd78adaea851b": "r1",
    "a8bb8af253f1fdecc": "a1", "ae5c024e2e05e9601": "r2",
    "ab1b6aaafbdd79aa9": "a2", "a0e711e43817955e1": "r3",
    "a1a88f35a0bc3be60": "a2_killed", "a9e5c514de314a964": "r3_killed",
}
ATLAS = {"r1", "r2", "r3"}

# API list prices, USD per million tokens (Anthropic, cached 2026-10-06). Cache writes: 1.25x input (5-min TTL)
# or 2x (1-hour TTL); we report both because the TTL Claude Code used is not recorded in the transcript.
PRICE = {"claude-opus-5-5": {"in": 4.0, "out": 20.0, "read": 0.20, "w5m": 5.0, "w1h": 8.0}}

# Scorer timings measured on the 1660S (naive run: 40 probes + 6 gen-checks in 97.8 min, alone on the GPU).
PROBE_S, GEN_MIN = 52, 10.5


def usage(path):
    msgs, model = {}, None
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = r.get("message") or {}
        if r.get("type") == "assistant" and isinstance(m, dict) and m.get("usage"):
            msgs[m.get("id") or len(msgs)] = m["usage"]
            model = m.get("model") or model
    keys = ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]
    tot = {k: sum(u.get(k) or 0 for u in msgs.values()) for k in keys}
    tot["turns"], tot["model"] = len(msgs), model
    return tot


def dollars(u):
    p = PRICE[u["model"]]
    base = (u["input_tokens"] * p["in"] + u["cache_read_input_tokens"] * p["read"]
            + u["output_tokens"] * p["out"]) / 1e6
    return {"usd_5m": round(base + u["cache_creation_input_tokens"] * p["w5m"] / 1e6, 2),
            "usd_1h": round(base + u["cache_creation_input_tokens"] * p["w1h"] / 1e6, 2)}


def lines(p):
    return sum(1 for _ in open(p)) if p.exists() else 0


out = {"assumptions": {"probe_s": PROBE_S, "gen_check_min": GEN_MIN, "price_per_mtok": PRICE}, "agents": {}}
for aid, run in AGENTS.items():
    rd = DATA / "m2a" / "runs" / run.replace("_killed", "")
    probes = 0 if "killed" in run else lines(rd / "ledger.jsonl")
    gens = 0 if "killed" in run else lines(rd / "gen_ledger.jsonl")
    row = {"run": run, "atlas": run[:2] in ATLAS, "probes": probes, "gen_checks": gens,
           "gpu_h": round((probes * PROBE_S / 60 + gens * GEN_MIN) / 60, 2)}
    if SESSION:
        f = SESSION / "subagents" / f"agent-{aid}.jsonl"
        if f.exists():
            u = usage(f)
            row.update(u, **dollars(u))
    out["agents"][run] = row

# controls with no LLM: random (v1) and naive (E3) search
for run in ["rnd", "naive"]:
    rd = DATA / "m2a" / "runs" / run
    p, g = lines(rd / "ledger.jsonl"), lines(rd / "gen_ledger.jsonl")
    out["agents"][run] = {"run": run, "probes": p, "gen_checks": g,
                          "gpu_h": round((p * PROBE_S / 60 + g * GEN_MIN) / 60, 2)}

# E4 (LoRA-SFT): GPU wall-clock per stage, from the E4 logs (CADERNO 2026-11-08)
e4 = {"spike": 0.18, "data_phase_a": 3.09 + 5.93, "data_phase_b": 1.17, "train_export": 6.29,
      "epoch_pick": round((604 + 539 + 578) / 3600, 2)}
e4["total"] = round(sum(e4.values()), 1)
out["e4_gpu_h"] = e4

# Decoding baselines: tuning cost = their dev48 runs (3 logit-bias + 2 budget-forcing configurations)
dev = json.load(open(DATA / "gen_dev48.json"))["arms"]
tune = ["lb_w10_t0", "lbx_w10_t1", "lb_w3_t1", "bf4096", "bf8192"]
out["decoding_tuning_gpu_h"] = round(sum(dev[a]["summary"]["secs_this_run"] for a in tune) / 3600, 2)

# Serving cost: mean generated tokens per problem (dev48, 12k cap) and MATH-148 where available
serve = {a: round(dev[a]["summary"]["mean_tokens"]) for a in
         ["base", "astra", "lbx_w10_t1", "bf4096", "m2a_naive", "ft_sft_ep2", "m2a_a1_062959_102", "m2a_a2",
          "m2a_a1_062959_102+bf4096"]}
out["dev48_mean_tokens"] = serve
math = {}
for tag in ["math48", "math100b"]:
    d = json.load(open(DATA / f"gen_{tag}.json"))["arms"]
    for a, v in d.items():
        n = len(v["rows"])
        t = sum(r["n_tokens"] for r in v["rows"])
        math.setdefault(a, [0, 0])
        math[a][0] += t
        math[a][1] += n
out["math_mean_tokens"] = {a: {"n": n, "mean": round(t / n)} for a, (t, n) in math.items()}

# Astra (paper, GPT-6 through the Codex CLI): per-turn usage from the public logs
astra = {}
for run in ["p10c", "p25c", "p50c"]:
    tot, n = {}, 0
    for f in sorted(glob.glob(str(REFS / "runs" / run / "turns" / "t*" / "events.jsonl"))):
        for line in open(f):
            if '"usage"' not in line:
                continue
            e = json.loads(line)
            u = e.get("usage") or (e.get("turn") or {}).get("usage") or e.get("msg", {}).get("usage")
            if not isinstance(u, dict):
                continue
            n += 1
            for k, v in u.items():
                if isinstance(v, (int, float)):
                    tot[k] = tot.get(k, 0) + v
    astra[run] = {"turns_with_usage": n, **tot}
out["astra"] = astra

if SESSION:
    main = SESSION.with_suffix(".jsonl")
    if main.exists():
        u = usage(main)
        out["supervisor_session"] = {**u, **dollars(u)}

(DATA / "cost.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
