# Agentic Weight Editing

**Can a frontier-model agent repair a specific behavioral defect in a small language model by editing its weights
directly, with no gradients, and do it better and cheaper than fine-tuning?**

This repository holds the code, experiment records and research notes for a local, low-budget follow-up to
*Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026). In that paper, GPT-6 "Astra"
edited 0.016% of the weights of Qwen3.5-0.8B and lifted MATH-500 from 7.9% to 48.1% by removing a direction that
sustains repetition loops in the model's reasoning. Here we ask whether this is reproducible, whether it is more than a
decoding trick, and what part of the agent's process matters, all on a single consumer GPU (GTX 1660 SUPER, 6 GB).

> Status: **pre-paper research notes**. All numbers are small-n (32–148 problems per evaluation) and come from one
> model and one defect. The fine-tuning baseline (E4) is still running. See [docs/research-notes.md](docs/research-notes.md).

## The defect

With thinking enabled, Qwen3.5-0.8B tends to fall into **reasoning loops** ("Wait, ... Yes, ... Wait, ...") and never
closes `</think>`. On GSM8K with a 12k-token cap, the base model solves 14.6% of dev problems and **85% of its
generations hit the cap**. Prompting does not fix it: anti-repetition system prompts perform the same as a placebo prompt.

## The intervention

Agents (Claude Opus 5.5 subagents, supervised by Claude Code) get the model weights, a budgeted **scorer daemon** and a
brief. The edit space is at most **4 rank-1, output-side edits**:

```
W  <-  W - alpha * u (mask ⊙ uᵀ W)          on mlp.down_proj, linear_attn.out_proj or self_attn.o_proj,  |alpha| <= 10
```

The scorer returns cheap teacher-forced probe metrics (40 calls per run) and, in protocol v2, **6 generation checks**
on 16 held-out GSM8K training problems. A supervisor then confirms the agent's top candidate on a separate dev set
(GSM8K train[0:48]) and runs the pre-registered winner once on a locked test split.

## Headline results (as of 2026-11-08)

Accuracy with thinking on and a 12,288-token cap. "a1" is the edit the winning agent found: it suppresses the
"Wait" token through 3 MLP neurons and strengthens the `</think>` write in layer 23, using 4 rank-1 edits.

| arm | GSM8K dev48 | GSM8K locked test (32) | MATH-500 (148) |
|---|---|---|---|
| base | 0.146 | 0.125 | 0.081 |
| Astra edit (paper) | 0.458 | 0.406 | 0.209 |
| logit bias (best of 3, tuned on dev48) | 0.521 | 0.469 | 0.271¹ |
| budget forcing at 4096 (best of 2, tuned on dev48) | 0.479 | **0.625** | 0.372 |
| **agent edit a1** | **0.667** | 0.562 | **0.466** |
| **a1 + budget forcing** | **0.688** | **0.625** | **0.527** |

¹ MATH-500[0:48] only.

Paired differences on MATH-500 (148 problems): a1 vs Astra **+25.7 ± 4.3 pp** (44 rescues / 6 losses); a1 vs budget
forcing **+9.5 ± 3.8 pp** (23/9); combo vs a1 **+6.1 ± 3.2 pp** (16/7). The search ran on GSM8K only, so MATH-500 is
out-of-distribution transfer.

What we have learned so far:

1. **Agents find a strong, interpretable edit, and the result replicates.** Two independent no-atlas agents running protocol v2 (a1, a2)
   converged on the same mechanism (shared "Wait" neurons L17n1915 and L22n573, plus a `</think>` lever) and reached
   0.667 / 0.646 on dev48.
2. **It is not just a decoding trick.** It beats a tuned logit bias everywhere. It ties budget forcing on GSM8K but
   beats it on the longer MATH problems, and combining the two helps further.
3. **The agent's reasoning matters.** A naive search with the same tools, the same budget and the same generation
   signal reaches 0.271 on dev48 (−38 pp).
4. **Generation in the loop is essential.** Every cheap proxy (teacher-forced Δlog-prob, branch probes, the 8-gram loop
   detector) was Goodharted by a strong optimizer at least once.
5. **More interpretability context hurt.** Agents given an activation atlas found "internal-state" mechanisms that looked
   good on probes and generalized worse in generation. This held in 3 of 3 paired replicates.

## Repository layout

```
src/wsp/            library: model I/O, probes, generation (llama-server client), evaluation runner, agent scorer API
scripts/            experiments (poc*, m2a_*, ft_*), evaluation (gen_eval, math_eval, tool_eval), queues (queue_*.sh)
docs/
  research-notes.md   findings, with evidence and caveats (start here)
  experiments.md      registry of every experiment: question, pre-registered prediction, outcome, data
  paper-draft.md      pre-paper outline: framing, contributions, sections, figures, TODO
  reproducing.md      environment, hardware gotchas, how to run each stage
  CADERNO.md          original lab notebook (pt-BR), timestamped pre-registrations; the primary record
  PLANO.md            original experiment plan (pt-BR)
data/               only small records are versioned (see .gitignore): result summaries (*.json), agent briefs,
                    per-run scorer ledgers / final picks / agent notes, every submitted edit
                    (`m2a/runs/*/done/*.pt`), E4 configs and logs
```

Raw generations (~210 MB), GGUF models (~31 GB) and agent work directories stay local and will be released separately.

## Quick start

```bash
uv sync                                   # pinned stack: torch 2.14.1+cu130, transformers 5.22.1, fla 0.6.0, peft 0.21.2
scripts/run_bg.sh <name> <script> [args]  # every long job runs as a systemd --user unit (wsp-<name>), log in data/logs/
```

See [docs/reproducing.md](docs/reproducing.md) for the llama-server setup, the GGUF conversion path and each experiment's command.

## Provenance note

The project began inside a larger harness (`deep_traces`), so historical records (CADERNO, agent briefs, agent notes,
ledgers) refer to the old paths `src/deep_traces/wsp/`, `scripts/wsp/` and `data/wsp/`. In this repository these are
`src/wsp/`, `scripts/` and `data/`.
