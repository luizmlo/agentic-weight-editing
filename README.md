# Agentic Weight Editing

**Can a frontier-model agent repair a specific behavioral defect in a small language model by editing its weights
directly, with no gradients, and do it better and cheaper than fine-tuning?**

This repository holds the code, experiment records and research notes for a local, low-budget follow-up to
*Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026). In that paper, GPT-6 "Astra"
edited 0.016% of the weights of Qwen3.5-0.8B and lifted MATH-500 from 7.9% to 48.1% by removing a direction that
sustains repetition loops in the model's reasoning. Here we ask whether this is reproducible, whether it is more than a
decoding trick, and what part of the agent's process matters, all on a single consumer GPU (GTX 1660 SUPER, 6 GB).

> Status: **pre-paper research notes**. All numbers are small-n (32–148 problems per evaluation) and come from one
> model and one defect. See [docs/research-notes.md](docs/research-notes.md).

## The defect

With thinking enabled, Qwen3.5-0.8B tends to fall into **reasoning loops** ("Wait, ... Yes, ... Wait, ...") and never
closes `</think>`. On GSM8K with a 12k-token cap, the base model solves 14.6% of dev problems and **85% of its
generations hit the cap**. Prompting does not fix it: anti-repetition system prompts perform the same as a placebo prompt.

## The intervention

Agents (Claude Opus 5.5 subagents, supervised by Claude Code) get the model weights, a budgeted **scorer daemon** and a
brief. They may change the model with at most **4 edits**. Each edit targets one output projection: an MLP
`down_proj`, a linear-attention `out_proj` or an attention `o_proj`. Such a matrix $W \in \mathbb{R}^{d \times k}$ maps
$k$ neurons (or head channels) into the residual stream, and each column $w_j$ is what neuron $j$ writes there. An edit
picks a unit direction $u \in \mathbb{R}^{d}$ in the residual stream, a set $S$ of columns and a strength $\alpha$, and
changes only the part of those columns that lies along $u$:

$$
w_j \leftarrow w_j - \alpha \,(u^\top w_j)\, u \quad \text{for } j \in S, \qquad |\alpha| \le 10 .
$$

So the selected neurons' write along $u$ is scaled by $(1-\alpha)$:

- $\alpha = 1$ removes it;
- $\alpha = 6$ flips it to $-5\times$ (the Astra edit);
- $\alpha < 0$ amplifies it.

Everything else in $W$ is untouched, and each edit is a rank-1 update.

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
| LoRA-SFT on self-distilled traces (E4, ~17 GPU-h) | 0.292 | — | — |
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
5. **It beats fine-tuning at a fraction of the compute.** LoRA-SFT on 175 self-distilled traces (17 GPU-hours)
   reaches 0.292 on dev48, 37.5 pp below a1 (0 rescues, 18 losses). An agent run uses 1.3 GPU-hours plus about $5–7
   of agent tokens at API prices. The edited model also generates 60% fewer tokens per problem. See
   [docs/cost.md](docs/cost.md).
6. **More interpretability context hurt.** Agents given an activation atlas found "internal-state" mechanisms that looked
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
  cost.md             cost study: GPU-hours, agent tokens, serving cost, comparison with Astra
  CADERNO.md          original lab notebook (pt-BR), timestamped pre-registrations; the primary record
  PLANO.md            original experiment plan (pt-BR)
data/               only small records are versioned (see .gitignore): result summaries (*.json), agent briefs,
                    per-run scorer ledgers / final picks / agent notes, every submitted edit
                    (`m2a/runs/*/done/*.pt`), E4 configs and logs
```

Raw generations (~210 MB), GGUF models (~31 GB) and agent work directories stay local and will be released separately.
Agent briefs and notes are kept verbatim, as the agents saw them (see [data/m2a/README.md](data/m2a/README.md)).

## Quick start

```bash
uv sync                                   # pinned stack: torch 2.14.1+cu130, transformers 5.22.1, fla 0.6.0, peft 0.21.2
scripts/run_bg.sh <name> <script> [args]  # every long job runs as a systemd --user unit (wsp-<name>), log in data/logs/
```

See [docs/reproducing.md](docs/reproducing.md) for the llama-server setup, the GGUF conversion path and each experiment's command.
