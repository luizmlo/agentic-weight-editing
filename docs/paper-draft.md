# Pre-paper draft

Working document for paper 1. The findings and their evidence are in [research-notes.md](research-notes.md). This
file holds the framing, the structure and the plan to submission.

## Working title

The claim in the title depends on E4 (the fine-tuning baseline). Candidates:

1. **Agents as Weight-Space Programmers: Repairing Reasoning Loops in a Small Language Model with Four Rank-1 Edits**
   (safe whatever E4 shows).
2. *Agentic Weight Editing as a Fine-Tuning Substitute for Specific Behavioral Defects* (only if E4 shows the edit
   ≥ LoRA-SFT at a fraction of the cost; otherwise "complement", not "substitute").
3. *Generation in the Loop: What Makes Agentic Weight Editing Work* (process-centric, if E4 is a wash).

## Abstract (draft, numbers as of 2026-11-08)

> Recent work showed that a frontier model can improve a small language model by editing its weights directly. We
> study this "weight-space programming" in a controlled, low-budget setting: Qwen3.5-0.8B, whose reasoning frequently
> collapses into loops, and agents limited to four rank-1 output-side edits, a budgeted scorer and a single consumer
> GPU. Two independent agents converge on the same interpretable mechanism: suppressing the "Wait" token through three
> MLP neurons while strengthening the `</think>` write in the last layer. They lift GSM8K accuracy from 15% to 65–67%
> on a dev set and transfer to MATH-500 (8% → 47%), outperforming the published edit (21%), tuned logit biasing and
> tuned budget forcing. The edit also composes with budget forcing (53%). Controls isolate what matters. A naive search
> with the same tools and budget reaches only 27%. Every cheap proxy we tried (teacher-forced log-probabilities,
> branch-point probes, n-gram loop detectors) was exploited by at least one agent, and only generation in the loop
> prevented this. Surprisingly, giving agents a richer activation atlas led them to mechanistically interesting but
> behaviorally weaker edits in 3 of 3 paired replicates. [E4: comparison with LoRA fine-tuning on self-distilled data,
> at X× the compute.]

## Contributions

1. **A controlled replication** of agentic weight editing on a consumer GPU, with the reference edit reverse-engineered
   exactly (rank-1, shared direction across two layers) and reproduced behaviorally.
2. **A stronger, interpretable, replicated edit.** Two independent agents find it; it is analyzed by ablation (the
   "way back in / way out" synergy) and transfers out of distribution.
3. **Baselines that rule out "just decoding."** The edit beats logit bias everywhere and beats budget forcing on long
   problems, and the two compose. [+ the fine-tuning baseline, E4.]
4. **Process findings for agentic model editing.** (a) Proxy objectives are Goodharted by strong optimizers;
   generation in the loop is necessary. (b) The agent's reasoning, not the toolkit, carries the result (naive-search
   control). (c) More interpretability context (an activation atlas) can hurt.

## Outline

1. **Introduction.** The defect: loops in small reasoning models, which prompting does not fix (F3). Agentic weight
   editing as an alternative to fine-tuning. Questions: is it reproducible, is it more than decoding, and what in the
   process matters?
2. **Background and related work.** Weight-space programming (Takehi et al. 2026). Model editing (ROME, MEMIT,
   rank-1/ablation edits, refusal-direction ablation). Activation steering (ActAdd, representation engineering).
   Repetition and overthinking in reasoning models: NoWait, budget forcing (s1), length control. LLM agents doing
   interpretability research. Reward hacking and Goodhart in automated optimization.
3. **Setup.** Model and defect; edit space; scorer (probes, gen-check, budgets, sealed holdout, locked test); agent
   briefs and honor-system restrictions with audit; the supervisor protocol (pre-registration, dev confirmation, a
   single locked-test run).
4. **Dissecting the reference edit.** Exact rank-1 recovery; dose-response; the proxy/behavior dissociation
   (PoC-7/8), which motivates the scorer design.
5. **Agents.** v1 (probes): a0 matches Astra; r1's Goodhart. v2 (generation in the loop): a1/a2 and their convergence;
   mechanism and ablations.
6. **Baselines and controls.** E1 (logit bias, budget forcing), the combo, E3 (naive search), E4 (LoRA-SFT), cost.
7. **The atlas paradox (H4).** 3/3 pairs; what the atlas agents found and why it fails in generation.
8. **Limitations.** n, one model and one defect, protocol advantage, sampler, honor system.
9. **Conclusion.**

## Tables and figures plan

| # | content | source |
|---|---|---|
| T1 | Main results: base / Astra / logit bias / budget forcing / LoRA / a1 / combo × dev48 / t12k / MATH-148 | `gen_*.json` |
| T2 | Paired comparisons with rescues/losses | `gen_*.json` (`evalrun.paired`) |
| T3 | Agent runs: protocol, atlas yes/no, calls used, mechanism, gen-check, dev48 | `m2a/runs/*/final.json` |
| T4 | Proxy vs behavior: branch_rep vs accuracy across the families (PoC-8 cells + agent candidates) | `poc8_cells.json`, run ledgers |
| F1 | Edit anatomy: the edited neurons and layers, logit-lens readout ("Wait", `</think>`) | a1/a2 notes |
| F2 | Ablation: close-only / NoWait-only / both → accuracy and cut rate | a1/a2 notes (gen-checks) |
| F3 | Scatter of proxy score vs dev accuracy, colored by family (shows Goodhart) | as above |
| F4 | Token-length distributions (base, Astra, a1, bf4096, combo) on MATH-148 | raw generations |
| F5 | Cost: GPU-hours + agent tokens vs LoRA data generation + training | to be computed after E4 |

## To do before submission

- [ ] **E4** completes (LoRA-SFT); E4.3 full evaluation if it is competitive.
- [ ] **Cost accounting:** GPU-hours per agent run (scorer + gen-checks) and agent token usage, vs E4 data generation
      and training.
- [ ] **Larger n on GSM8K** (test ≥ 200) for base, Astra, bf4096, a1 and combo, to tighten the GSM8K comparison
      (currently n = 32).
- [ ] **More replicates** of the winning protocol (a3, a4) and of the atlas arm, to firm up H4 (currently 3 pairs).
- [ ] **Second defect** without a lexical marker (for example, tool-call format, premature stopping, JSON validity) to
      test whether the output-lever story generalizes.
- [ ] **Second model** (another small hybrid or dense reasoning model), if compute allows.
- [ ] A **stronger MATH grader** (re-grade the saved raw generations with math_verify).
- [ ] **Release:** edit parameters (`done/<id>.pt`), raw generations (HF Datasets/Zenodo), agent briefs and transcripts
      (after review).
- [ ] Decide the venue: arXiv preprint first; candidates are an interpretability/agents workshop or a main-track
      submission if the second defect works.
