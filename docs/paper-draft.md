# Agents as Weight-Space Programmers: Repairing Reasoning Loops in a Small Language Model with Four Rank-1 Edits

*Pre-paper draft, 2026-11-08. Authors and affiliation: [TBD], Universidade do Sul de Santa Catarina (Unisul).*

> **Status for readers of this draft.**
> - **Done:** the core experiments. That covers the reference-edit analysis, 6 agent runs under two protocols, the
>   decoding baselines, the naive-search control, the LoRA fine-tuning baseline, out-of-distribution transfer, side
>   effects and cost.
> - **Not yet:** the larger-n confirmation on GSM8K test, more agent replicates, and a second defect and model
>   (Section 10).
> - **Where the numbers come from:** every number below traces to a summary file in `data/` and a dated entry in the
>   lab notebook (`docs/CADERNO.md`).
> - **Figures:** the notebook `notebooks/results.ipynb` renders every figure referenced here, with plain-language
>   explanations.
> - **Citations:** marked [verify] still need checking against the original papers.

## Abstract

Recent work showed that a frontier model can improve a small language model by editing its weights directly, without
gradients ("weight-space programming"). We study this in a controlled, low-budget setting:
- **Model and defect:** Qwen3.5-0.8B, whose reasoning frequently collapses into repetition loops. 85% of its GSM8K
  generations exhaust a 12k-token budget without an answer.
- **Agents:** Claude Opus 5.5, limited to four rank-1 output-side edits and a budgeted scorer.
- **Hardware:** a single consumer GPU.

Two independent agents converge on the same interpretable mechanism. They suppress the "Wait" token through three MLP
neurons and strengthen the `</think>` write in the last layer. This lifts GSM8K accuracy from 15% to 65–67% on a dev
set and 13% to 56% on a locked test, and it transfers to MATH-500 (8% → 47%). The edit outperforms:
- the published reference edit (21% on MATH-500);
- tuned logit biasing;
- LoRA fine-tuning on self-distilled, loop-excised traces (29% on dev, at 13× the GPU time).

It also composes with budget forcing (53% on MATH-500). The edited model generates 60% fewer tokens per problem, and
it shows no measurable damage on unrelated tasks.

Controls isolate what matters:
- **The agent's reasoning:** a naive search with the same tools, budget and feedback reaches only 27%.
- **Generation in the loop:** every cheap proxy we tried (teacher-forced log-probabilities, branch-point probes,
  n-gram loop detectors) was exploited by at least one optimizer, and only real generation checks prevented this.
- **Less context, not more:** agents given a richer activation atlas found mechanistically interesting but behaviorally
  weaker edits, in 3 of 3 paired replicates.

## 1. Introduction

Small open models are increasingly deployed with "thinking" modes, and they inherit a characteristic failure. The
reasoning trace degenerates into loops ("Wait, … Yes, … Wait, …") and never reaches an answer. The standard remedies
both have costs:
- **Decoding constraints** (repetition penalties, budget forcing) are blunt: they cut productive reasoning too.
- **Fine-tuning** needs data, compute and care. For a looping model, even generating the training data is expensive,
  because most samples run to the token limit.

Takehi et al. (2026) proposed an alternative. A frontier agent (GPT-6 "Astra") inspected the weights of Qwen3.5-0.8B
and edited 0.016% of them, lifting MATH-500 from 7.9% to 48.1% by inverting a direction that sustains repetition. The
result is striking, but it leaves three questions open:
1. Is it reproducible, and how specific is the edit?
2. Is it more than a decoding trick in disguise, and how does it compare with fine-tuning?
3. What in the agent's process actually matters?

We answer them on a single 6 GB consumer GPU, with Claude Opus 5.5 agents supervised by an automated research
harness. We pre-registered predictions and kept a locked test split.

**Contributions.**
1. **Replication and dissection of the reference edit.** It is exactly rank-1 per layer, with one direction shared
   across two layers. It reproduces behaviorally on our stack (Section 4).
2. **A stronger, interpretable and replicated edit.**
   - Two independent agents find the same two-part mechanism: a "Wait" suppressor plus a `</think>` lever.
   - Ablations show the two parts are synergistic: the loop needs "a way back in and no way out".
   - It transfers to harder, unseen problems (Section 5).
3. **Baselines that rule out "just decoding" and "just fine-tuning".**
   - The edit beats tuned logit biasing everywhere and beats budget forcing on long problems; the two compose.
   - LoRA fine-tuning on self-distilled data lags by 37.5 pp at 13× the GPU time (Section 6).
4. **Process findings for agentic model editing.**
   - Cheap proxy objectives are Goodharted by strong optimizers, and generation in the loop is necessary.
   - The agent's reasoning, not the toolkit, carries the result.
   - More interpretability context (an activation atlas) can hurt (Section 7).
5. **Cost accounting.** We report GPU-hours, agent tokens and serving cost for every arm, and compare token usage with
   the original agent (Section 8).

## 2. Background and related work

**Weight-space programming.** Takehi et al. (2026) gave a frontier agent a CPU kernel over the weights of a small model
and a budget of forward passes, and asked it to improve benchmark performance. In their run with the largest budget,
the agent found a rank-1 inversion in two MLP layers. We use their published edit as our reference and their public
logs for the cost comparison.

**Model editing.** Rank-one and low-rank edits have been used to insert or modify factual associations: ROME (Meng et
al., 2022) and MEMIT (Meng et al., 2023). Directional ablation removes behaviors such as refusal (Arditi et al., 2024).
Our edit space, rank-1 updates to output projections along a chosen residual direction, belongs to this family. The
difference is that the edit is chosen by an agent that reasons about behavior, not by a closed-form objective.

**Activation steering.** Adding vectors to the residual stream at inference time steers behavior: ActAdd (Turner et
al., 2023) and representation engineering (Zou et al., 2023). Steering acts at run time, whereas our edits are
permanent weight changes, but the two share the residual-direction view.

**Overthinking and loops in reasoning models.** Long chains of thought often contain redundant self-verification
(Chen et al., 2024 [verify]). Interventions include suppressing reflection tokens such as "Wait" ("NoWait", Wang et
al., 2025 [verify]) and budget forcing (Muennighoff et al., 2025), which caps or extends thinking at decode time. We
use both as baselines.

**Agents doing interpretability.** Automated interpretability agents run experiments on models to explain components,
for example MAIA (Shaham et al., 2024). Our agents go one step further: they intervene on the weights to change
behavior, with a budget and a fixed evaluation protocol.

**Goodhart's law in optimization.** Optimizing a proxy diverges from the true objective under strong optimization
pressure, as shown for reward models (Gao et al., 2023) and formalized as reward hacking (Skalse et al., 2022). We
observe the same phenomenon with interpretability-derived proxies.

## 3. Setup

**Model and defect.**
- **Model:** Qwen3.5-0.8B at the same revision as the reference work. It is a hybrid with 24 layers: full attention
  in layers 3, 7, 11, 15, 19 and 23, Gated DeltaNet linear attention elsewhere, d = 1024, vocabulary 248k.
- **Defect:** with thinking enabled and a 12,288-token cap, 85% of its GSM8K generations end without closing
  `</think>`, and it solves 14.6% of our dev problems.
- **Prompting does not help.** Anti-repetition system prompts perform the same as a placebo prompt, both in
  teacher-forced measurements and in generation. In fact both slightly *raise* the loop rate (0.31 / 0.30 vs 0.23).

**Edit space.** An agent may apply at most four edits. Each targets one output projection: MLP `down_proj`, linear
attention `out_proj` or attention `o_proj`. Column $w_j$ is what unit $j$ writes into the residual stream. An edit
chooses a unit direction $u$, a column set $S$ and a strength $\alpha$:
$$
w_j \leftarrow w_j - \alpha\,(u^\top w_j)\,u \quad (j\in S),\qquad |\alpha|\le 10.
$$
- Each edit is a rank-1 update that scales the selected units' write along $u$ by $(1-\alpha)$.
- $\alpha = 1$ removes the write and $\alpha > 1$ inverts it. The reference edit has $\alpha = 6$. A negative
  $\alpha$ amplifies it.
- Embedding, unembedding and the final norm are off-limits, and so is scaling a whole tensor.

**Scorer and protocols.** Agents never run the model directly. They submit edits to a budgeted scorer daemon.
- **Protocol v1:** teacher-forced *branch-point probes*, 60 calls. Each probe compares the log-probability of a
  repetitive vs a productive continuation after an identical prefix, on reasoning traces where the base model loops.
- **Protocol v2:** 40 probe calls plus **6 generation checks**. A generation check runs the edited model on 16 held-out
  GSM8K training problems with a 6k cap and returns accuracy and the share of generations cut at the cap. We
  introduced it after observing Goodhart under v1 (Section 7).

**Agents.**
- Each run is one Claude Opus 5.5 subagent with a written brief, its own directory and budget, and roughly 2 hours of
  wall-clock time.
- Agents may not read the reference artifacts, the project documentation, other runs, the sealed holdout or the
  scorer's source. Compliance is on the honor system; the supervisor audits every tool call afterwards and found no
  violations.
- *Atlas* arms additionally receive an activation atlas: residual-stream statistics on looping vs productive
  continuations under several system prompts.

**Supervisor protocol.**
1. Each agent submits a ranked list of candidates.
2. The supervisor confirms the top candidate on a separate dev set: dev48, GSM8K train[0:48], 12k cap.
3. A single pre-registered candidate per arm is evaluated once on the locked test: t12k, GSM8K test[0:32], opened only
   through an explicit unlock that is logged.
4. Out-of-distribution transfer is measured on MATH-500[0:148]. The agents' search used GSM8K only.

**Generation and statistics.**
- **Engine:** llama.cpp server on f16 GGUFs.
- **Sampling:** the model's recommended thinking settings (temperature 0.6, top-p 0.95, top-k 20), with a presence
  penalty of 1.5 applied over the whole output. Every problem uses the same seed in every arm.
- **Comparisons:** paired by problem. We report the mean difference ± one standard error and the counts of rescues
  (wrong → right) and losses (right → wrong).
- **Uncertainty:** with n = 32–48, one SE is about 7–10 pp; on MATH-148 it is about 3–4 pp.

## 4. Dissecting the reference edit

**The edit.**
- Only `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj` differ from the base model.
- Each difference is rank-1 to 0.99999 of its energy: the update above with $\alpha = 6$ on 64 columns, with
  reconstruction error below 0.3%.
- The selected columns' component along $u$ becomes −5× the original: the edit inverts and amplifies that component;
  it does not ablate it.
- The two layers' directions have |cos| = 0.85, essentially one "repetition direction", but are written by almost
  disjoint neurons (3 of 64 shared).
- Rebuilding the edit from the base model and the recovered parameters reproduces the published checkpoint to bf16
  rounding.

**It reproduces behaviorally** (Table 1).
- dev48: 0.146 → 0.458, a paired gain of +31 ± 7 pp (16 rescues, 1 loss).
- MATH-148: 0.081 → 0.209.
- Our absolute MATH numbers are lower than the paper's (7.9% → 48.1%) because our cap is 12k rather than 32k and the
  sampler differs. All arms share our sampler.

**Proxies and behavior come apart.**
- A mean-difference direction beat the reference edit on teacher-forced Δlog-prob over loop tokens, yet gained only
  +6 pp in generation against +28 pp for the reference edit.
- Branch-point probes rank edits correctly *within* the reference edit's family: Spearman 0.89 with accuracy over 10
  variants. They fail across families (Section 7).

This dissociation motivated the v2 scorer.

## 5. Agents find a stronger edit

**Protocol v1 (probes only).** Agent a0, without the atlas, found a token-level anti-copy edit that inverts the "Yes",
"?" and "Wait" unembedding directions in two attention output projections. It matches the reference edit on the locked
test: 0.438 vs 0.406, with 10 rescues and 0 losses against the base.

**Protocol v2 (generation in the loop).** Two independent no-atlas agents converged on the same two-part mechanism:

| run | mechanism (≤ 4 rank-1 edits) | gen-check | dev48 |
|---|---|---|---|
| a1 (`062959_102`) | amplify the "Wait"-unembedding component of 3 MLP neurons (L17n1915, L18n881, L22n573; α = −6) + amplify the `</think>` write of the L23 MLP (α = −0.8) | 8/16 | **0.667** (+52 ± 8; 26/1) |
| a2 (`015739_897`) | remove the "Wait" direction from L16n998, L17n1915, L22n573 (α = 10) + a `</think>` boost on L22 linear attention (α = −4) | 11/16 | 0.646 (+50 ± 7; 24/0) |

- **Convergence.** Both agents found L17n1915 and L22n573 independently, using a weights-only logit lens over MLP
  columns.
- **Mechanism: a way back in, and no way out.** Ablations in generation (n = 16) show the two parts are synergistic:

  | edit | accuracy | cut at cap |
  |---|---|---|
  | `</think>` boost alone | 0.19 | 0.81 |
  | "Wait" suppression alone | 0.44–0.50 | 0.50 |
  | both | 0.69 | 0.06–0.13 |

  With re-checking suppressed, closing the thought becomes the model's natural alternative. The effect is superadditive
  on probes as well.
- **Transfer.** a1 scores 0.562 on the locked test (+44 ± 10 vs base, +16 ± 8 vs the reference edit) and 0.466 on
  MATH-148 (+25.7 ± 4.3 vs the reference edit; 44 rescues, 6 losses).
- **Thinking becomes useful again.** At 0.667, a1 exceeds the base model's GSM8K score *without* thinking (about 0.48).

## 6. Baselines and controls

**Table 1. Accuracy, thinking on, 12,288-token cap.**

| arm | GSM8K dev48 | GSM8K locked test (32) | MATH-500[0:148] |
|---|---|---|---|
| base | 0.146 | 0.125 | 0.081 |
| reference edit (Astra) | 0.458 | 0.406 | 0.209 |
| logit bias (best of 3, tuned on dev48) | 0.521 | 0.469 | 0.271* |
| budget forcing at 4096 (best of 2, tuned on dev48) | 0.479 | **0.625** | 0.372 |
| naive search, same tools and budget (E3) | 0.271 | — | — |
| LoRA-SFT on self-distilled traces (E4) | 0.292 | — | — |
| **agent edit a1** | **0.667** | 0.562 | **0.466** |
| **a1 + budget forcing** | **0.688** | **0.625** | **0.527** |

\* MATH-500[0:48] only.

**Decoding baselines (E1).** We tuned three logit-bias settings (penalizing "Wait" variants and boosting `</think>`)
and two budget-forcing caps on dev48. That is 48 problems against the agents' 16, which favors the baselines.
- **Logit bias:** a1 beats the best setting everywhere: +15 ± 7 pp on dev48, +9 ± 8 on the locked test, +12.5 ± 5.7 on
  MATH-48. The edit is context-conditional: it acts through neurons that fire in loop contexts, not through a global
  logit shift.
- **Budget forcing:** a strong baseline on GSM8K, and the best on the locked test (a1 −6 ± 6), because the answer is
  often computed before the loop begins. On the longer MATH problems the edit wins clearly: +9.5 ± 3.8 on MATH-148,
  with 23 rescues and 9 losses.
- **The two compose.** a1 + budget forcing beats a1 by +6.1 ± 3.2 on MATH-148 (16/7) and budget forcing alone by
  +15.5 ± 3.6 (27/4).
- **MATH-148 ordering:** combo > a1 > budget forcing > reference edit > base, with every adjacent gap ≥ 1.9 SE.

**Naive search (E3).** A script sampled edits from the toolkit the winning agents used:
- logit-lens directions of the 300 most frequent output tokens;
- columns chosen by alignment;
- layers 12–23;
- α uniform in [−10, 10].

It had the same budget, 40 probes plus 6 generation checks. Its best generation check was 3/16 and it reached 0.271 on
dev48, 38–40 pp below a1 and a2. The tools and the generation signal are not sufficient; the agent's choices are what
matter.

**Fine-tuning (E4).**
- **Data:** we self-distilled training data from GSM8K train[300:900], disjoint from all evaluation and search sets.
  Of 600 samples (6k cap), 57 closed `</think>` without looping and were correct ("natural"). For 279 hard loops, we
  cut the trace at the loop onset, injected `</think>` and let the base model answer; 118 of those answers were
  correct ("excised").
- **Training:** LoRA (r = 16, all linear layers, lr 1e-4, 3 epochs) on these 175 sequences, with the loss on the
  completion only.
- **Selection:** we picked the epoch on the agents' 16 generation-check problems (epoch 2, 7/16).
- **Result:** 0.292 on dev48. That is +14.6 ± 5.9 over the base, but **−37.5 ± 7.1 vs a1, with 0 rescues and 18
  losses**, and below every decoding baseline.
- **Why:** the LoRA barely changes the looping behavior; the cut rate goes from 0.85 to 0.71, against 0.17 for a1.
  Excised traces teach the model what to write after a cut, not when to cut.
- **Scope:** this is one reasonable SFT configuration. More samples per problem, more data or RL might do better, at a
  higher cost (Section 8).

## 7. Process findings

**Cheap proxies are Goodharted.** Every inexpensive objective we tried was eventually exploited:
1. **Teacher-forced Δlog-prob:** the mean-difference direction (Section 4).
2. **Branch-point probes:** agent r1 (v1, with atlas) found an edit with a branch score of −2.03, 20× the reference
   edit's, that *hurts* accuracy (0.042 vs base 0.146). The edit blocks *literal* copying, and the model routes around
   it with non-literal loops (alternating values, growing sequences). A sealed holdout in the same probe format did not
   catch this.
3. **8-gram loop detector:** an edit with 0% literal loops at the cap still had 77% of its generations cut.
4. **Dose:** r1's directions at Astra-scale probe values gave −2 / +6 pp, while the reference edit at the same probe
   value gave +31 pp.

Generation checks caught the problem immediately. Under v2, both agents' best-probe candidates scored 0/16 in
generation; for example, negating the "Wait" neurons produced "wait wait wait…". The agents dropped those candidates.
We conclude that an agent's objective must come from generation, even a cheap one (16 problems, 6k cap). Probes are
useful as constraints, not as targets.

**More interpretability context hurt.**

| pair | with atlas | without atlas |
|---|---|---|
| v1 | r1: 0.042 | a0: 0.375 |
| v2 | r2: 0.354 | a1: 0.667 |
| v2 | r3: 0.396 | a2: 0.646 |

- **With the atlas,** agents pursued internal-state mechanisms, such as a re-checking state in the DeltaNet layers
  L10–L15 and "loop heads". These reduce literal copying and look clean on probes, but they do not close the
  reasoning.
- **Without it,** agents read the weights through the vocabulary and found output levers ("Wait", `</think>`) that
  control the behavior directly.
- **Hypothesis:** richer interpretability context biases agents toward mechanistically interesting targets rather than
  behaviorally causal ones. Three pairs make this suggestive, not conclusive.

## 8. Cost

| method | dev48 | GPU-hours (GTX 1660 SUPER) | agent tokens (context / output) | API-equivalent |
|---|---|---|---|---|
| agent edit (a1) | 0.667 | 1.31 | 5.39 M / 47.5 k | $5–7 |
| naive search | 0.271 | 1.63 | — | — |
| decoding baselines (tuning 5 settings) | 0.48–0.52 | 4.41 | — | — |
| LoRA-SFT (E4) | 0.292 | 17.1 | — | — |

- **GPU time.** The agent edit used 13× less GPU time than fine-tuning, or 8× counting the scorer's full residency on
  the GPU. Fine-tuning cost is dominated by data generation (10.2 of 17.1 GPU-hours), which the defect itself makes
  expensive.
- **Dollars.** At API list prices, the agent's tokens cost $5–7. The agent is also cheaper in dollars whenever a GPU-hour
  of this class costs more than about $0.32–0.44. The runs actually used a subscription, so the dollar figures are a
  reference only.
- **Serving cost.** Tokens generated per problem on dev48:

  | | tokens per problem | vs base |
  |---|---|---|
  | base | 11,714 | — |
  | LoRA | 9,915 | −15% |
  | budget forcing | 8,073 | −31% |
  | **a1** | **4,729** | **−60%** |

  On MATH-148, a1 generates 29% fewer tokens than the base. This saving recurs on every query.
- **Comparison with the original agent.** The published Astra run used 86.8 M input and 2.9 M output tokens over 20.3
  hours (from its public logs). a1 used 16× fewer input tokens, 60× fewer output tokens and about a tenth of the
  wall-clock time. The harnesses and hardware differ, so this is not a controlled comparison.

## 9. Specificity

Without thinking, no edit (reference, a0, a1) significantly changes GSM8K accuracy (n = 200) or tool-calling accuracy
(n = 300): all differences are within about 2 SE. The effect exists only with thinking on, consistent with the
mechanism. With thinking on, the reference edit slightly improves "irrelevance" tool-calling tasks, where the correct
action is to call no tool (16/16 vs 11/16). A plausible reading is that fewer reasoning tokens leave less room for
hallucinated calls.

## 10. Limitations and next steps

- **Small n.** Dev and test numbers carry SEs of 7–10 pp; MATH-148 is the strongest evidence. *Next:* GSM8K test with
  n ≥ 200 for base, reference, budget forcing, a1 and the combo.
- **One model, one defect.** "Wait" is a lexical marker, which may make loops unusually easy to target. *Next:* a
  defect without a lexical marker (tool-call format errors, premature stopping, invalid JSON) and a second small
  reasoning model.
- **Replicates.** The winning protocol has 2 replicates and the atlas comparison has 3 pairs. *Next:* two more of each.
- **One fine-tuning recipe.** A stronger SFT or RL baseline could narrow the gap. Our cost comparison shows each such
  step costs more GPU time.
- **Protocol advantage.** Agents saw 16 generation-check problems that the reference agent did not have. E3 controls
  for the signal, and E4 shows that seeing far more data does not help a standard method.
- **Sampler.** llama.cpp vs vLLM, and 12k vs 32k caps, shift absolute numbers relative to the reference work.
  Comparisons between arms share the sampler.
- **Honor system.** Agent restrictions were audited, not enforced.
- **Grading.** MATH answers are extracted from `\boxed{}` and compared numerically. *Next:* re-grade the saved
  generations with a symbolic checker.

## 11. Conclusion

An agent with a budgeted scorer, generation checks and four rank-1 edits repaired a specific behavioral defect in a
small reasoning model better than decoding constraints and a standard fine-tuning recipe. It used an order of magnitude
less compute than the fine-tuning recipe and produced a model that is cheaper to serve. The resulting edit is
interpretable, it replicates across independent agents, and it transfers to harder problems. The process lessons may
matter as much as the edit itself:
- optimize against behavior, not proxies;
- the agent's judgment carries the result;
- more interpretability context is not automatically better.

---

## Figures and tables

All figures are produced by `notebooks/results.ipynb`.

| # | content | notebook section |
|---|---|---|
| T1 | Main results (Table 1 above) | §2 |
| T2 | Paired rescues/losses vs each competitor | §3 |
| T3 | Agent runs: protocol, atlas, calls, mechanism, gen-check, dev48 | to build from `data/m2a/runs/*/final.json` |
| F1 | Response-length distribution, base vs a1 (the defect) | §1 |
| F2 | Accuracy vs cut rate across all 20 evaluated arms (r = −0.99) | §4 |
| F3 | Accuracy by MATH difficulty level | §5 |
| F4 | Proxy score vs real accuracy for each agent's pick (Goodhart) | §6 |
| F5 | Atlas vs no-atlas pairs | §7 |
| F6 | Agent search timelines (generation checks over time) | §8 |
| F7 | LoRA training loss and epoch selection | §9 |
| F8 | Cost: GPU-hours to produce, tokens per answer; token usage vs Astra | §10 |
| F9 | Side effects on unrelated tasks | §11 |
| — | Edit anatomy diagram (neurons, layers, logit-lens readouts) | to draw |

## Plan to submission

**Working title.** Title 1 above is the safe choice. Alternatives:
- *Agentic Weight Editing as a Fine-Tuning Substitute for Specific Behavioral Defects*. E4 supports it, but only for
  one SFT recipe.
- *Generation in the Loop: What Makes Agentic Weight Editing Work*, a process-centric framing.

**Remaining work, in order of value:**
- [ ] Larger-n locked test (GSM8K test ≥ 200) for the main arms. About 1 GPU-hour per arm.
- [ ] Two more replicates of the winning protocol and of the atlas arm.
- [ ] A second defect without a lexical marker. This is the step that would justify a main-track submission.
- [ ] A second model, if compute allows.
- [ ] Re-grade MATH with a symbolic checker.
- [ ] Table T3 and the edit-anatomy figure.
- [ ] Verify every citation; write related work in full.
- [ ] Release: edit parameters (`data/m2a/runs/*/done/*.pt`), raw generations, agent briefs and transcripts (after
      review).
- [x] Core experiments E1–E4, MATH-148 transfer, side effects, cost accounting, results notebook.

**Venue.**
- arXiv preprint first.
- Then an interpretability or LLM-agents workshop with the current results.
- A main-track venue if the second defect works.

## References (to verify and complete)

- Takehi, R., Teoh, J., Zhang, M., Khattab, O. (2026). *Sparks of Weight-Space Programming.* MIT CSAIL blog.
  <https://rikiyat.github.io/blog/weight-space-programming/>. Logs: <https://github.com/RikiyaT/lm_understand-logs>.
- Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT (ROME). NeurIPS.
- Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., Bau, D. (2023). Mass-Editing Memory in a Transformer (MEMIT). ICLR.
- Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. NeurIPS.
- Turner, A. M., et al. (2023). Steering Language Models With Activation Engineering (ActAdd). arXiv.
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv.
- Muennighoff, N., et al. (2025). s1: Simple Test-Time Scaling. arXiv.
- Chen, X., et al. (2024). Do NOT Think That Much for 2+3=? On the Overthinking of o1-like LLMs. arXiv. [verify]
- Wang, C., et al. (2025). Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency. arXiv. [verify]
- Shaham, T. R., et al. (2024). A Multimodal Automated Interpretability Agent (MAIA). ICML.
- Gao, L., Schulman, J., Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. ICML.
- Skalse, J., et al. (2022). Defining and Characterizing Reward Hacking. NeurIPS.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems (GSM8K). arXiv.
- Hendrycks, D., et al. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS.
- Lightman, H., et al. (2023). Let's Verify Step by Step (MATH-500 subset). arXiv.
- Qwen Team (2026). Qwen3.5 model card. [verify]
