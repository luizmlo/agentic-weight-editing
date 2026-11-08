# Research notes

Living summary of what we know, with the evidence behind each claim. The primary, timestamped record is
[CADERNO.md](CADERNO.md) (pt-BR); every number here can be traced to an entry there and to a summary file in `data/`.
The experiment registry is in [experiments.md](experiments.md).

Last updated: 2026-11-08.

---

## 0. Setup and conventions

**Model.** Qwen3.5-0.8B @ `2fc0636`: 24 layers, d_model = 1024, vocabulary 248k. It is a hybrid: full attention in layers
3, 7, 11, 15, 19 and 23, and Gated DeltaNet (linear attention) in the rest. fp16 everywhere (the GPU has no bf16);
fp16 vs fp32 logits agree on top-1 for 100% of tokens, with mean KL 9.5e-6.

**Defect.** With thinking enabled, the model enters reasoning loops and fails to close `</think>`. GSM8K dev48, 12k cap:
base accuracy 0.146, with 85% of generations cut at the cap.

**Generation.** llama-server (the binary shipped with Ollama) runs on f16 GGUFs with 16 parallel slots, at ~150 tok/s
aggregate. Sampling uses the Qwen thinking defaults (THINK35), with presence penalty applied over the whole output
(`repeat_last_n = max_new + 512`). Every problem gets the same seed in every arm.

**Splits.**

| name | content | cap | use |
|---|---|---|---|
| gen-check | GSM8K train[200:216] | 6144 | the only generation signal agents see (protocol v2) |
| dev48 | GSM8K train[0:48] | 12288 | supervisor confirmation, baseline tuning |
| t12k | GSM8K test[0:32] | 12288 | **locked** test: opened only with `WSP_UNLOCK_TEST=1`, every use logged |
| math48 / math100b | MATH-500[0:48] / [48:148] | 12288 | out-of-distribution transfer (search used GSM8K only) |

**Statistics.** All comparisons are paired by problem. We report the mean difference ± its standard error, plus
rescues/losses (problems that flip from wrong to right / right to wrong). With n = 32–48, one SE is ~7–10 pp; with
MATH-148 it is ~3–4 pp. Batching in llama.cpp is not bit-deterministic, so pairing is approximate.

**Edit space.** At most 4 edits, each on one output projection (`mlp.down_proj`, `linear_attn.out_proj` or
`self_attn.o_proj`). Column $w_j$ of such a matrix is what neuron (or head channel) $j$ writes into the residual
stream. An edit chooses a unit direction $u$ in the residual stream, a set $S$ of columns and a strength $\alpha$:

$$
w_j \leftarrow w_j - \alpha \,(u^\top w_j)\, u \quad (j \in S), \qquad |\alpha| \le 10 .
$$

The selected neurons' write along $u$ is scaled by $(1-\alpha)$. $\alpha = 1$ removes it, $\alpha > 1$ flips it, and
$\alpha < 0$ amplifies it. Each edit is a rank-1 update. In the code, $S$ is a column mask (`cols`; `None` means all
columns).

**Agents.** Claude Opus 5.5 subagents, launched and audited by a Claude Code supervisor. Each run has its own
directory, budget and brief (`data/m2a/BRIEF*.md`). Agents may not read the Astra artifacts, the docs, other runs, the
sealed holdout or the scorer source. Compliance is on the honor system, and the supervisor audits the tool calls in
each transcript afterwards. No violations were found in any run.

---

## F1. The Astra edit is exactly rank-1, and one direction serves two layers

Only `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj` differ from the base. Each difference is rank-1 to
0.99999 energy: the edit above with α = 6 on 64 columns, with reconstruction error < 0.3% and relative weight
change of 10.6% / 8.1%. With α = 6, the component of each selected column along `u` becomes −5× the original: the edit
inverts and amplifies that component; it does not ablate it. The two layers' directions have |cos| = 0.849 (essentially
one "repetition direction"), written by different neurons (3 shared indices out of 64).
*Evidence:* PoC-1, `data/poc1_diff.json`. Rebuilding the edit from base + recovered parameters reproduces the
published checkpoint to bf16 rounding (`astra_check`).

## F2. The Astra result reproduces locally

| | base | Astra | paired gain |
|---|---|---|---|
| t12k | 0.125 | 0.406 | +28 ± 10 pp (11/2) |
| dev48 | 0.146 | 0.458 | +31 ± 7 pp (16/1) |
| MATH-500 (148) | 0.081 | 0.209 | +12.8 ± 2.8 pp (19/0) |

As in the paper, correct answers remain long (median ~7k tokens). Absolute MATH numbers are lower than the paper's
(7.9% → 48.1%) because our cap is 12k, not 32k, and the sampler differs (llama.cpp, not vLLM).
*Evidence:* PoC-7b, `gen_t12k.json`, `gen_dev48.json`, `gen_math48.json`, `gen_math100b.json`.

## F3. Prompting does not fix the loop, and a placebo prompt behaves like an anti-repetition prompt

- **Teacher-forced (PoC-5):** no system prompt changes the probability of continuing a loop, because the repeated
  context dominates. *Any* system prompt shifts activations along the repetition direction, and a neutral placebo shifts
  them more. The instruction-specific component (anti-repetition − placebo) points the "right" way, with cos 0.12–0.21,
  but it is far too weak to matter.
- **Behavioral (PoC-6, 4k cap):** anti-repetition and placebo prompts both *raise* the rate of ending in a loop (0.31 /
  0.30 vs 0.23), and neither changes accuracy.
- **Agent r3's atlas:** the system prompt moves the internal state roughly orthogonally to the loop direction
  (cos 0.999 between the rep − prod contrasts under 4 prompts).
- *Method lesson:* always include a placebo prompt.

## F4. Cheap proxies dissociate from behavior, and strong optimizers find the gap

This is the central methodological finding.

1. **Mean Δlog-prob on loop tokens is not a valid proxy.** A mean-difference direction (`meandiff`) beat Astra on that
   metric but gained only +6 pp in generation, against +28 pp for Astra (PoC-7).
2. **Branch-point probes are valid only within one family of edits.** A probe that compares paired continuations
   (repetitive vs productive) after an identical prefix ranks 10 Astra-like edits correctly: Spearman 0.89 with
   accuracy and 0.96 with the cut rate (PoC-8b). The onset metric scores −0.19.
3. **Outside that family, the branch probe is Goodharted.** Agent r1 (protocol v1, probes only) found an edit with
   branch_rep −2.03, 20× Astra's value, that **hurts** accuracy: 0.042 vs base 0.146. The edit blocks *literal*
   copying, and the model routes around it with non-literal loops (alternating values, growing sequences). The sealed
   holdout did not catch this, because it uses the same probe format.
4. **Dose does not rescue a wrong direction.** r1's directions at a small α, with Astra-scale branch_rep (−0.10 / −0.13),
   give −2 / +6 pp. Astra at the same branch_rep gives +31 pp. The probe cannot tell causal directions apart across
   families.
5. **The 8-gram loop detector is Goodharted too.** One edit had 0% literal loops at the cap but 77% of its generations
   cut. Since then, the primary generation metrics are accuracy and the cut rate.
6. **Generation checks catch Goodhart.** In protocol v2, both agents' best-probe candidates scored 0/16 in generation:
   a1's negated "Wait" neurons produced "wait wait wait…", and r2's heads alone did nothing. The agents dropped them.

*Implication:* the objective an agent optimizes must come from generation, even a cheap one (16 problems, 6k cap).
Probes are useful as constraints and safety signals, not as targets.

## F5. With probes only (v1), an agent already matches Astra on the locked test

Agent a0 (no atlas, 48 of 60 probe calls, ~2 h on the 1660S) found a token-level anti-copy edit: it inverts the
unembedding directions of "Yes" and "?" in L15 `o_proj` and of "Wait" in L23.

| | dev48 | t12k (locked) | math48 |
|---|---|---|---|
| Astra | 0.458 | 0.406 | 0.208 |
| a0 | 0.375 (+23 ± 7) | **0.438** (+31 ± 8, 10/0) | **0.271** (+23 ± 6, 11/0) |

Collateral checks found no detectable damage (tool calling without thinking +3.7 ± 1.6 pp; GSM8K without thinking −2 ± 4 pp).

## F6. Generation in the loop (v2) changes what agents find: a1 and a2 at about +50 pp

| arm | mechanism (≤ 4 rank-1 edits) | gen-check | dev48 |
|---|---|---|---|
| a1 #1 `074537_659` | amplify the "Wait"-unembedding component of 3 MLP neurons (L17n1915, L18n881, L22n573; α = −9) + amplify the `</think>` write of L23 MLP (α = −1.1) | 11/16 | 0.646 (+50 ± 8) |
| **a1 `062959_102`** (milder, used downstream) | same neurons, α = −6; `</think>` α = −0.8 | 8/16 | **0.667 (+52 ± 8, 26/1)** |
| a2 #1 `015739_897` | remove the "Wait" direction from L16n998(+n3422), L17n1915, L22n573 (α = 10) + a `</think>` boost on L22 `linear_attn.out_proj` (α = −4) | 11/16 | 0.646 (+50 ± 7, 24/0) |

- **Convergence.** Two independent agents found the same neurons (L17n1915 and L22n573 are shared) and the same
  two-part anatomy. They found them with a weights-only logit lens over MLP columns, reading direction
  `E[t] ⊙ (1 + g_final)`.
- **The loop needs a way back in and no way out.** Ablations in generation (n = 16): `</think>` boost alone 0.19, cut
  0.81; "Wait" suppression alone 0.44–0.50, cut 0.50; both 0.69, cut 0.06–0.125. The two parts are synergistic: with
  "Wait" suppressed, closing becomes the model's alternative. The effect is superadditive on probes too (rep −0.162
  alone, −0.286 combined).
- **Thinking helps again.** a1 at 0.667 exceeds the base's no-thinking GSM8K score (~0.48).
- **Locked test and transfer (a1 `062959_102`):** t12k 0.562 (+44 ± 10, 15/1; vs Astra +16 ± 8). math48 0.396
  (+35 ± 7, 17/0; vs Astra +19 ± 7).
- **Collateral checks (a1):** tool calling without thinking 0.927 (+3 ± 1.6), GSM8K without thinking 0.485 (equal to
  base, 25/25), tool calling with thinking 0.94 (+1). No detectable damage.
- **Caveat.** Agents had 16 generation-check problems that Astra did not have. Part of the advantage is the protocol,
  not the agent. F8 and F9 separate the two.

## F7. H4 is refuted: the activation atlas hurts (3 of 3 pairs)

Original hypothesis: mapping activations under prompt variants (the "atlas") reveals the directions to edit.

| pair | with atlas | without atlas |
|---|---|---|
| v1 | r1: 0.042 (#1), 0.229 (#3) | a0: 0.375 |
| v2 | r2: 0.354 | a1: 0.667 |
| v2 | r3: 0.396 | a2: 0.646 |

The pattern is stable. With the atlas, agents find internal-state mechanisms: a re-checking state in the DeltaNet layers
L10–L15 and "loop heads". These reduce literal copying and look clean on probes, but they do not close the reasoning.
Without the atlas, agents read the weights through the vocabulary and find **output levers** ("Wait", `</think>`) that
control the behavior directly.
*Hypothesis for the paper:* richer interpretability context biases the agent toward mechanistically interesting
targets rather than behaviorally causal ones. Testing this needs more than one defect.

## F8. The edit is not just a decoding trick (E1 and the combo)

Baselines were tuned on dev48 (48 problems, more than the agent's 16), which favors the baselines.

| a1 vs | dev48 | t12k | MATH |
|---|---|---|---|
| best logit bias (`lbx_w10_t1`: −10 on 4 "wait" tokens, +1 on `</think>`) | +15 ± 7 | +9 ± 8 | +12.5 ± 5.7 (math48) |
| best budget forcing (`bf4096`) | +19 ± 8 | −6 ± 6 | **+9.5 ± 3.8 (MATH-148, 23/9)** |

- **Logit bias is worse everywhere.** The edit is context-conditional: it acts through neurons that fire in loop
  contexts, not through a global logit shift.
- **Budget forcing is a strong baseline on GSM8K.** It has the best t12k score (0.625), because the answer is often
  computed before the loop starts. On the longer MATH problems the edit wins clearly.
- **The two methods are complementary.** On MATH-148, a1 + bf4096 scores 0.527 vs a1's 0.466 (+6.1 ± 3.2, 16/7) and vs
  bf4096's 0.372 (+15.5 ± 3.6, 27/4). The edit breaks loops from the inside, and forcing collects the residual long
  runs.
- MATH-148 ordering: combo 0.527 > a1 0.466 > bf4096 0.372 > Astra 0.209 > base 0.081. Every adjacent gap is ≥ 1.9 SE.

## F9. The agent's reasoning matters: naive search with the same tools fails (E3)

`m2a_naive.py` sampled edits from the toolkit the winning agents used: logit-lens directions of the 300 most frequent
tokens in base outputs, columns chosen by |cos|, layers 12–23, α ~ U[−10, 10], 1–2 edits. It had the same budget: 40
probes, with the 6 best in-limit candidates sent to generation.

| | best branch_rep | gen-check | dev48 |
|---|---|---|---|
| naive | −0.007 | 1–3/16 | 0.271 (+12.5 ± 6 vs base) |
| a1 / a2 | −0.29 / −0.33 | 8–11/16 | 0.667 / 0.646 |

The gap of 38–40 pp shows that the generation signal and the tools alone are not enough. The agent's choices are what
make the difference: it targets "Wait" and `</think>` and reads the synergy from its failures.

## F10. Specificity

Without thinking, no edit (Astra, a0, a1) significantly changes GSM8K or tool calling (all within ~2 SE). The effect
exists only with thinking on, which matches the mechanism. Astra (but not a1) also improves "irrelevance" tool-calling
tasks with thinking (16/16 vs 11/16): fewer reasoning tokens means fewer hallucinated calls.

## F11. LoRA fine-tuning on self-distilled data falls far short of the edit (E4)

The baseline is LoRA-SFT (r16 on all linear layers, lr 1e-4, 3 epochs) on 175 correct self-distilled traces from GSM8K
train[300:900]:
- 57 "natural" traces, where the model closed `</think>` itself without a loop;
- 118 "excised" traces, cut at the loop onset with `</think>` injected, after which the base model answers.

The epoch was picked on the agents' 16 gen-check problems (epoch 2, 7/16).

| | dev48 | cut | vs ft (paired) | GPU-h to produce |
|---|---|---|---|---|
| base | 0.146 | 0.85 | ft +14.6 ± 5.9 (8/1) | — |
| **LoRA-SFT (ep2)** | **0.292** | 0.71 | — | ≈ 17 |
| Astra | 0.458 | 0.48 | ft −16.7 ± 6.2 | — |
| bf4096 | 0.479 | 0.46 | ft −18.8 ± 7.1 | 0 (tuning: 2 dev48 runs) |
| a1 | 0.667 | 0.17 | ft **−37.5 ± 7.1 (0/18)** | ≈ 2 |

- **The LoRA barely changes the looping.** The cut rate only drops from 0.85 to 0.71. Excised traces teach the model
  what to write after a cut, not when to cut.
- **Its cost is mostly data generation, which the defect itself makes expensive.** About 9 of the 17 GPU-hours went to
  sampling 600 traces, 90% of which ran into the 6k cap.
- **What the comparison covers.** It is a single, reasonable SFT configuration. More samples per problem, more data or
  RL (GRPO) could do better, at an even higher cost.
- **What the costs exclude.** They are GPU wall-clock hours on the 1660S. They leave out agent tokens (Claude Opus 5.5
  through Claude Code, with no metered API cost) and the dev48 confirmation runs of both arms.

## F12. Cost: an order of magnitude less GPU, and cheaper to serve ([cost.md](cost.md))

| | dev48 | GPU-h (1660S) | agent tokens (context / output) | API-equivalent |
|---|---|---|---|---|
| agent edit a1 | 0.667 | 1.31 (≤ 2.1 wall-clock) | 5.39 M / 47.5 k | $5–7 |
| LoRA-SFT (E4) | 0.292 | 17.1 | — | — |
| Astra p50c (paper, GPT-6) | — | — | 86.8 M / 2.9 M, 20.3 h | — |

- **Repair cost:** 13× less GPU time than LoRA-SFT, or 8× counting the scorer's full residency. In dollars, the agent is
  also cheaper whenever a 1660S-class GPU-hour costs more than about $0.32–0.44. It used 16× fewer input tokens and 60×
  fewer output tokens than the published Astra run.
- **Serving cost:** a1 cuts the tokens generated per problem by 60% on dev48 and by 29% on MATH-148. LoRA-SFT cuts them
  by 15% on dev48. This saving recurs on every query.
- **Overhead:** the supervisor session behind all arms, fine-tuning included, came to about $45–56 API-equivalent for
  the whole project.

---

## Open questions and threats to validity

- **Small n.** Each dev/test number has an SE of ~7–10 pp. MATH-148 is the strongest evidence. Replicating at n ≥ 200
  on GSM8K test is planned.
- **One model, one defect.** The output-lever story may be specific to loops ("Wait" is a lexical marker). A second
  defect with no lexical marker (for example, tool-call format errors or premature stopping) is the obvious next test,
  and so is a second model.
- **Protocol advantage.** Agents had 16 generation-check problems. E3 controls for the generation signal. E4 shows that a
  method that sees 600 problems' worth of the defect in data does much worse. The epoch pick used the same 16 problems.
- **Sampler differences** from the paper (llama.cpp vs vLLM, 12k vs 32k cap) shift absolute numbers. Comparisons
  between arms share the same sampler.
- **Honor-system restrictions** on agents. Audits found no violations, but this is a limitation to state.
- **Cost accounting** (F12) uses API list prices as a reference: the runs used a Claude Code subscription. GPU-hours
  are for a 6 GB Turing card and may not transfer to faster GPUs.

## Pending

- E4.3 (optional, since E4 was not competitive on dev48): MATH-148 for the LoRA. The locked test is not worth opening
  for an arm that already lost on dev.
