# Experiment registry

One row per experiment, in chronological order. "Prediction" is what was written in [CADERNO.md](CADERNO.md) **before**
the results came in, when one was registered. Outcomes are paired against base unless noted (± SE, rescues/losses).
Script and data paths use this repository's layout.

## Phase 0: reproduce and dissect the Astra edit (2026-11-03/04)

| id | question | prediction | outcome | script → data |
|---|---|---|---|---|
| F0 | Does the stack run on a GTX 1660S (sm_75, no bf16)? | — | fp16 safe (top-1 100%, KL 9.5e-6); HF generate too slow; llama-server -np 16 reaches ~150–180 tok/s | `poc2_hw.py` → `poc2_hw.json` |
| PoC-1 | What exactly did Astra change? | — | 2 tensors (L15/L19 down_proj), exactly rank-1, 64 cols, α = 6, \|cos(u15, u19)\| = 0.849 | `poc1_diff.py` → `poc1_diff.json`, `edit_dirs.pt` |
| PoC-3 | Dose-response on in-loop tokens (teacher-forced) | — | monotone and selective but ~11× weaker than Astra's own metric → the edit acts at loop *entry* | `poc3_tf.py` → `poc3_tf_w4096.json` |
| PoC-4 | Can a mean-difference direction recover u without an LLM? | — | \|cos\| = 0.52 (~20× chance) | `poc3_tf.py` |
| PoC-5 | Do system prompts move the loop direction? | — | any prompt shifts along u; the anti-rep − placebo component is weak (cos 0.12–0.21) | `poc3_tf.py` |
| PoC-6 | Do prompts fix loops in generation (4k cap)? | — | no; anti_rep ≈ placebo (both raise end-in-loop to 0.30); Astra removes hard loops (0.00) | `poc6_gen.py` → `poc6_gen_n64_t4096.json` |
| PoC-7 | Does the mean-difference edit reproduce Astra's behavior? (H-M) | meandiff ≈ astra > random | **no**: astra +28 ± 10, meandiff +6 ± 9, random −6 ± 4 (t12k) → proxy/behavior dissociation | `poc7_dirs.py`, `poc7_build.py`, `gen_eval.py` → `poc7_dirs.json`, `gen_t12k.json` |
| PoC-7c | Collateral effects (tool calling and GSM8K without thinking) | no damage off-target; gains only with thinking | confirmed; Astra +5 pp on tool calling with thinking (irrelevance tasks) | `tool_eval.py` → `tool_*.json`, `gen_nothink_n200.json` |
| PoC-8 | Direction × columns factorial (10 cells); is the branch probe a valid proxy? | Spearman(branch_rep, acc) > 0.7 | **confirmed**: 0.89 (acc), 0.96 (−loop); onset metric −0.19 | `poc8_cells.py` → `poc8_cells.json`, `gen_t12k.json` |

## Phase 1: agents with probes only, protocol v1 (2026-11-04)

Scorer: 12 paired branch probes (objective), branch_prod, close and neutral (constraints), and a sealed holdout.
Budget: 60 calls (rnd: 40).

| id | arm | prediction | outcome | data |
|---|---|---|---|---|
| sanity | known edits through the scorer | — | matches PoC-8 to 3 decimals; the holdout preserves order | `m2a/runs/sanity` |
| rnd | random search | branch_rep ≥ −0.02 | **confirmed** (−0.014) | `m2a/runs/rnd` |
| r1 | agent + atlas (M→A) | finds ≤ −0.05 and ≥ +15 pp on dev48 | branch −2.03 but dev48 **0.042 (−10 ± 5)** → **Goodhart**; #3 0.229 (+8) | `m2a/runs/r1`, `gen_dev48.json` |
| a0 | agent, no atlas (control for H4) | r1 > a0 (H4) | dev48 0.375 (+23 ± 7); **H4 refuted**; holdout at 40% of the search value (template overfit) | `m2a/runs/a0` |
| dose | r1's directions at Astra-scale branch_rep | better at a small dose → the direction is good | **refuted**: −2 / +6 pp (Astra at the same branch value: +31) | `m2a/runs/dose` |
| a0-test | a0 on the locked test | — | t12k 0.438 (+31 ± 8, 10/0) ≈ Astra 0.406 | `gen_t12k.json` |
| a0-collat | a0 collateral checks + MATH transfer | — | no damage; math48 0.271 (+23 ± 6, 11/0) vs Astra 0.208 | `tool_*.json`, `gen_math48.json` |

## Phase 2: generation in the loop, protocol v2 (2026-11-05)

Scorer v2 adds `submit_gen`: GGUF + llama-server on GSM8K train[200:216], 6k cap. Budget: 40 probes + 6 generation checks.

| id | arm | prediction | outcome | data |
|---|---|---|---|---|
| sanity2 | a0's edit through the generation check | — | 7/16 (base 1/16), ~10 min per check | `m2a/runs/sanity2` |
| r2 | agent + atlas | H4: r2 ≥ a1 | dev48 0.354 (+21 ± 7) | `m2a/runs/r2` |
| a1 | agent, no atlas | — | dev48 0.646 (#1), **0.667 (`062959_102`, +52 ± 8)**; **H4 refuted again** | `m2a/runs/a1` |
| a1-test | a1 `062959_102` on the locked test + MATH + collateral checks | pre-registered pick before the test | t12k **0.562** (+44 ± 10; vs Astra +16 ± 8); math48 **0.396** (+35 ± 7); no collateral damage | `gen_t12k.json`, `gen_math48.json`, `tool_*.json` |

## Paper 1, week 1 (2026-11-06/07)

| id | question | prediction | outcome | script → data |
|---|---|---|---|---|
| E1 | Is the a1 edit just a logit bias, or budget forcing in disguise? (5 decoding arms, tuned on dev48) | (1) best logit bias < a1; (2) budget forcing < Astra on MATH | (1) **confirmed** (+15 / +9 / +12.5); (2) **refuted**: bf4096 beats Astra; a1 vs bf: dev +19, t12k −6, MATH-148 **+9.5 ± 3.8** | `queue_e1.sh` → `gen_dev48.json`, `gen_t12k.json`, `gen_math*.json`, `e1_picks.txt` |
| E2 | Replicate H4 (a2 no atlas, r3 atlas) | a2 > r3; a2 ≥ +35 pp | **confirmed**: a2 0.646 (+50, 24/0), r3 0.396 (+25); a2 converges on a1's neurons | `m2a/runs/a2`, `m2a/runs/r3` |
| combo | Are the a1 edit and budget forcing complementary? | combo ≥ best of the two by +5 pp off-dev | **confirmed**: +9 pp vs a1 (11/4) and vs bf (13/6) on t12k + math48; MATH-148 +6.1 ± 3.2 vs a1 | `queue_combo.sh` |
| E3 | Is the agent needed, or do the tools + generation signal suffice? (naive search, same budget) | naive ≥ 25 pp below a1/a2 on dev48 | **confirmed**: naive 0.271 (−38 to −40 pp) | `m2a_naive.py` → `m2a/runs/naive` |
| MATH-148 | Larger OOD evaluation (MATH-500[0:148]) | combo > a1 ≥ bf > Astra; combo − a1 ≥ +5 pp | **confirmed**: 0.527 > 0.466 > 0.372 > 0.209 > 0.081; combo − a1 +6.1 ± 3.2 (borderline) | `queue_e3_math.sh` → `gen_math48.json`, `gen_math100b.json` |
| E4.0 | Is LoRA-SFT feasible on a 6 GB 1660S? | — | yes after a fix (`model.train()` for gradient checkpointing): T = 8192 peaks at 3.14 GB, 81 s/sequence | `ft_spike.py` → `ft/spike.json` |
| E4.1 | Self-distilled SFT data (GSM8K train[300:900]) | natural ~5–10%; excised ~25–40% of loops | 175 sequences: natural 57 (9.5%, confirmed); excised 118 (19.7% of 600, 42% of loops; the prediction did not state the base) | `ft_data.py` → `ft/sft_stats.json` |
| E4.2 | LoRA-SFT baseline vs the agent edit | ≥ +20 pp on dev48, below a1 (0.667); cost ≥ 5× the agent's | epoch 2 (picked on gen-check, 7/16). dev48 **0.292**: +14.6 ± 5.9 vs base (below +20), **−37.5 ± 7.1 vs a1 (0/18)**, below Astra, logit bias and bf4096; cut 0.71. Cost 17.1 GPU-h vs 1.31 per agent run (13×; 8× counting the scorer's full residency), **confirmed** ([cost.md](cost.md)) | `ft_train.py`, `ft_export.py`, `queue_e4.sh` |

## Deviations from pre-registration

- **2026-11-07, E4.2:** training max_len raised from 4096 (the automatic E4.0 pick) to 8192, *before* training. With
  4096, 7/17 natural traces and 65/103 loop prefixes would have been dropped (median loop onset at token 4186), which
  would bias the baseline toward short traces and weaken it.
- **2026-11-04:** presence-penalty window. llama.cpp applies it to the last 64 tokens by default, while vLLM applies it
  to the whole output. PoC-6 ran with 64. From PoC-7 on, `repeat_last_n = max_new + 512`.
