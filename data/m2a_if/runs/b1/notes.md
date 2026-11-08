# Run b1 notes (IFEval weight-space programming)

## Setup
- helper: work/lib.py (tokdir = (1+norm_g) ⊙ E[tok], normalized; topsv; batch submit with threads; results.jsonl)
- final norm weight mean 3.3 -> Qwen3.5 uses (1+w) RMSNorm gain.
- Logit lens of top SVs of late o_proj / down_proj: not interpretable (junk tokens).

## Hypotheses
H1 EOS sharpening: amplify the EOS (<|im_end|>) component written by late MLPs -> model stops when done, fewer
   degenerate loops (1/3 of base failures). Risk: shorter answers hurt "at least N words".
H2 Full-attn amplification: scale principal output subspace of a full-attention o_proj (only layers that can
   retrieve the instruction verbatim at long range) -> more instruction influence.

## Batch 1 (8 probes)
- eos_amp_L23 (a=-3, all cols): neutral -2.96 (!), if -0.15. Amplifying a residual direction x4 in the last MLP inflates
  the pre-norm residual and flattens logits. eos_amp_L20-23 a=-2: neutral -1.02. => EOS amp with full cols is broken.
- attn_amp4_L23/L19 (top-4 SV x1.5): all metrics ~0 (spectrum flat; top-4 SVs are a tiny part of o_proj).
- Weight-only OV analysis (E->V->O->unembed diag z-score): copy heads L3h0, L7h1/h2, L11h3/h4/h7, L15h1;
  copy-suppression heads L7h0/h3, L11h0/h6, L23h2 (z=-5.8). But copy/suppression is spread across ~all dims
  (symmetrized OV eigen spectrum is flat besides a mean-component ± pair) -> rank-1 cannot scale a head. Dropped.

## Batch 2: instruction-keyword directions (weights-only estimates; crude), gentle EOS on 64 neurons
- Batch 2 results: ALL within noise (|if_sel| < 0.004). instrV_L15 best (if_sel +0.0037, title +0.018), instrV_L19 -0.003,
  instrN_L16/L18 ~0, eos_amp_n64_L23 (a=-1.5, 64 cols) neutral -0.0099, ctrl +0.0035, if_sel -0.004.
- Lesson: weights-only "instruction" directions (built from raw embeddings) don't line up with real residual features.
- Insight: rank-1 with cols=[n] and u = W[:,n]/|W[:,n]| scales neuron n exactly by (1-alpha). So 4 edits = 4 neuron
  gains. Entropy-neuron candidates (L23 cols with high norm in unembedding null space): 291, 46, 64.
- Gen check #1 on eos_amp_n64_L23 (184512_099): tests the "sharper stopping" hypothesis + gives base fail modes.
- Batch 3: instrV_L15 a=-10 neutral -0.030, L15+L11 a=-10 neutral -0.037, ctrl up. instrV family dead.
## GEN #1: eos_amp_n64_L23 (184512_099) -> prompt_strict 0.625 vs base 0.500 (rescues 8, losses 2), inst 0.676 vs 0.662,
   cut 0.125 vs base 0.1875, mean_tokens 524. Fails still show degenerate loops ("a. a. a." list loop), unrepeated
   request, commas, Hebrew instead of Telugu. Gen is evidently greedy and fast (~214 s).
   => EOS-stop sharpening is a real lever (fewer cap hits). Next: strengthen while keeping neutral >= -0.01.
- Batch 4: eosnn_n32_L23 (u=EOS-"\n\n", 32 neurons, a=-1.5): neutral +0.0027, probes ~0 except postscript -0.019.
  ent_amp3 / ent_kill3 (L23 neurons 291,46,64 x2 / x0): all ~0. Entropy-neuron idea dead in probes.
- GEN #2 queued on eosnn_n32_L23. Batch 5 = EOS-family variants (n32 a=-3, L22+L23, eos64+eosnn32, n128 a=-1).
- Batch 5: eos_n32 a=-3 neutral -0.0147 (too much); eos_n64_L22L23 neutral -0.0099 (~same as n64 L23 alone);
  eos64+eosnn32 neutral -0.0036 but postscript -0.019; eos_n128 a=-1 neutral -0.014. All if_sel ~ -0.004 (probes
  don't capture stopping).
## GEN #2: eosnn_n32_L23 (185945_884) -> 0.4375 vs base 0.50 (rescues 5, losses 8). Ends before P.S./end phrase
   (fail examples: missing P.P.S, missing final phrase). The probe postscript -0.019 predicted this. Suppressing
   "\n\n" vs EOS is harmful; plain EOS amplification is the useful part.
- GEN #3 on eos64+eosnn32 (190300_554) already queued; GEN #4 on eos_n64_L22L23 (190259_353) as replication of EOS family.
## GEN #3: eos64+eosnn32 (190300_554) -> 0.50 (rescues 8, losses 8), cut 0.104, mean_tokens 373, punctuation 0.0,
   detectable_content 0.333 (P.S. dropped again). eosnn part cancels the eos64 gain. High churn (8/8) => gen noise
   on 48 prompts is large; treat single-gen deltas of ~±6 pts as weak evidence.
## GEN #4: eos_n64_L22L23 (190259_353) -> 0.50 (rescues 6, losses 6), inst 0.606, cut 0.125, punctuation 0.0.
   Adding a probe-neutral L22 EOS edit erased gen #1's +12.5 pts => gen #1 gain is fragile, likely substantially
   noise from greedy-decoding churn (every edit flips 10-16 of 48 prompts). The consistent effect across the EOS
   family is a lower cap-hit rate (0.104-0.146 vs 0.1875) - the loop-termination mechanism is real, but its net
   strict-accuracy gain is not established.
## Final: ranked [184512_099, 190259_353, 190300_554]. Used 23/40 probes, 4/6 gens (stopped for wall time).
