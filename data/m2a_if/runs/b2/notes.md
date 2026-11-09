# Run b2 notes (IFEval weight edit)

Start 11:10. Helpers in work/ (common.py: weight loading; sub.py: submit+log to work/results.jsonl).

## Facts
- eos for chat = <|im_end|> id 248046. Final norm gain = 1+w (mean w≈3.3).
- Effective unembed for token t: (1+g)⊙E[t]. EOS direction (raw) also promotes \n\n, \n, space, comma: very generic.

## Probes
1. 111252_433 eos23_a-2: amplify raw EOS dir in L23 MLP x3 -> neutral -0.76 (catastrophic), if_sel ~0. Lesson: L23 MLP writes heavily along
   generic output directions; token-unembed directions in L23 need |alpha| << 1 or centered/specific u.
2-3. 111621_086 / 111745_199 scanA: amplify top-4 left-SVs of full-attn o_proj (L11, L15) x2 -> all effects < 0.01. SVD directions
   are arbitrary; aborted the scan (L19/23/7 not run).
4. 112126_253 stopneur4_a-2: amplify x3 the most EOS-specific late MLP neurons (L23n1116 rank1 EOS, L22n2835 rank1, L23n2025, L19n177)
   (cols=[j], u=col dir). neutral -0.002; end_phrase -0.036, postscript -0.038 (neurons fire +EOS at paragraph end, as expected),
   others ~0. -> GEN CHECK 1 launched 11:23 (anti-loop family test).

## Weight analyses
- OV copy scores (embedding-approx) full-attn heads: strong copy L7H2 8.6, L11H4 7.7, L11H7 8.3, L3H0 6.2; strong negative (copy
  suppression?) L7H0 -9.8, L11H6 -9.4, L23H2 -5.7. Not usable directly: token copying is high-rank, edits are rank-1.
- Key constraint: rank-1 edit = gain on ONE residual direction from chosen columns. Only "global scalar" features can be modulated
  (stop, case, comma-ness...). Selectivity idea: amplify feature in ATTENTION writes (carry instruction info from prompt) rather than MLP.
- Case direction (mean G[ALLCAPS]-G[lower], 1819 pairs, 98% sign-consistent): L23 attn writes 1.9x isotropic along it.
- Note: scorer processes calls serially; probes queue behind a gen check (~40 min idle). Batch probes before gen checks.

## GEN 1 (11:31, took 8 min, not 40): stopneur4_a-2 (112126_253)
prompt_strict 0.5625 vs base 0.578; inst 0.705 vs base 0.728; cut 0.023 vs base 0.078; mean_tokens 250; rescues 12 / losses 14.
-> Stopping neurons DO cut loops by 2/3 but loop prompts still fail (repeated content, e.g. ALLCAPS eulogy loop that then stops; still
   missing keywords). Net ~0 (noise). Base cut is only 7.8% (10 prompts) so loop-fixing alone has a small ceiling.
Fail examples: missing placeholders [..], missing keywords (existence) x2, forbidden word used (keywords:frequency "lecture < 1").
Per-group (cand): change_case .55, punctuation .62, length .63, combination .70, startend .71, keywords .71, det_content .75, det_format .88.
5-9. Feature-direction amplification in attention o_proj (unembed-derived dirs):
   - raw case dir attn19+23 a-3: upper +0.46 but neutral -1.04 (raw unembed dirs = generic component -> catastrophic).
   - whitened (cov^-1) dirs are clean (logit lens specific). wcase attn23 a-3: upper -0.06 (wrong way), neu -0.026.
   - wcomma attn19+23: no_comma -0.064 (attention writes PROMOTE commas = prior, not instruction).
   - weosnn attn19+23: posts +0.074, end +0.031, title +0.044, ctrl +0.013, neu -0.033.  wquote: ~0.
   => late-attention feature writes mostly carry the prior; per-format amplification is not selective. Dropping this family.
- Gen sampling (src/wsp/llama_gen.py): likely NOTHINK T=0.7 top_p .8 top_k 20 presence_penalty 1.5 (llama-server, prompt in sampler).
10-11. F3 neuron-level format features (late MLP neurons aligned with whitened feature dirs, amplify feature component x3):
   - case neurons L22/23 (14+7 cols): upper -0.93, ctrl -0.159, neu -0.146 -> these are PRIOR normal-case enforcers (not selective).
   - comma neurons: ~0. Neuron/feature-level format edits look non-selective; abandon.
- Instruction-token neuron detection from embeddings (L0-7): selectivity ~0.5 sd only; too weak to use.
- Dim 0 is a bulk/massive dim: final norm gain 0.22 (smallest), heavy rows in many layers. Idea: shrink/grow dim 0 in L23 output
  to change final rms -> global logit temperature. Signature: per_constraint Δ ∝ base_margin.
12-13. dim0 at L23 MLP: amplify x4 -> neu -0.15, partial flattening signature; remove 60% -> neu +0.008, margins ~0.
   remove dim0 writes L22m/L23a/L23m (a=1): neu +0.002, all ~0. Not a usable temperature knob. Dropped.
14. weosnn attn19+23 a-2: if .014 ctrl .009 neu -.015; posts +.054 title +.031 end +.019.
15-16. stop2+eosnn attn combo: if .004 ctrl .002 neu -.014 (weak). wcase attn23 removal: ~0.
- Rank-1-ish heads: L0 DeltaNet heads have OV energy 0.6-0.76 in top SV (structural), not interpretable.
## KEY FINDING (weights): global DeltaNet memory knob
- Stacking the 288 decay-gate rows (in_proj_a * input-ln gain, 18 GDN layers x 16 heads): top SV energy 0.215 (iso .001), and 99.7% of
  rows read it with the SAME sign (mean cos 0.44). Same for beta gate rows (in_proj_b; 98.6% same sign, cos(da,db)=0.37).
  => one residual direction d_a sets forgetting rate of ALL DeltaNet heads (larger proj -> larger a -> faster decay).
  Written mainly by early linear_attn out_proj (L0-L5: 1.6-2.1x isotropic); embeddings ~0 along it. Saved work/dir_a.pt, dir_b.pt.
  Hypothesis: lowering the residual component along d_a = slower forgetting = instructions remembered over long answers.
17-20. d_a edits on linear_attn L0,1,2,4 (all cols): rm(a=1): neu -.0125, margins slightly -; amp x2: neu +.0056; amp x5 (a=-4):
   neu +.0104 (!), if +.0045 ctrl +.007, end +.02 quote +.018 posts +.018 no_comma -.009; amp x10: neu -.06, upper -.093 (too far).
   -> a=-4 (120302_651) improves generic LM logp; sent to GEN 2 at 12:05.

## GEN 2 (12:11): decay_amp4_L0124 (120302_651) -- BEST SO FAR
prompt_strict 0.625 vs base 0.578 (+4.7); inst 0.746 vs 0.728; rescues 15 / losses 9 (net +6); cut 0.0625 (base .078); mean_tokens 363.
per-group: change_case .62, combination .70, det_content .83, det_format .88, keywords .71, length .65, punctuation .69, startend .94.
(vs GEN1 stopneur: case .55, content .75, punct .62, startend .71). Gain spread over several families; within the 8pt noise band but
consistent in inst_strict + rescues/losses. Fails: placeholders missing, keywords missing (x2, one ALLCAPS loop cut), quotation (P.P.S outside quotes).
21-23. d_a amp x5 on L1,4,5,8: neu -.014, if_sel -.012 (worse). d_a amp x7 on L0,1,2,4: neu -.002 (past optimum).
   beta dir (orth) amp x5 L1,4,8,10: neu -.021, worse. -> effect is layer-specific; decomposing per layer.
24-27. per-layer d_a amp x5: L0 neu -.0096 (if .0067 ctrl .0088), L1 neu -.0058, L2 neu +.012, L4 neu +.003; margins all small.
   Non-additive (sum -.0003 vs joint +.0104). Probe margins don't see the gen effect (short answers); gens needed.
28. decay_rw_L2L4x7_L0L1x3 (122209_766): neutral +0.026 (!), margins ~0 (no_comma -.026). -> GEN 3 at 12:24.

## GEN 3 (12:31): decay_rw (122209_766, neutral +.026)
prompt_strict 0.578 = base; inst .719 (base .728); rescues 14 / losses 14; cut .078; mean_tokens 384. punct .54, case .66, startend .94.
-> big neutral gain did NOT translate to IFEval. GEN2's +6 net is ~1.1 sigma (discordant ~28 -> sd ~5). Family effect is small/uncertain.
Next idea: other global gate channels (full-attn output gate rows in q_proj, GDN z gate rows, beta) -> shared read directions.
- Full-attn OUTPUT GATE: per layer, a single residual direction is read by ~100% of the 2048 gate units (mean cos .29-.47; top SV
  energy .10-.25). => per-layer "attention volume" knob (sigmoid gate of all 8 heads). Saved work/dir_gates.pt. GDN z-gate: no shared dir.
29-31. Attention-volume knob, amplify x5 writes along gate dir in top writers: gate11: if -.048 ctrl -.239 neu -.22; gate15: if -.056
   ctrl -.333 neu -.40; gate19: if -.43 ctrl -.49 neu -.98 (upper -3.05). Knob is STRONG; amplification hurts (ctrl more than if).
   Testing the removal sign at small alpha.
32-33. gate15/gate11 removal 50%: all ~0, neu -.012/-.007. Gate knob only destructive when amplified. Dropped.

## GEN 4 (12:45): decay L0-heavy (123756_003)
prompt_strict 0.594 (+1.6); inst .728 (= base); rescues 15 / losses 13; cut .078; mean_tokens 383. case .69, combination .50, startend .82.
Decay family over 3 gens: +4.7, 0.0, +1.6 (mean ~+2.1, all within noise). Per-prompt overlap shows 47/128 prompts flip across
candidates (pure sampling reshuffle); 29 never pass (placeholders, keywords existence, forbidden word, ALLCAPS loops).
GEN 5 launched 12:51: CONTROL = opposite sign of GEN2 (d_a removal a=+1 on L0,1,2,4; 115957_748, neu -.0125). If knob is real, expect a loss.

## GEN 5 (12:56): CONTROL, opposite sign of GEN2 (115957_748, d_a removal)
prompt_strict 0.594 (+1.6), inst .750, rescues 17 / losses 15, cut .047, mean_tokens 304.
=> the decay knob has NO detectable directional effect on IFEval: both signs land at +1.6..+4.7. GEN2's +4.7 is noise-compatible.
All 5 gens: 0.5625 .. 0.625 vs base .578 (mean .591). Nothing beyond the 8-pt noise band.
35. weosnn attn19+23 a-1.5 (125218_060): if +.0108 ctrl +.007 sel +.0037 neu -.0092; posts +.042 title +.024 end +.014.
36. weosnn attn11/15/19/23 a-1.5: if +.014 ctrl +.0098 neu -.0134 (over the neutral limit).
-> GEN 6 (last) on 125218_060 at 12:55: the only weight-derived edit with a consistent (if weak) instruction-selective probe signal.

## GEN 6 (13:01): weosnn attn19+23 a-1.5 (125218_060)
prompt_strict 0.555 (-2.3), inst .710, rescues 9 / losses 12, cut .078, mean_tokens 399. det_content 1.00, startend .94 (P.S. helped as the
probe said) but length .52, case .52 (longer answers break length limits). Net negative -> probe-selective signal did not pay off.

## SUMMARY (gens used 6/6, probes 36/40)
| gen | cid | edit | prompt | inst | net | cut | neutral |
| 1 | 112126_253 | 4 EOS stop neurons x3 (L23,L22,L19) | .5625 | .705 | -2 | .023 | -.002 |
| 2 | 120302_651 | d_a amp x5, linear_attn L0,1,2,4 | .625 | .746 | +6 | .0625 | +.010 |
| 3 | 122209_766 | d_a amp L2,L4 x7 / L0,L1 x3 | .578 | .719 | 0 | .078 | +.026 |
| 4 | 123756_003 | d_a amp L0 x7, L1 x5, L2,L4 x3 | .594 | .728 | +2 | .078 | -.012 |
| 5 | 115957_748 | d_a REMOVAL (control) L0,1,2,4 | .594 | .750 | +2 | .047 | -.0125 |
| 6 | 125218_060 | EOS-vs-\n\n attn L19,L23 x2.5 | .555 | .710 | -3 | .078 | -.009 |
base .578 / .728 / cut .078. No candidate beyond the ~8-pt noise band; the d_a knob gives +0..+4.7 with EITHER sign, so no directional effect.
Stopped with 4 probes left: probes cannot change a gen-based ranking and no gen budget remains.
