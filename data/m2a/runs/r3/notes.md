# r3 notebook

Start 01:14 (2026-11-06). Budget 40 probe / 6 gen. Deadline ~04:45.

## Atlas
- natural: 18 MATH loop traces, loop vs preloop means per layer (resid, mlp_in, attn_in).
- branch: 12 pairs rep vs prod, 4 system prompts.

## Atlas findings (attempt 2, started ~01:20)
- Residual rep-prod diff (branch) is identical across 4 system prompts (cos .999) -> prompts don't touch the loop mechanism; prompt effect on prod is ~orthogonal to loop dir. Explains why prompts fail.
- natural loop dir vs branch rep dir cos ~0.5-0.6 in L8-21.
- Logit lens of branch rep dir (L19-23): promotes "?", "真的", "Yes", "Okay", "是吗"; demotes "thus", "therefore", "yields", "results", "hence" -> a "re-question/re-check" vs "progress" axis.
- Exact linear decomposition (down_proj / out_proj inputs): biggest writers of T21 loop dir: L19 mlp (+.16 nat), L15 mlp (+.12), L19 attn, L16 attn, L17 attn, L21.
- L19 mlp is very sparse: neuron 11 (pre -.39 -> loop -1.16; br prod -.06 -> rep -.71, 100% consistent) and neuron 1 dominate.
- Neuron activations unaffected by anti_rep prompt.
- Files: work/resid_dirs.pt, module_deltas.pt, neuron_contrib.pt, targets.pt; log in work/log.jsonl.

## Probes
- A1 ablate L19 n11 alone: everything ~0 (rep -0.0001). Single-neuron edits too small for probes.
- B1 L19 mlp all cols, u=T19 (nat+br combined loop dir), a=2: rep -0.014, close +0.13, neutral -0.006. weak.
- B2 4 modules (L15 mlp, L19 mlp, L19 attn, L16 lin-attn) all cols a=2: rep -0.124 prod +0.023 close +0.04 neutral -0.0138 (slightly over).
- B3 MASKED top-256 cols (ranked by consistent nat+br loop contribution along u) at L15 mlp, L19 mlp, L19 attn, L14 lin-attn, a=2:
  rep -0.213 prod +0.010 close -0.057 neutral -0.0037. Masking = 2x effect, 1/4 neutral cost.  (id 012823_415)
- B4 same a=4: rep -0.50, prod +0.014, close -0.14, neutral -0.086 (!! nonlinear blowup). a window is ~2-3.
- GEN #1 on B3 started ~01:31.
- GEN#1 B3: acc 3/16 (base 1/16), cut 0.81 (base 1.00), rescues 2 losses 0, mean_tokens 5658, closed_think .19.
  Cut tails: not verbatim loops but endless doubt/over-verification ("Let's consider if there's a trick", "Wait, I might have swapped", "This suggests there is a discrepancy") with ever deeper markdown bullet nesting. Family helps a bit; need stronger.
- B4 neutral blowup suspect: L19 attn masked cols have big preloop component (pre .56 > Δ). Testing per-module at a=4.
- Singles, top256 masked, a=4:
  S19attn rep -.155 close -.46 neutral -.067  <- culprit of B4 blowup; drop L19 self_attn.
  S15mlp  rep -.073 prod -.008 close +.003 neutral +.008
  S19mlp  rep -.050 close +.126 neutral -.001
  S14attn (lin-attn) rep -.305 prod -.004 close +.175 neutral -.006  <- most efficient; DeltaNet mid-layer writes carry loop state.
- Next: map other lin-attn layers (L12,13,16,17) singly.
  S13attn rep -.271 prod +.015 close -.045 neutral +.022 (!)
  S16attn rep -.100 close +.227 neutral +.009
  S17attn rep -.075 close -.12 neutral +.003
  S12attn rep -.243 prod +.013 close -.034 neutral +.012
  => hot spot = linear-attn (Gated DeltaNet) out_proj at L12-L14. Self-attn L19 harmful.
- C1a4 (lin-attn L12,L13,L14 + L15 mlp, top256, a=4) id 022748_993: rep -.463 prod +.0004 close +.097 neutral +.013. All limits passed.
- C1a3 same a=3 id 023620_171: rep -.380 prod +.003 close +.075 neutral +.017.
- GEN#2 on C1a4 started 02:38.
- GEN#2 C1a4: acc 4/16 (base 1/16), cut .75, rescues 3 losses 0, mean_tokens 5456. Better than B3 (3/16).
  Tails: still reconsideration loops ("Let's consider the phrase...", "Is it possible that...", "Wait, if...") with deep tab/bullet nesting. Not verbatim.
- C3 (top512 a=4) 024949_792: rep -.505 prod -.011 close -.002 neutral +.012
- C4 (top256 a=6) 025053_813: rep -.532 prod -.007 close +.097 neutral -.006   -> family saturates ~ -0.5
- C2 (lin-attn L10,L12,L13,L14 top256 a=4) 030051_017: rep -.536 prod -.012 close +.344 neutral +.005  (more willing to close)
- </think> direction (tied unembed through final norm) ~orthogonal to loop dirs (|cos|<.03); closure-specific edit from mean data too crude -> skipped.
- GEN#3 on C2 started 03:04.
- GEN#3 C2: acc 1/16, cut .94, rescues 0 -> NO gain despite strongest probe (rep -.536) and close +.34. Proxy gaming confirmed.
  Tails: "Maybe she sells...? No", "Is there any interpretation where...? Yes" - question/answer reconsideration.
  Comparison: B3 (L15mlp,L19mlp,L19attn,L14lin) 3/16; C1a4 (L12-14 lin + L15 mlp) 4/16; C2 (L10,12,13,14 lin, no MLP) 1/16.
  Hypothesis: the generation-relevant part is the MLP writers of the NATURAL loop direction (L15 mlp is in both good ones);
  lin-attn edits mostly lower verbatim re-check likelihood (branch proxy). Next: MLP-only, natural-loop direction family (E).
- E1 (nat dir, MLP L13,L14,L15,L19 top256 a=4) 031439_047: rep -.119 prod +.002 close +.535 neutral +.002. Distinct profile: small rep, big close.
- E2 (nat dir, MLP L15,16,17,19 a=6) 032202_207: rep -.052 close +.69 neutral -.024 (fails neutral).
- GEN#4 on E1 started 03:23.
- GEN#4 E1: acc 2/16, cut .875, rescues 1. close +.54 on probe did NOT yield more closing (closed_think .125). Not better.
  Tally: C1a4 4/16 > B3 3/16 > E1 2/16 > C2 1/16 = base 1/16. Both good ones contain L14 lin-attn + L15 mlp with combined (nat+br) dir.
- GEN#5 on C4 (C1 family, a=6) started 03:36. Plan: probe hybrid F1 (L13 lin, L14 lin, L15 mlp, L19 mlp; combined dir) then GEN#6.
- GEN#5 C4 (C1 family a=6): acc 4/16, cut .6875 (best), closed_think .31, mean_tokens 5181 (lowest), rescues 3 losses 0.
  Solved {3,8,10,15} vs C1a4 {2,3,7,8}. Stronger within family -> lower cut/tokens. C1 family is the real effect.
- F1 hybrid (L13,L14 lin + L15 mlp + L19 mlp, comb dir, top256 a=4) 034502_932: rep -.392 prod -.002 close +.259 neutral +.010.
- C5 (C4 but L15 mlp a=9) 034606_403: rep -.541 neutral -.057 FAIL (L15 mlp blows up at high a).
- C6 (lin-attn a=8, L15 a=6) 034710_488: rep -.492 (overshoot, weaker than C4) neutral -.018 FAIL. => C4 is near the sweet spot.
- GEN#6 on F1 started 03:48.
- GEN#6 F1: acc 5/16 (best), cut .6875, closed_think .31, mean_tokens 5514, rescues 4 losses 0.
## Final (04:00): ranked F1 034502_932 > C4 025053_813 > C1a4 022748_993. Probes used 23/40, gens 6/6.
