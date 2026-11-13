# Run a3 — no-mapping control arm (no atlas)
Start 03:50. Sources: brief, base weights (CPU, read-only), scorer.
Budget: 40 probes, 6 gen checks.

## Plan
- Weight-only analysis: effective unembedding dirs (final norm ⊙ E) for loop-related tokens
  ("Wait", "Hmm", re-check words, "Yes"), and for closing/progress tokens.
- Find late-layer MLP neurons whose down_proj columns write those directions (logit lens on weights).
- Families: (A) suppress re-check/"Wait" writing; (B) copy/induction head OV suppression; (C) mild
  boost of conclusion. Probe as filter, gen-check as objective.

## Weight analysis (no atlas)
- Readout dir for token t = (1+g_final) ⊙ E_t (Qwen3.5 RMSNorm is 1+w). dirs saved in work/dirs.pt
  (wait = mean of 4 wait variants; hes = wait+Hmm+Actually+Alternatively; close = </think>).
- "wait" neurons (down_proj col cos~0.5 with wait dir): L16/998 (gate reads actually/hmm/maybe/? -> hesitation!),
  L17/1915, L18/881, L20/405, L22/573 (semantic wait/delay neurons). L14/1443 writes '?'. L19/635 'expected'.
- No standout </think> neurons.
- MEMORY: full fp32 Eg copy => 3.2GB peak. Use chunked bf16 matmul (an3.py pattern, 1.3GB).

## Probes
- A1 035252_568 4 wait neurons, u=wait, a=3: rep -0.015 prod +0.015 close -0.06 neu -0.001. Tiny.
- A2 035409_666 u=hes all cols L16,17,18,22 a=3: rep -0.182 prod +0.025 close -0.98 neu -0.001.
  Cases 6-11 big (-0.25..-0.39), 0-5 small. close drops ~1 nat (less stopping) -> risk for cut rate.
- A3 035523_924 u=hes ⊥ close, same: rep -0.177 prod +0.021 close +0.645 neu -0.001.
  => orthogonalizing to </think> flips close effect from -0.98 to +0.65 (softmax competition: Wait loses, </think> gains).
  GEN CHECK #1 launched on A3 (03:56).
- GEN #1 A3: acc 0.4375 cut 0.50 closed 0.56 tokens 4863 | base acc 0.0625 cut 1.00 | rescues 6 losses 0.
  Tails: 1 variation loop ("Does this mean they bring *another* person?" repeat), 2 long deliberation (not hard loops).
  => hesitation suppression family works. Next: strength/layers/close knob.
- B1 040936_507 A3 a=6: rep -0.576 prod +0.015 close +1.50 neu -0.020 (violates close & neutral).
- B2 041041_487 MLP L20-23 a=3: rep -0.49 prod -0.41 close -0.92 neu -0.17 -> BAD (L23 MLP/late disrupts everything).
- B5 041227_475 attn outs L15/19/23 self + L22 lin a=3: rep -0.183 prod +0.004 close +0.34 neu -0.011.
  Attention path also carries hesitation. Next: single-edit layer scan to pick best 4 slots.
- C single-slot scan (hes_pc, a=4): L22mlp rep -0.112 close +0.95 (dominant, hits cases 0-5 too) | L16mlp -0.041 close -0.07 |
  L23self -0.048 neu -0.010 (bad neutral) | L19self -0.037 close 0 | L21mlp -0.016 | L15self -0.015 close -0.17 |
  L17mlp -0.009 | L18mlp -0.004 close -0.28. Combos are super-additive (A3 > sum of parts).
- D1 044632_919 L22mlp a3 + L16mlp a5 + L19attn a5 + L21mlp a4: rep -0.374 prod +0.026 close +0.75 neu -0.003. (in limits)
- D2 044757_970 L22mlp a4 + L16mlp a6 + L19attn a6 + L15attn a6: rep -0.573 prod +0.021 close +0.70 neu -0.0100 (at limit).
  GEN #2 on D2 (05:00).
- GEN #2 D2: acc 0.625 cut 0.25 closed 0.75 tokens 4084 | rescues 9 losses 0. BEST so far.
  Tails: 2 of 3 are verbatim bullet loops ("*   $9 = 27$." x N, "This implies the van has 28 spots..." x N),
  1 long deliberation. Remaining failure type = bullet-list verbatim copy loop (no 'Wait').
- E1 050947_620 L22 a6 + L16 a6 + L19attn a6 + L18mlp a6: rep -0.634 prod +0.020 close +0.90 neu -0.0146 (neu violates)
- E2 051112_718 D2 with L22 a6: rep -0.645 close +1.25 neu -0.0144 (violates both)
  => neutral cost grows with strength; try direction excluding lowercase prose 'wait'.
- F1 052229_828 D2 config, u=caps-hesitation ⊥{close, lowercase wait}: rep -0.081 close +0.77 neu -0.0144. WORSE: 'Wait' itself is key; neutral cost not from prose 'wait'.
- G1 052408_494 u=close L23mlp a=-2: close +2.10 neu -0.015 rep ~0. (L23 = dirty)
- G2 052532_458 u=close L21+L22 mlp a=-3: close +3.03 neu -0.0007 prod -0.003 rep 0. CLEAN close knob (L21/22).
- H1 052739_989 L22 hes a4 + L16 hes a6 + L19attn hes a6 + L21 close a=-1: rep -0.457 prod +0.022 close +1.50 neu -0.004.
  (dropping L15attn improved neutral; L15attn likely source of D2's neutral cost). GEN #3 on H1 (05:29).
- GEN #3 H1: acc 0.4375 cut 0.4375 closed 0.56 tokens 4818 | rescues 6 losses 0. WORSE than D2.
  Close amplification did not help (bullet/question loops persist: "If 15 people eat... ?" x N). Dropping L15attn may also have cost.
  Trend: gen gain tracks hesitation-suppression strength (A3 rep -0.18 -> 0.44; D2 rep -0.57 -> 0.625).
- I1 055128_409 L22 a4, L16 a8, L19attn a8, L15attn a6: rep -0.703 prod +0.021 close +0.69 neu -0.0142
- I2 055255_463 L22 a4, L16 a8, L19attn a8, L17mlp a8: rep -0.847 prod +0.017 close +0.99 neu -0.0191
  GEN #4 on I1 (06:06): does more hesitation strength keep helping?
- GEN #4 I1: acc 0.625 cut 0.3125 closed 0.69 tokens 3851 | rescues 9 losses 0. Same correct set as D2 (plateau), shorter.
  Remaining failures: verbatim bullet-line loops ("If they are distinct, they need separate plates." xN), long deliberation.
- J1 061511_593 L22 a4, L16 a8, L19attn a8, L23attn a4: rep -1.00 prod +0.008 close +1.45 neu -0.0086 (cases 0-5 doubled)
- J2 061613_590 same, L22 a2.5: rep -0.90 prod +0.013 close +0.97 neu -0.0079. ALL LIMITS PASS, strongest. GEN #5 on J2 (06:17).
- GEN #5 J2: acc 0.625 cut 0.375 closed 0.625 tokens 4228 | rescues 9 losses 0. Same correct set again (plateau).
  Tails now show VARIATION loops (alternating 2/3 apples per week with growing week numbers; alternating boys/girls),
  i.e. the brief's warned failure mode appears as branch_rep is pushed to -0.9.
## Conclusion
Probe strength past rep ~ -0.57 (D2) buys no gen gain (D2 = I1 = J2 = 10/16 correct, identical sets), and pushing it
harder turns verbatim loops into variation loops. Ranking: D2 > I1 > J2. Used 26/40 probes, 5/6 gens (last gen not spent;
marginal value low vs. freeing the GPU queue).
