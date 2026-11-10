# f1 notes (no-atlas control arm)

## Weight-only analysis (work/a1.py, a2.py)
- Final norm weight g is plain (mean 3.3). Logit direction for token t in residual: u = g * (E[t] - mean(E)).
- Directions saved in work/U.pt: wait (Wait/ Wait), yes, close (</think>), q (?), but (But/Actually/Hmm).
- Projection of every MLP down_proj column on u_wait (work/mlp_proj.pt): clear outlier "wait neurons"
  (z ~ 14, logit lens = wait/waiting/Wait): L17 n1915, L18 n881, L22 n573; weaker L20 n405 (delay/wait, z 9).
  L16 has a strong anti-wait neuron (z=-15). L23 has a yes-neuron (z=20.7), q-outliers at L21-23.
- Hypothesis H1: re-check loops are entered via "Wait"; flipping what the wait-neurons write along u_wait
  (alpha>1 on the single column) suppresses Wait only when these neurons fire -> minimal side effects.

## Probe log
- p1 153129_537: 4 wait neurons (L17 n1915, L18 n881, L22 n573, L20 n405) u_wait a=1: rep -0.033 prod -0.005.
- p2 153325_760: same a=6: rep -1.30 prod -0.156 close +0.05 neutral -0.0008  (prod over soft limit)
- p3 153508_876: u_wait removed from ALL cols L20-23 MLP a=1: rep -0.038 prod +0.025 close +0.32 -> weak; neuron-targeted is far stronger.
- p4 L22 n573 alone a=6: rep -0.58 prod -0.13.  p5 L17 alone: ~0. p6 L18 alone: ~0. p7 L20 alone: ~0.
  -> non-additive in logp (margin effect: Wait is predicted with large margin in rep continuation; logit shifts add up).
- Whitened (C^-1) wait direction is ~same as plain (cos .93); plain is already token-specific (work/a3.py).
- GEN1 on p2 to see whether the Wait-suppression family helps and how the cut answers look.
- **GEN1 (p2 153325_760): acc 0/16, cut 16/16 (base acc 1/16, base cut 16/16!). Tails = "WaitWaitWaitWait...".**
  Lesson: MLP is SwiGLU (silu(gate)*up) so neuron activations are SIGNED. The wait-neurons act as anti-Wait
  (negative act) right after a Wait token; flipping (alpha>1) turns that into positive feedback -> runaway.
  => never use alpha>1 on token neurons without sign knowledge. Sign-safe options: alpha in (0,1] (dampen,
  both promotion and suppression) or alpha<0 (sharpen what a layer already writes along u).
- Base is cut on 16/16 gen problems at 6144 tokens -> finishing the thinking at all is the bottleneck.
- p3 hint: removing Wait dir from late MLPs raises close (+0.32): Wait competes with </think> at decision points.
- H2: amplify (alpha<0) the </think> logit direction in the last layers' writes = sharpen close decisions
  (more close where the layers already vote close, more suppression elsewhere). Keep close <= +1.
- Token-neuron inventory (work/a4.py, outliers.pt): check-neurons L18 n1824/n1126/n1118/n2486, L20 n2684, L21 n474;
  yes-neuron L23 n3308; anti-again L22 n338; let-neurons L18 n1130/n1261; etc. (signs of activations unknown)
- p8 155731_496: close dir a=-3 all cols L22/L23 mlp+attn: close +5.78 (!) neutral -0.47 rep -0.23 prod -0.13. Far too strong, but close is very controllable.
- p9 155913_353: whitened close (lam=1) a=-1 same 4 matrices: close +0.27 neutral -0.0085 rep -0.01.
- p10 160053_326: L23 only (mlp+attn), whitened0.1, a=-3: close +3.40 neutral -0.156 prod -0.07. ~0.04 neutral per close nat -> inefficient.
- p3 in contrast gets close +0.32 with neutral +0.005 by *dampening* Wait. Explore Wait dampening (alpha=1) more widely.
- p11 161317_630: Wait dampen a=1 all cols L22/L23 mlp+attn: rep -0.131 (cases 6-11 -0.25) prod +0.028 close +0.31 neutral +0.003. Safe -> GEN2.
- **GEN2 (p11 161317_630): acc 7/16 (base 1/16), cut 0.50 (base 1.0), closed_think 0.56, mean_tokens 4681, rescues 6, losses 0.**
  Remaining cut tails: verbatim sentence repeats / restating the question (no "Wait"). Wait-dampening family works.
- Next: strengthen (alpha 2?), find which matrices matter (singles), other layers, add close sharpening.
- p12 162813_917: same as p11 with a=2: rep -0.378 close +1.09 but neutral -0.054 -> over the limit; a=1 is the sweet spot.
- singles (Wait a=1 all cols): L23 self_attn: rep -0.072 close -0.18 | L23 mlp: rep -0.022 close +0.34 prod +0.019 | L22 linear_attn: ~0.
  -> effect lives in L23 (pure logit edits). L23 mlp = close raiser, L23 attn = rep reducer (cases 6-11).
- Cut tails of GEN2 show self-questions ("How many ... per week?") -> try dampening "?" and But/Actually/Hmm at L23.
- p16 163548_970: "?" dampen a=1 L23 mlp+attn: neutral -0.21 prod -0.45 -> destructive (direction not specific). Dead.
- p17 163644_140: But/Actually/Hmm plain dir a=1 L23 mlp+attn: rep -0.133 (cases 0-5) prod +0.025 close +1.08 neutral -0.033 -> too costly on neutral.
- p18 164851_114: Wait + whitened Actually/Hmm (orth to wait) a=1 L23 mlp+attn: rep -0.139 prod +0.017 close -0.13 neutral +0.004 -> no gain over p11, loses the close bonus.
- p19 165020_381: Wait L23 attn a=2 + L23 mlp a=1: rep -0.239 prod +0.027 close +0.26 neutral -0.002 -> the a=2 neutral harm came from the MLP, not attn. -> GEN3
- p20 165200_724: attn a=4: rep -0.478 neutral -0.027 (too much). attn alpha ~2-2.5 is the limit.
- **GEN3 (p19 165020_381): acc 9/16, cut 0.375, closed_think 0.625, mean_tokens 4281, rescues 8, losses 0.** Best so far.
  Remaining tails: "do they use plates ...? Yes." self-Q&A loops; enumerating variations ("Or a year of 366 days (unlikely)").
- Next: add Yes dampening (L23), attn alpha 2.5-3.
- p21 170531_698: p19 + Yes dampen (L23 mlp a=1, attn a=2): rep -0.369 prod -0.010 close +1.38 neutral -0.055 -> over neutral/close limits. Dead.
- (interrupted by rate limit ~17:20-18:10; resumed 18:11 with 19 probes / 3 gens left, ~45 min of wall-clock target)
- p22 181117_952: Wait L23 attn a=2.5 + mlp a=1: rep -0.297 prod +0.027 close +0.26 neutral -0.0067 (within limits) -> GEN4
- **GEN4 (p22 181117_952): acc 9/16, cut 0.4375, closed_think 0.5625, mean_tokens 4709, rescues 8, losses 0.** Ties p19 on acc, slightly more cuts -> no gain from a=2.5.
- FINAL ranking: p19 165020_381 > p22 181117_952 > p11 161317_630. Used 22/40 probes, 4/6 gens. Stopped for wall-clock.
