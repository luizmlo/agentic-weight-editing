# a1 notes (no-mapping control arm: no atlas, no forward passes)

## Setup
- Budget 40 probes / 6 gens. Full attn: q_proj 4096 (8 heads x256, + gate), kv 2 heads x256, o_proj 8 heads x 256.
  linear_attn out_proj 2048 = 16 heads x 128.

## Weight-only analysis (logit lens of MLP neurons, reading dir = E*(1+g_final))
- Per-layer write capacity along token dirs is unremarkable (~1x random) -> need neuron-level view.
- Token neurons (z-score vs vocab): Wait: L17n1915(16.5) L18n881(15.4) L20n405(10.4, delay) L22n573(18.4).
  Yes: L23n3308(18.7). check: L21n474 L20n2684 L17n1743 L16n1343. double: L21n456 L18n597 L19n2740.
  re-: L18n1405 L23n1628. verify: L22n865. '</' closers: L23n1116 n2025, L22n2835.
  L23n80/n211: sentence starters (This/If/Actually/But...).

## Probes
- P01 (060214_421) ablate 4 Wait neurons (single-col, u=own col, a=1): rep -0.104, prod -0.009, close +0.036, neutral -0.0001.
  per-case: cases 0-5 ~ -0.15..-0.29, cases 6-11 ~0. => repetitive continuations 0-5 involve 'Wait'.
- P02 (060319_643) same, a=3: rep -1.07 (all cases ~-0.6..-1.7), prod -0.092, close +0.09. Too disruptive on prod.
- P03 (060500_570) a=2: rep -0.43 (0-5 ~-0.5..-0.9, 6-11 ~-0.17..-0.36), prod -0.043, close +0.065, neutral -0.0002.
- P04 (060648_332) check neurons a=2: all ~0. Repetitive continuations don't hinge on 'check'.
- P05 (060833_060) Yes neuron L23n3308 a=2: rep +0.025 (worse), prod -0.007. Dead end.
- Wait neuron cols are only ~0.45 cos with R_wait; the rest is other content.
- P06 (061034_804) u=R_wait token dir, cols=None, L17/18/20/22, a=3: rep -0.066 but ONLY cases 6-11 (-0.08..-0.17), cases 0-5 ~0;
  prod +0.015, close -0.246 (less closing). => two different things: neuron non-logit content drives 0-5; Wait logit drives 6-11.
- P07 (061211_585) Wait neurons, negate only part orthogonal to Wait unembed dirs, a=2: rep -0.017 (0-5 only), prod +0.005.
  => the Wait-token-aligned part of these neurons is what matters for 0-5 (when gated by these neurons' firing).
## Gen checks
(G1 launched on P03 060500_570 at ~06:13)
- Full scan (z>9) saved via d_scan.py: also L16n998 writes -Wait (17.7). Many neurons for But/So/same/again/Alternatively/final/answer.
  Note SwiGLU act sign unknown w/o forward pass -> '+'/'-' only says column orientation.
- Copy-head OV analysis (embedding->OV->unembed self-rank<10): copy heads L3h0,h3; L7h1,h2; L11h1,h2,h3,h4,h7; L15h1,h2.
  Rank-1 edits can't ablate a head; parked.
- G1 on P03 (Wait neurons negated a=2): CATASTROPHE. acc 0/16, cut 1.0, tails = " wait wait wait ...".
  base_acc 0.0625, base_cut 1.0 (base never finishes these 16 within 6144!). losses 1.
  => These neurons normally fire with NEGATIVE activation in thinking context (they suppress 'wait');
     negating them promotes 'wait'. branch_rep was fooled (mass moved to 'wait' everywhere).
  => Lesson: rep proxy rewards any mass-spreading. Base never closes in 6144 -> key lever is finishing/conciseness.
  Next: amplify Wait neurons (alpha<0) = NoWait-like suppression; watch 'close'.
- P08 (062454_256) Wait neurons AMPLIFIED a=-1: rep -0.031 (0-5: +0.01; 6-11: -0.06..-0.09), prod -0.0003, close -0.037, neutral -0.0001.
  => cases 6-11 = continuations starting with Wait (suppressed). Consistent with P06 pattern.
- P09 (062637_007) u=R_wait, cols=4 Wait neurons, a=-6 (amplify only their Wait-logit part): rep -0.162 (6-11: -0.21..-0.44; 0-5 ~0),
  prod +0.011, close -0.077, neutral +0.0002. Clean NoWait-like candidate. Concern: close goes down.
- P10 (062815_258) u=R_</think>, cols=None, L22+L23 a=-3: close +4.07 (!), neutral -0.050, prod -0.037. Strong closure knob; scale down a lot.
- P11 (062959_102) Wait-logit x7 on L17n1915,L18n881,L22n573 (u=R_wait,a=-6) + L23 MLP u=R_</think> a=-0.8:
  rep -0.286 (6-11 -0.47..-0.64), prod +0.012, close +0.80, neutral -0.0011. All limits OK. -> G2.
- G2 on P11 (062959_102): acc 0.50 (base 0.0625), cut 0.3125 (base 1.0), closed_think 0.75, mean_tokens 3686, rescues 7, losses 0.
  Cut tails: no verbatim loops; long deliberation over interpretations / re-reading question; one already wrote
  "*Conclusion:* The answer is 100." then kept "Drafting the response" inside think. => more closure could help.
- P12 (064104_047) same with L23 close a=-1.5: rep -0.286, prod +0.012, close +1.57 (over soft +1), neutral -0.0068. (scorer queue ~10 min now)
- G3 on P12 (064104_047): acc 0.4375, cut 0.0, closed 1.0, mean_tokens 1110, rescues 6, losses 0.
  Correct sets differ from G2 (G2 0,2,3,7,8,10,13,15; G3 3,4,8,9,10,11,13) -> n=16 noisy.
  Stronger closure = much shorter thinking, no acc gain. For 12k final cap prefer milder closure (P11).
- P13 (065623_057) close-only L23 a=-0.8: rep -0.003, prod +0.0007, close +0.84, neutral -0.0016.
  => P11's rep -0.286 is superadditive (NoWait alone -0.162): with Wait suppressed, closing becomes the alternative.
  G4 = P13 for attribution (is NoWait needed in generation?).
- G4 on P13 close-only (065623_057): acc 0.1875, cut 0.81, closed 0.5, mean_tokens 5493, rescues 2, losses 0.
  Tails: verbatim "Wait, if I add \$20, I have \$120." loop; alternating "2 weeks/3 weeks" bullets; re-checking.
  => closure alone does NOT break loops. NoWait is essential; P11 combination is synergistic.
  G5 = P09 (NoWait-only, 4 neurons) to complete attribution.
- G5 on P09 NoWait-only (062637_007): acc 0.4375, cut 0.50, closed 0.5, mean_tokens 4775, rescues 6, losses 0.
  Tails: verbatim bullet loops WITHOUT Wait ("*   What if '1 apple a day' means ..." x N; "*   If the height of each book is 2 inches..." x N).
  => NoWait does most of the work (0.06->0.44); closure adds finishing (cut .5->.31, acc .44->.50 in P11). Synergy confirmed.
  Remaining failure: Wait-free verbatim line loops (bullets). 1 gen left.
## Search for an anti-variant-loop edit (target per_case 0-5, which NoWait doesn't touch)
- P14 (072937_213) copy-head top singular dir reversed (L7/11/15, a=2): rep -0.012, prod -0.024, neutral -0.017. Non-selective; dead.
- P15 (073104_100) Let-neurons amplified a=-6: rep -0.009, prod -0.019; nothing selective.
- P16 (073237_433) So-neurons (L23n489,1516,1714; L22n1065; L19n2038; L18n2423), So-component amplified a=-6:
  prod -0.255 (!), rep -0.019 (0-5 -0.02..-0.05), close -0.20. => productive continuations hinge on 'So'; these neurons'
  So-component net-suppresses 'So' in context (like Wait neurons fire negatively). Reverse (a>1) should PROMOTE 'So' = forward progress.
- P17 (074346_174) So-component reversed a=2.5: prod -0.062, rep -0.05. So-lever disruptive both ways; dropped.
- P18 (074537_659) Wait-logit a=-9 (L17,18,22) + L23 close a=-1.1: rep -0.428 (6-11 -0.66..-0.96; 0-5 -0.01..-0.06), prod +0.017,
  close +1.13, neutral -0.0029. G6 = P18 (stronger NoWait, closure between P11 and P12).
- G6 on P18 (074537_659): acc 0.6875 (11/16), cut 0.125, closed_think 1.0, mean_tokens 2036, rescues 10, losses 0. BEST.
  Cut tails: both after </think> (in the answer): one growing-number-sequence loop ("Week 207: ... 208 weeks" etc.,
  the variant loop the brief describes) and one long final answer.

## Summary of gens (16 GSM8K train problems, cap 6144; base acc 0.0625, cut 1.0)
| cand | edit | acc | cut | tokens |
| P03 060500_570 | Wait neurons negated | 0.00 | 1.00 | 6144 ("wait wait ...") |
| P13 065623_057 | close boost only (L23 a=-0.8) | 0.19 | 0.81 | 5493 |
| P09 062637_007 | NoWait only (4 neurons a=-6) | 0.44 | 0.50 | 4775 |
| P12 064104_047 | NoWait(3,a=-6)+close a=-1.5 | 0.44 | 0.00 | 1110 |
| P11 062959_102 | NoWait(3,a=-6)+close a=-0.8 | 0.50 | 0.31 | 3686 |
| P18 074537_659 | NoWait(3,a=-9)+close a=-1.1 | 0.69 | 0.125 | 2036 |
Final ranking: P18 > P11 > P12.
