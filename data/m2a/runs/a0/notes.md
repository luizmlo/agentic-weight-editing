# a0 notes (no-atlas control arm, budget 60)

Sources: weights (CPU), tokenizer, scorer only. Final norm is (1+w) parametrized (w mean 3.3).
Token-direction u = (1+g_final) ⊙ (E[tok] - mean(E))  ("unembedding direction" of tok in residual).

## Weight analysis
- OV copy scores (direct token->logit through head OV, 3000 random word tokens; fraction of rows where diagonal in top-10):
  full-attn heads strongly copying: L3h0, L7h2, L11h4, L11h7 (1.0), L7h1, L15h1, L11h3 (~0.99), L3h3, L11h1, L15h2, L11h2 (0.75-0.87).
  DeltaNet heads: weak copying (max 0.18 at L22h12).

## Submissions
1. c01 L23 MLP, u=Wait/ Wait tokdir, all cols, a=1: rep -0.0233, prod +0.0209, close +0.36, neutral +0.0047.
   per-case rep: cases 0-5 ~0/positive, 6-11 strongly negative (-0.04..-0.07). -> token-level "Wait" suppression already works.
2. c02 Wait L23 a=4: rep -0.245 but prod -0.097, close 1.35, neutral -0.139 -> too strong, nonlinear.
3. c03 Wait L23 a=2: rep -0.078, prod +0.012, close 0.70, neutral -0.017 (neutral slightly over).
4. c04 Wait L23 top-512 Wait-promoting cols a=3: rep -0.010 only -> effect is distributed, not in Wait-promoting neurons.
5. c05 Wait MLP20-23 a=1 each: rep -0.036, prod +0.026, close 0.34, neutral +0.005 (all ok). L23 dominant.
6. c06 Yes(" Yes","Yes") MLP20-23 a=1: rep -0.079, prod +0.009, close 0.74, neutral +0.001. Hits cases 0,2,3,4 (complements Wait which hits 6-11).
7. c07 Yes@22,23 + Wait@22,23 a=1: rep -0.056, close 0.88, neutral -0.011. Worse than Yes alone at 4 layers.
8. c08 recheck vocab (check,verify,confirm,double,correct) MLP20-23 a=1: rep -0.069 but prod -0.135, neutral -0.072, close 1.14. BAD: non-selective.
9. c09 </think> dir MLP20-23 a=1: close +0.15, rep +0.019 -> MLPs write -</think> in prefix; amplifying (alpha<0) could lower close.
10. c10 Yes L23 MLP a=2 only: rep -0.040, prod -0.020, close 1.29, neutral -0.031. bad; L23 alone not the locus.
11. c11 Yes MLP16-19 a=1: rep -0.017, all side effects ~0. weak.
12. c12 Yes dir from ATTENTION outputs (self_attn 15,19,23 + linear_attn 22) a=1: rep -0.043, prod -0.003, close 0.18, neutral +0.003. Hits all cases evenly.
13. c13 same a=2: rep -0.230 (!), prod -0.009, close 0.40, neutral +0.0007. Very selective. Sign flip of attention's write along Yes-dir = anti-copy.
14. c14 Yes dir attn (sa15,19,23 + lin22) a=3: rep -0.515, prod -0.018, close 0.68, neutral -0.006. Cases 1,5 barely move (-0.04,-0.05) -> those rep texts likely lack Yes.
15. c15 Yes a=3 sa23+lin22: rep -0.340, prod -0.020, close 0.86, neutral -0.015.
16. c16 Yes a=3 sa15+sa19: rep -0.012, close -0.12. Small alone, but synergizes (c14 vs c15) and lowers close.
17. c17 Yes a=3 sa23 alone: rep -0.275, prod -0.017, close 0.74, neutral -0.0096. L23 full attention is the main locus.
   Weight-only per-head analysis (u^T W_O per head, OV self-copy of Yes) shows no clear single head in L23.
18. c18 Wait dir sa23 a=3: rep -0.297, prod -0.005, close -0.54 (!), neutral -0.004. Cases 6-11 ~-0.5, cases 0-5 -0.04..-0.08 (incl. 1,5). Close goes DOWN -> complements Yes.
19. c19 sa23 Yes a=3 + sa23 (Wait orth Yes) a=3 + sa15,sa19 Yes a=3: rep -0.598, prod -0.026 (slightly over), close -0.13, neutral -0.0078. Weakest cases 1,5 (-0.10).
20. c20 sa23 Yes a=2.5 + Wait-orth a=3.5 + sa15/19 Yes a=3: rep -0.533, prod -0.022, close -0.34, neutral -0.0064.
21-23. Head attribution at sa23 (Yes a=3): heads0-3 -0.031, heads4-5 -0.017, heads6-7 -0.046; whole layer -0.275 -> strongly superlinear/collective; head restriction not useful.
24. c24 '?' dir sa23 a=3: rep -0.423 broad (cases 1,5 strongly too) but prod -0.035, close +1.45, neutral -0.044. Too non-selective.
25. c25 sa23 Yes3 + Wait3 + ?-orth 1.75, sa19 Yes3: rep -0.49, prod -0.045, neutral -0.025. '?' ruins selectivity. Drop it.
26. c26 sa23 Yes 2.5 + sa23 Wait-orth 3 + sa15/sa19 Yes 4: rep -0.572, prod -0.018, close -0.35, neutral -0.0036. ALL OK.
27. c27 same with sa15/19 a=6: rep -0.727, prod -0.018, close -0.49, neutral -0.0057. ALL OK. (best feasible so far)
28. c28 sa15/19 a=9: rep -0.927, prod -0.018, close -0.69, neutral -0.018 (over). mid-layer Yes reversal is cheap on prod, costs neutral at high a.
29. c29 sa23 Yes2.5/Wait3, sa15 Yes 10, sa19 Yes 6: rep -0.976, prod -0.0209, close -0.53, neutral -0.0086. sa15 is a cheap strong lever; sa19 high a drives neutral.
30. c30 sa23 Yes2/Wait3, sa15 Yes10, sa11 Yes10: rep -0.997, prod -0.023, close -0.34, neutral -0.004. prod slightly over.
31. c31 sa15 Yes10, sa11 Yes10, sa19 Yes4, sa23 Wait3 (no sa23 Yes): rep -0.745, prod -0.0098, close -0.85, neutral -0.0064. sa23-Yes is the prod cost AND what kept close up.
32. c32 sa15 Yes10 + sa15 Wait-orth10, sa11 Yes10, sa23 Yes2: rep -0.905, prod -0.022, close +0.07, neutral -0.016 (over). Mid-layer Wait costs neutral.
33. c33 sa15/sa11/sa7 Yes10 + sa23 Wait3: rep -0.691, prod -0.030, neutral -0.017. sa7 harmful -> drop.
34. c34 sa15 Yes10, sa11 Yes10, sa23 Wait3, mlp23 Yes a=1: rep -0.912, prod -0.0018 (!), close +0.61, neutral -0.021 (over). MLP23 Yes swings close +1.46 and fixes prod but costs neutral.
35. c35 sa15 Yes10, sa11 Yes10, sa23 Wait3, mlp23 Yes0.5: rep -0.812, prod +0.005, close -0.03, neutral -0.003. CLEAN on all side metrics. best balanced.
36. c36 Wait4 + mlp23 Yes0.75: rep -1.011, prod -0.003, close +0.39, neutral -0.024 (over).
37. c37 Wait4 + mlp23 Yes0.5: rep -0.957, prod -0.002, close +0.01, neutral -0.0139 (slightly over). Wait 3->4 costs ~-0.011 neutral.
38. c38 '?' dir sa15 a=10 alone: rep -0.262 BROAD (case1 -0.44, case5 -0.25), prod -0.003, close +0.12, neutral -0.004. Mid-layer '?' is cheap (vs sa23 '?' which was damaging).
39. c39 sa15 Yes10 + sa15 (?⊥Yes)10 + sa23 Wait3 + mlp23 Yes0.5: rep -1.100, prod +0.005, close -0.02, neutral -0.0058. ALL CLEAN, broad (all 12 cases <= -0.40). NEW BEST.
40. c40 c39 with Wait3.5: rep -1.181, prod +0.002, close -0.01, neutral -0.0104 (at limit).
41. c41 '?' moved to sa11: rep -0.664 -> '?' at sa11 does ~nothing; sa15 is the locus for '?'.
42. c42 'So' at sa15 a=10: ~0 effect.
43. c43 Wait at sa15 a=10: rep -0.048, prod -0.010, neutral -0.015. Bad: Wait belongs at sa23.
44. c44 c39 with Wait3.25: rep -1.141, prod +0.003, close -0.02, neutral -0.0079. CLEAN. best feasible.
45. c45 sa15 Yes10 + ?⊥10, sa23 Yes2.5 + Wait⊥ 3.25: rep -1.369, prod -0.029 (over), close -0.23, neutral -0.011 (over).
46. c46 sa15 Yes10 + ?⊥10, sa23 Yes1.75 + Wait⊥ 3.0: rep -1.191, prod -0.018, close -0.39, neutral -0.0053. feasible; slightly more rep than c44 but prod near limit, close more negative.
47. c47 c44 with Wait u incl ' wait': rep -1.086, close +0.18. worse than c44.
48. c48 sa15 single combined u=norm(Yes+?⊥) a=10 + sa11 Yes10 + sa23 Wait3.25 + mlp23 Yes0.5: rep -0.591. Combined direction much worse; tokens need separate edits.

## Conclusions
- Mechanism: reversing (alpha>=2) the component that FULL-ATTENTION layers write along a loop token's unembedding
  direction converts copy-from-context into anti-copy for that token. Strongly superlinear in alpha (a=1 -> -0.04, a=2 -> -0.23, a=3 -> -0.52).
- Loci: Yes and '?' at sa15 (alpha 10, cheap on prod/neutral); sa11 Yes also helps; Wait at sa23 (a~3, lowers close);
  sa23 Yes strong but costs prod and raises close; sa19 Yes at high alpha costs neutral; sa7 harmful; sa11 '?' useless; sa15 Wait useless+costly.
- MLP token removal (alpha~1) is weaker; MLP23 Yes a=0.5 is a handy close balancer (+~0.8 close) with ~0 prod cost.
- Failures: recheck vocabulary (check/verify/...) MLP edit (non-selective), '?' at sa23 (damaging), 'So' (null), head-restricted cols (effect is collective), combined multi-token u in one edit.
- Final ranking: 201707_653 (c44) > 202036_894 (c46) > 200817_596 (c39). Used 48/60 evals.
