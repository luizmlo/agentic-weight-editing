# r1 notebook (M->A weight editing, budget 60)

## Setup
- branch_cases tuples = (question, prefix, productive cont., repetitive cont.). Rep cont is a
  "But is X? Yes, X." / "Wait, does ...? Yes, ..." re-check.
- atlas: natural[sig][loop|preloop] [18,24,dim]; branch[sys][sig][rep|prod] [12,24,dim].

## Atlas findings (pre-submit)
- Residual Δ(rep-prod) is exactly reconstructible from per-writer Δ(write)=W·Δinput (means are linear).
- Branch rep direction and natural loop direction agree: cos 0.55-0.63 for resid L8-L17; per-writer Δwrite
  cos(br,nat) highest at L10-11 attn (0.65-0.68), L14 attn (0.63), L22 attn (0.66), L23 mlp (0.72).
- System prompts barely move internals: prompt shift ~5% of |rep-prod Δ|, not aligned with it
  (anti_rep even slightly *increases* rep-prod separation at L19, cos 0.37) -> explains prompt failure.
- Logit lens: natural-loop final Δ promotes '重复'(repetition), '纠正', <|im_end|>; branch Δ promotes Okay/Yes/Ah.
- Edit recipe: u = nrm(nrm(W Δa_br)+nrm(W Δa_nat)) for writer W at layer L; cols top-k by (u·W_j)(Δa_br+2Δa_nat); alpha.

## Results log (sel = rep - prod; lower better)
- 183028_226 L23 mlp top64 a=2: rep -0.014 prod -0.051 neutral -0.029 -> BAD (late layer = token-level, hurts all)
- 183208_828 L14 lin-attn all a=2: rep -0.127 prod -0.005 close -0.05 neutral -0.001 -> GOOD, all 12 cases
- 183453_472 L11 full-attn all a=2: rep -0.032 prod -0.001 close +0.07 neutral -0.0005 (weak)
- 183722_484 L14 mlp all a=2: rep -0.089 prod -0.001 close -0.13 neutral -0.002 (good)
- 184001_334 L15 mlp all a=2: rep -0.039 prod -0.001 close +0.06 neutral +0.009 (weak but clean)
- per-writer Δwrite directions are ~orthogonal across layers (|cos|<0.2) -> edits should compose additively.
- 184239_086 L14 lin-attn all a=4: rep -0.424 prod -0.011 close -0.09 neutral -0.0045 (superlinear in alpha)
- 184512_748 L14 lin-attn all a=6: rep -0.928 prod -0.017 close -0.14 neutral -0.0097 (neutral at limit)
- 184807_665 L14 lin-attn top128 a=6: rep -0.968 prod -0.005 close -0.02 neutral -0.0032 (column restriction: same rep, 3x less neutral cost)
- 185027_886 L14 lin-attn top128 a=10: rep -2.61 prod -0.011 close -0.04 neutral -0.018 (neutral violated; neutral cost superquadratic in alpha)
- 185308_215 L14 lin-attn top64 a=10: rep -1.88 prod -0.010 close -0.08 neutral -0.0116 (slightly over)
- 185548_307 L14 mlp top256 a=6: rep -0.226 prod -0.004 close -0.34 neutral -0.013 (MLP much worse ratio than lin-attn)
- 185821_912 L10 lin-attn top128 a=6: rep -0.596 prod +0.001 close +0.23 neutral -0.0068
- 190056_673 L13 lin-attn top128 a=6: rep -0.397 prod +0.009 close -0.03 neutral +0.009 (neutral improves!)
- 190336_943 combo L14+L13+L10 attn k128 a6: rep -0.807 prod -0.004 close +0.20 neutral -0.0087 -> SUBADDITIVE (worse than L14 alone); same pathway, interfering
- 190635_697 L14 per-head dirs (h14,7,9,12) full head cols a=5: rep -0.869 prod +0.004 close +0.09 neutral -0.0135 -> worse ratio than global-u top128
- 190854_363 L14 lin-attn top128 a=8: rep -1.742 prod -0.008 close -0.025 neutral -0.0094 -> BEST within limits so far
- 191133_890 L14 a8 + L13 a6: rep -0.517 (!!) neutral -0.0095 -> upstream edit DESTROYS L14 effect. Only add edits downstream of L14 (they cannot change L14's inputs).
- 191408_739 L16 lin-attn k128 a6: rep -0.140 prod -0.011 neutral +0.001 (weak)
- 191656_966 L14 k128 a8 + L15 mlp all a3: rep -1.760 prod -0.009 close -0.001 neutral -0.0044 -> L15 mlp is a downstream NEUTRAL COMPENSATOR (+0.005), no rep loss. NEW BEST
- 191917_830 L14 k128 a10 + L15 mlp a5: rep -2.708 prod -0.015 neutral -0.027 -> compensator non-monotonic (a3 helps, a5 hurts)
- 192156_139 L14 k128 a9 + L15 mlp all a3: rep -2.213 prod -0.011 close -0.024 neutral -0.0092 -> NEW BEST within limits
- 192455_907 L14 dir+cols from NATURAL loops only, k128 a8: rep -0.822 prod -0.006 close +0.004 neutral -0.0088 -> natural-loop direction transfers to branch metric (mechanism shared, not branch overfit); combined dir 2x stronger on branch
- 192714_283 best + L16 attn k128 a4: rep -2.260 prod -0.018 neutral -0.0126 -> L16 hurts, discard
- 193002_046 best + L19 mlp all a2: rep -2.211 prod -0.010 close -0.09 neutral -0.0167 -> L19 mlp hurts neutral, discard
- 193233_057 L15 self_attn (full attn) k128 a6: rep -0.504 prod -0.018 close +0.25 neutral -0.059 -> full attention o_proj is essential for generic text: AVOID
- 193524_428 L14 cols=top160 by per-case t-stat a9 + L15mlp a3: rep -2.048 prod -0.009 close +0.11 neutral -0.0097 -> no better than plain top128 score
- 193752_827 best + upstream L10 attn k128 a4: rep -0.935 (!!) neutral -0.0125 -> confirms upstream levers kill L14 reversal (L10->L13->L14 is one pathway)
- 194020_908 best + L14 head15 own dir a6: rep -2.158 prod -0.008 neutral -0.0099 -> no gain
- 194253_891 L12 lin-attn k128 a6: rep -0.683 prod +0.004 close -0.19 neutral +0.0147 (!) -> excellent side-effect profile
- 194529_320 L12 lin-attn k128 a10: rep -1.521 prod +0.002 close -0.25 neutral +0.011 -> CLEANEST strong lever
- 194812_975 L12 a10 + L14 a6: rep -1.386 neutral -0.0004 -> interference again (levers along one pathway don't stack)
- 195044_322 L12 lin-attn k256 a10: rep -1.906 prod -0.005 close -0.23 neutral +0.0097 -> strong AND clean
- 195319_943 L12 lin-attn k512 a10: rep -2.176 prod -0.020 close -0.15 neutral +0.005 -> prod at soft limit
- Interference hypothesis: upstream reversal already flips L14's rep component to anti-rep; reversing again at L14 flips it back. Test: L12 reversal + L14 AMPLIFY (alpha<0).
- 195605_902 L12 k256 a10 + L14 k128 a=-3 (amplify): rep -1.998 prod -0.004 close -0.20 neutral +0.009 -> small gain vs L12 alone (-1.906): supports flip hypothesis
- 195839_814 L12 k256 a10 + L14 k128 a=-8: rep -2.147 prod -0.005 close -0.13 neutral -0.0021 -> NEW BEST (cleaner than L14-based best)
- 200114_879 L12 a10 + L13 a-6 + L14 a-8: rep -2.054 prod -0.061 neutral -0.015 -> amplifying L13 hurts; discard
- 200351_163 L12 k384 a10 + L14 a-8: rep -2.357 prod -0.017 close -0.14 neutral -0.0056 -> stronger but prod near soft limit
- 200632_845 L12 dir+cols from NATURAL only k256 a10: rep -0.373 prod -0.040 -> L12 lever does NOT transfer from natural loops (L12 br/nat cos only 0.38). L14 nat-only kept ~47% of effect, L12 only ~19%. Overfitting concern: branch Δ used in direction = same 12 cases as metric. Next: cross-validate on held-out cases.
- 200915_614 CV L14 k128 a8 fit on even br cases: rep -1.697 (fit -1.74, HELD-OUT -1.66) prod -0.007 neutral -0.0098 -> L14 lever generalizes across cases
- 201050_229 CV L12 k256 a10 fit on even br cases: rep -1.947 (fit -2.04, HELD-OUT -1.86) prod -0.002 neutral +0.011 -> generalizes across branch cases too (but same template; nat transfer weak)
- RISK note: in 195839_814 (L12 rev + L14 amplify), L14 amplification may be pro-loop on NATURAL loops if L12 reversal doesn't flip L14 there (L12 weakly tied to natural loops). Down-weight.
- 201251_186 L14 k96 a10 + L15 mlp a3: rep -2.328 prod -0.014 neutral -0.0114 -> over neutral limit
- 201429_745 L12 k256 a10 + L14 k128 a=1 (removal): rep -1.875 prod -0.005 close -0.24 neutral +0.009 -> clean hedge
- 201611_736 L12 k256 a10 + L14 NAT-only dir a8: rep -1.748 prod -0.013 neutral -0.006 -> interference again: L12 reversal also flips L14's natural-loop component
- atlas check: L12(k256) and L14(k128) edit projections rise in 18/18 natural loop traces (nat/br Δ ratio 0.34 and 0.43; natural Δs are ~half of branch overall) -> L12 lever does engage in natural loops
- 201814_307 L12 k320 a10 + L14 k128 a-8: rep -2.279 prod -0.010 close -0.15 neutral -0.0040 -> NEW BEST balance
- 201939_974 L12 k320 a10 ALONE: rep -2.031 prod -0.007 close -0.23 neutral +0.008 -> best single rank-1 edit, very clean
- 202135_739 L14 k128 a=-8 ALONE (amplify): rep -0.349 prod -0.001 close +0.06 neutral -0.025 -> amplifying ALSO lowers rep (non-directional 'disruption' component, ~20% of reversal effect); reversal is 5x stronger -> mostly directional. Neutral cost of amplification vanishes when L12 reversal is upstream.
- 202318_101 CONTROL random u, L12 same 320 cols a10: all metrics ~0 -> L12 effect is directional.

## Conclusions
- Mechanism: a "re-check/repetition state" carried by mid-depth Gated-DeltaNet (linear attention) heads,
  L10->L12->L13->L14 forming one pathway (L14 heads 14,7,9,12,11 dominant). Its writer-specific Δ direction is
  shared by branch re-check text and natural loops (fires in 18/18 natural traces). Reversing (alpha>1)
  that component on the most rep-selective columns sharply lowers logp of repetitive continuations
  while leaving productive text, </think> closing and WikiText nearly unchanged.
- Reversals along the pathway do NOT stack: an upstream reversal flips the downstream component, so a
  second reversal flips it back (L13/L10/L12 + L14 all collapse). Downstream amplification (alpha<0) adds a little.
- Late-layer (L23 MLP) and full-attention (L15 o_proj) edits hurt generic text; MLP levers have worse ratios.
- L15 MLP reversal at small alpha (a=3) is a downstream neutral compensator for the L14 lever.
- Column restriction to top-k rep-contributing columns is key for neutral cost (L14: all-cols a6 neutral -0.0097 vs top128 -0.0032).

## Final ranking
1. 201939_974 L12 lin-attn k320 a10 (single rank-1): rep -2.031 prod -0.007 close -0.23 neutral +0.008
2. 201814_307 L12 k320 a10 + L14 k128 a-8: rep -2.279 prod -0.010 close -0.15 neutral -0.004
3. 192156_139 L14 k128 a9 + L15 mlp a3: rep -2.213 prod -0.011 close -0.024 neutral -0.0092
