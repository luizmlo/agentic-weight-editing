# r2 notebook

Budget: 40 probe, 6 gen. Start 2026-11-05.

## Setup
- atlas: natural[sig][loop|preloop] [18,24,dim]; branch[sys][sig][rep|prod] [12,24,dim]; sigs resid/mlp_in/attn_in.
- branch cases: (question, prefix, productive, recheck-sentence). rep continuation = re-check sentence(s).

## Atlas findings (before probes)
- Residual loop diff (nat loop-preloop) and branch (rep-prod) share direction: cos ~0.5-0.6 at L8-21.
- Exact linear decomposition of mean diffs: resid diff = sum_L W_o attn_in diff + W_down mlp_in diff (verified cos=1.000).
- Loop diff written in a distributed way; biggest along-dir contributors L15 attn+mlp, L19, L21-23.
- Per-head: L16 DeltaNet h6 writes loop dir in both families (cos nat/br .98), L15 full-attn h6/h7, L14 h7, L22 h7, L11 h1/h2, L17 h6.
- Many MLP neurons loop-selective in both families (t>3 both): ~150-200 per mid layer.
- close-token direction: loop diffs barely touch it; close family looks weak.
- Final-layer nat loop diff logit-lens: '重复'(repeat), '纠正', '不同' up -> model internally "knows" it's repeating.

## Screening summary (alpha in parens)
- P01 L16 lin h6 (1): rep -.022 prod +.000 close -.02 neu +.004
- P02 L15 self h6+7 (2): rep -.058 prod +.006 close +.14 neu +.009  <- good
- P03 L22 lin h7 (2): ~0
- P04 L14 lin h7 (2): rep -.091 prod +.003 close -.01 neu +.001  <- best selective
- P05 L23 mlp top64 loop neurons (2): rep -.013 prod -.037 close -.23  <- bad (MLP neuron family hurts prod)
- P06 L19 mlp top64 (2): ~0
- P07 L16 lin h6 (3): rep -.084 prod -.010 neu +.003 (superlinear vs alpha1)
=> mid-layer (L14-16) attention/DeltaNet heads carrying loop signal are the lever; late MLP loop neurons are not.
- P08 L13 lin h1,2 (2): rep -.010 ; P10 L17 lin h6 (2): -.010 ; P11 L12 lin h9 (2): -.008  (weak)
- P09 L11 self h1,2 (2): rep -.117 prod +.004 close +.009 neu +.004  <- BEST single
=> lever = full-attn heads L11 h1/h2, L15 h6/h7 + DeltaNet L14 h7, L16 h6. Directions mutually ~orthogonal (cos<.2) so should add.
- C1 063054_472: rep -.428 prod +.003 close +.097 neu +.012 (superadditive). -> gen check G1
- Robustness (a11): each chosen head's projection on its loop dir is higher in loop than preloop in 18/18 natural texts; br rep-prod gap identical under all 4 system prompts (prompts don't touch this circuit -> consistent with prompt fixes failing).
- (housekeeping) fixed relative-path bug: per-probe autolog now in work/autolog.md, cand_*.pt in work/

## G1 (gen of C1 063054_472): acc 0/16, cut 1.0, closed_think 0, base_acc .0625 base_cut 1.0 (!), losses 1
- Gen set (GSM8K train[200:216], cap 6144): BASE NEVER CLOSES THINK either. The problem at 6k is over-deliberation, not just verbatim loops.
- C1 tails: huge whitespace indentation runs + nested "Counter-argument for a different interpretation" rumination. Verbatim suppression alone does nothing for termination.
- => pivot: need edits that make the model CONCLUDE (close </think>) / ruminate less. Probe 'close' is allowed up to +1.
- P12 065123_487 L23 attn+mlp amplify u_close a=-1.5: close +3.79 (!), neutral -.125, rep -.063 prod -.059. close is very sensitive; generic boost costs neutral.
- P13 065307_760 L23 mlp 32 loop-conditional close neurons, amplify u_close a=-6: close +.73, neu -.014, rep -.049, prod +.004.
- P14 065714_050 same a=-10: close +1.22, neu -.033, rep -.143, prod +.001. -> G2 gen check (is termination the lever?)

## G2 (gen of P14 065714_050, close-boost only): acc 7/16 (base 1/16), cut .44, closed_think .875, mean_tokens 4274, rescues 6, losses 0  <<< BIG
- Termination is THE lever on this set. Remaining failures are hard verbatim token loops ("3 3 3 3", "28... 28...", "1000 * 1.3 = 1300." repeated) -- probably after/at the answer. These are exactly what the C1 heads target -> combine.
- C2 071841_423 heads(L11 h1,2/L14 h7/L15 h6,7 a2)+close32 a-10: rep -.459 prod +.010 close +1.26 neu -.019 (over limits)
- C3 072949_822 same heads + close32 a-6: rep -.416 prod +.015 close +.81 neu -.0006  <- passes all probe limits -> G3

## G3 (gen of C3 072949_822): acc 6/16, cut .625, closed .4375, mean_tok 5008, rescues 5, losses 0
- Tails now = interpretive rumination inside think (no verbatim loops). Close strength is the main driver (P14 a-10 alone: closed .875).
- Next: make close boost more selective for LONG natural reasoning (fires in natural traces, ~0 in short branch ctx) so it can be stronger in gen while keeping close-probe<=1 & neutral>=-.01.
- P15 074324_612 L23 'long-context' close set (32) a=-6: close +2.12 (!), neu -.008, prod -.038, rep -.064. Branch-mean selectivity didn't transfer to the close probe position. Old (loop-conditional) set is better: per unit close-probe it gives more boost in loop contexts.
- a16: neurons with loop>pre but low pre-level have tiny </think> writes -> mean-level loop-conditionality is weak; keep old set and tune alpha to close~1.
- C4 074511_641 heads + close32 a-7.5: rep -.431 prod +.014 close +.979 neu -.0064 (compliant, max close) -> G4

## G4 (gen of C4 074511_641): acc 5/16, cut .625, closed .4375, rescues 4, losses 0
- Same as C3 despite stronger close. P14 (close-only, a-10) had closed .875. Hypothesis: the loop-copy heads REMOVE the loop/recheck state that drives the loop-conditional close neurons at L23 -> heads fight the termination mechanism in generation (probe can't see this). Drop the big heads.
- Neutral fixers with little loop effect: L17 lin h6 (P10 neu +.010, rep -.010), L13 lin h1,2 (P08 neu +.006, rep -.010).
- C5 075757_917 close32 a-8.5 + L17 h6 + L13 h1,2 (a2): rep -.107 prod +.011 close +.917 neu -.0011 (compliant) -> G5

## G5 (gen of C5 075757_917): acc 2/16, cut .75, closed .5625, rescues 2, losses 1  -- worse
- Tails: hard verbatim loops ("2 weeks = 3 apples." x N, "16 10 3 8 5" cycles). Without the big loop-copy heads, verbatim loops dominate.
- Revised picture: close strength drives closure (P14 close-probe 1.22 -> closed .875; configs at .8-1.0 -> closed .44-.56);
  the loop-copy heads (L11/L14/L15) don't block closing much but convert verbatim loops into (still cut) rumination and give the accuracy edge (C3 6/16, C4 5/16 vs C5 2/16).
- Last gen -> C2 (heads + close a-10; probe close 1.26, neu -.019, slightly past soft limits): tests strong close + anti-verbatim together.

## G6 (gen of C2 071841_423): acc 5/16, cut .5625, closed .6875, rescues 4, losses 0
- Tails: interpretive rumination ("Interpretation 1/2", "Wait, ... is ambiguous"), no verbatim loops.

## Gen table (16 GSM8K train problems, cap 6144; base 1/16, cut 1.0)
| cand | edits | probe rep/prod/close/neu | acc | closed | resc/loss |
| C1 063054_472 | 4 loop heads | -.43/+.003/+.10/+.012 | 0 | 0 | 0/1 |
| P14 065714_050 | close32 a-10 | -.14/+.001/+1.22/-.033 | 7 | .875 | 6/0 |
| C3 072949_822 | 3 heads + close32 a-6 | -.42/+.015/+.81/-.0006 | 6 | .44 | 5/0 |
| C4 074511_641 | 3 heads + close32 a-7.5 | -.43/+.014/+.98/-.006 | 5 | .44 | 4/0 |
| C5 075757_917 | close32 a-8.5 + L17h6 + L13h1,2 | -.11/+.011/+.92/-.001 | 2 | .56 | 2/1 |
| C2 071841_423 | 3 heads + close32 a-10 | -.46/+.010/+1.26/-.019 | 5 | .69 | 4/0 |
Conclusions: the active ingredient is the loop-conditional </think> boost at L23 MLP (all close-boost candidates beat base; heads-only = 0).
Loop-copy heads remove verbatim loops (tails become rumination) and make results consistent (5-6/16, 0 losses), at some cost in closure.
Close-only is fragile (P14 7/16 but nearest neighbour C5 2/16, verbatim loops, 1 loss).
Final ranking: C3 > C4 > C2 (heads+close family, 0 losses in all 3; C3/C4 meet all probe soft limits). P14 is the single best gen but violates neutral (-.033) and close (+1.22).
Budget used: 20/40 probe, 6/6 gen.
