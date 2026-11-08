# a2 notes (no-mapping control arm)
Start 01:14. Sources: weights (CPU, no forward), tokenizer, scorer. No atlas.

## Plan
- Logit-lens directions from tied embeddings + final norm: u = g ⊙ E[token] (residual direction that raises token logit).
- Families: (A) suppress "Wait"/re-check tokens in late MLP writes; (B) amplify </think> / answer closing; (C) combos.

## Session 2 (restart 01:20) — log in probe_log.md
- norm is (1+w); logit-lens dir for token t = (1+w_final) ⊙ E[t]. Cached in work/tokdirs.pt, layercos.pt.
- Single 'Wait' neurons (cos~0.5 with wait dir): L16n998(-), L17n1915(+), L18n881(+), L22n573(+), L20n405(0.34).
- Yes neurons: L17n90(-0.64), L23n3308(0.57), L17lin1800(0.51), L13n1043. Actually: L16n955, L17n349, L21lin1878, L22n787.
- </think>: no strong neurons; L22lin n1692 (0.36).
- P01 wait neurons a=1: tiny. P02 a=8: rep -0.23, prod +0.03, close -0.17, neutral -0.005 (good profile).
- P03 wait dir all cols L16/17/18/22 a=1: weak (-0.018).
- P04 </think> amplify a=-3 on 4 late tensors: close +8.7 (!), neutral -0.06. Way too much; very sensitive.
- P05 </think> amplify a=-0.4 same 4 tensors: close +1.00, everything else ~0. Pure exit booster (id 012857_136).
- P06 Yes neurons a=8 (L17n90,686; L23n3308,820; L17lin1800,1871; L13n1043): rep -0.20 (uniform over cases), prod -0.004, close -0.07, neutral -0.003.
- Wait neurons hit only cases 6-11; Yes hits all 12 cases.
- P07 Wait a=10 (+L16n3422) submitted 01:32, scorer queue busy (other run gen?).
- P07 Wait a=10 (013230_387): rep -0.33, prod +0.036, close -0.20, neutral -0.010 (borderline). GEN1 on it.
- **GEN1 (P07 Wait a=10)**: acc 0.50 vs base 0.0625; cut 0.50 vs base 1.00; 7 rescues / 0 losses; mean_tokens 5008.
  Cut tails: two are long structured case analysis (not verbatim loops), one is a degenerate "Total wait for a plus one = 1 wait." loop (lowercase 'wait' noun leak).
  => Wait suppression is a real generation win. Next: combine with </think> boost or Yes suppression (4-edit limit).
- P08 Wait3 (no L18) + L22lin think a=-1.5 (015603_194): rep -0.328 (L18 not needed), prod +0.035, close +0.115, neutral -0.008.
- P09 Wait3 + L22lin think a=-4 (015739_897): rep -0.328, prod +0.034, close +0.68, neutral -0.0095. GEN2 on it (tests exit booster on top of Wait).
- **GEN2 (P09 Wait3 + </think> boost)**: acc 0.6875, cut 0.0625, closed 0.94, mean_tokens 2859, 10 rescues / 0 losses. Best so far.
  Only cut tail: alternating 2-line loop ("He practices 5 days a month. / He takes 2 weeks off.") — variation loop w/o Wait.
  4 of 5 failures closed but wrong.
- P10 Wait L17+L22 (no L16) + think + Yes L23 a=8 (020848_691): rep -1.28, prod -0.47 (!), close +1.81 (!). Violates limits; discard.
  L16n998 is the main wait neuron for cases 6-11 (dropping it: case6 -0.78 -> -0.41). L23 Yes edit interacts badly here.
- GEN3 = P05 (think boost alone, close +1.0) as ablation: is Wait suppression needed, or is the exit booster doing the work?
- **GEN3 (P05 think boost alone)**: acc 0.1875, cut 0.81, 2 rescues/0 losses. Tails: "Wait, is ... ?" re-check loops persist.
  => Synergy: Wait-suppression alone 0.50/cut 0.50; think alone 0.19/0.81; both 0.69/0.06. Mechanism = remove the loop's re-entry token + lower the exit threshold.
- P11 Wait3 + think a=-6 (022229_724): close +1.13 (over), neutral -0.011, rep/prod same as P09.
- P12 L16 wait + L17/L22 u=wait+actually on [1915,349]/[573,787] a=10 + think a=-4 (022432_979): rep -0.31, prod +0.033, close +0.90, neutral -0.0086. GEN4 on it.
- **GEN4 (P12, + Actually suppression)**: acc 0.50, cut 0.31, 7 rescues/0 losses — WORSE than GEN2. Tails = variation loops
  (growing "Week 372: 373 weeks..." sequence, verbatim line loop, circular arithmetic). Exactly the brief's failure mode:
  suppressing the self-correction pivot ("Actually") removes the way out of loops. Wait is the loop marker; Actually is an escape.
  Idea: maybe *amplifying* Actually helps (test).
- P13 Wait L16+L22 + think a=-4 (023601_880): identical to P09 on probes -> L17n1915 adds ~nothing on probes.
- P14 Wait L16 ONLY [998,3422] + think a=-4 (023721_576): rep -0.336, prod +0.033, close +0.69, neutral -0.0049 (half). L16n998 carries the whole Wait effect; later Wait neurons likely downstream readers.
- P15 L16 wait + think a=-4 + amplify Actually L17n349,L22n787 a=-3 (023915_620): rep -0.338, prod +0.033, close +0.58, neutral -0.0048. GEN5 on it.
- **GEN5 (P15: L16-only Wait + think -4 + Actually amplify)**: acc 0.5625, cut 0.19, 8 rescues/0 losses. Worse than GEN2.
  Tails: counting loops ("Week 76/77/78"), "step 170/171/172 is correct" verification sequences, circular restatement.
  Either L17/L22 Wait edits matter in generation (despite probes saying no) or Actually amplification hurt; lower close (+0.58) too.
- Pattern so far: cut fraction tracks exit strength among Wait+think variants. Last gen: P09 with think a=-5 (close ~+0.9).
- P16 P09 w/ think a=-5 (030043_342): close +0.90, neutral -0.0102 (just over soft limit), rep/prod same.
- P17 L16 a=10, L17/L22 Wait a=6, think a=-5 (030154_277): rep -0.331, prod +0.033, close +0.90, neutral -0.0067. All limits OK. GEN6 (last) on it.
- **GEN6 (P17)**: acc 0.625, cut 0.0625, closed 0.94, mean_tokens 2131 (shortest), 9 rescues/0 losses.
  Only cut tail: degenerate " wait wait wait ..." spam -> the -9x reversal of L16n998 can misfire (likely contexts where the
  neuron's activation sign flips, so the reversed write ADDS wait). Also seen in GEN1 ("= 1 wait" loop). Watch-out for final eval.
  GEN2 vs GEN6 differ by 3 problems (11 vs 10 correct) -> within noise.

## Final ranking
1. 015739_897 (P09) gen 0.6875/cut 0.06; 2. 030154_277 (P17) gen 0.625/cut 0.06, cleaner neutral; 3. 023915_620 (P15) gen 0.5625/cut 0.19.
Budget used: 17/40 probes, 6/6 gen.
