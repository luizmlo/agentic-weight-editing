# f2 notes (no-atlas control arm)
Budget 40 probe / 6 gen. Sources: weights + embeddings only.

## Weight analysis
- Final norm weight g (mean 3.3). Residual dir for token t ~ g*(E[t]-mean).
- Token ids: Wait 13784, " Wait" 13428, </think> 248069.
- Neurons aligned with wait dir are mostly semantic verb "wait" (L17 mlp n1915, L18 n881, L22 n573, L20 n405) -> not discourse.
  Anti-Wait: L16 mlp n998 (cos -.49), L17 linear_attn n741 (BOT: Wait, Wait, WAIT, Actually).
- cached: work/dirs.pt (token dirs, g), work/proj.pt (per-column projections)

## Probes
- P1 raw wait dir u=g*(E_wait-mean), L22-23 all 4 writers a=1: rep -.107 prod +.02 close -.03 neut +.005. Cases 7-12 (Wait-initial?) move, 1-6 not.
- P2 same a=4: rep -.81, close +1.36, neut -.079 (too harmful)
- P3 recheck-prod dir a=4: rep -.19 prod -.19 -> bad, abandoned
- P4 amplify </think> dir a=-1 L22-23: close +1.53, neut -.03, rep ~0. close easy to move.
- P5 raw wait, mlps L20-23 a=4: rep -.31, neut -.033
- Whitened dir: u=(Cov(R)+lam*I)^-1 r_wait, R=E*g (readouts). selectivity z 15->30. cached white.pt
- P6 whitened wait, L22-23 all a=4: rep -.66 neut -.034 close -1.69 (!sign flips vs raw)
- P7 whitened wait, attn outs L20-23 a=4: rep -.28 prod +.006 neut -.007 close -1.47  <- clean-ish; MLP edits cause neutral damage
- weights: projection energy onto dirs is NOT concentrated in columns (top128 ~30-45%) -> no weight-only basis for col selection
- P8 u=wh_wait-0.5*wh_close attn L20-23 a=4: close +7.77 (way too much; close hypersensitive), rep -.195
- P9 RAW wait dir, attn outs L20-23 a=4 (id 155707_068): rep -.45 prod -.002 close -.26 neut -.003  <- clean, best. raw > whitened on attn.
- P10 whitened attn L20-23 a=8: rep -.74 but neut -.126 -> bad
- GEN1 on P9 launched 16:01
- **GEN1 P9 (155707_068): acc .50 (base .0625), cut .50 (base 1.0), rescues 7 losses 0, mean_tokens 4787.** Remaining cuts: non-Wait loops ("Period 334 data..." counting, "I will use \boxed{4}" repeated = wants to finish but never closes think, overthinking). -> add mild close promotion.
- P11 raw wait attn L20-23 a=6: rep -.76 prod -.014 close +1.05 neut -.020 (over limit)
- P12/P13 u=wait-b*close (b=.15/.3): close +4.26/+8.19, rest unchanged. close slope ~30/unit b.
- P14 b=0.04 a=4 (162904_881): rep -.446 prod -.002 close +.96 neut -.003  -> GEN2
- P15 a=5: rep -.60 prod -.009 close +.27 neut -.010
- P16 attn L19,21,22,23 a=4: rep -.53 close -.47 neut -.001
- P17 attn L15,19,22,23 a=4: rep -.64 prod -.007 close -.91 neut -.004 -> full-attention layers carry more of the Wait push
- **GEN2 P14 (162904_881): acc .625, cut .25, closed .75, rescues 9 losses 0, mean_tok 4520.** close-swap helps. Tails: verbatim quote loops, question re-reading.
- P18 b=.07 a=4 attn L15,19,22,23 (164753_511): rep -.641 prod -.006 close +.63 neut -.004 -> GEN3
- P19 b=.04 a=4 attn L11,15,19,23: rep -.51 close -.59 neut -.002
- P20 b=.03 a=5 attn L20-23: rep -.60 prod -.008 close +1.42 neut -.009
- P21 b=.08 a=4 attn L20-23 (165242_565): rep -.44 close +2.18 neut -.003
- **GEN3 P18 (164753_511): acc .4375, cut .5625, closed .44, rescues 6 losses 0, mean_tok 5348.** Worse than P14 despite stronger rep probe -> probe does not rank within family; L15/19 swap not better. Tails: verbatim line loops, counting loops ("Course 389...").
- (17:17 resumed after rate-limit interruption; 21 probes, 3 gens used) -> GEN4 on P21 (more close)
- **GEN4 P21 (165242_565): acc .5625, cut .375, closed .625, rescues 8 losses 0, mean_tok 4419.** ~same as P14 within n=16 noise; more close (+2.18) not clearly better and is over the soft limit.
- 18:23 stop (time budget). Used 21/40 probes, 4/6 gens. Final ranking: P14 > P21 > P9.
