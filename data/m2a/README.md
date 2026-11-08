# Agent run records

One directory per run under `runs/`. These files are **verbatim**: they are what the agents were given and wrote. They
were not rewritten when the project moved to this repository, so they still use the original paths, inside a larger
harness:

| in the records | in this repository |
|---|---|
| `/mnt/dados/dev/deep_traces/data/wsp/...` or `data/wsp/...` | `data/...` |
| `scripts/wsp/...` | `scripts/...` |
| `deep_traces.wsp` / `src/deep_traces/wsp/` | `wsp` / `src/wsp/` |

Contents:

- `BRIEF.md`: the protocol v1 brief (probes only; used by r1, a0).
- `BRIEF_v2.md`: the protocol v2 brief (probes + generation checks; used by r2, a1, a2, r3).
- `gen_ref_base.json`: the base model's reference on the generation-check set (GSM8K train[200:216], 6k cap).
- `runs/<run>/`:
  - `config.json`: the budget;
  - `ledger.jsonl` / `gen_ledger.jsonl`: every probe or generation call the scorer charged;
  - `log.jsonl`: scorer results;
  - `notes.md`: the agent's own lab notes;
  - `final.json`: the agent's ranked candidates and rationale;
  - `done/<id>.pt`: every edit the agent submitted, loadable with `wsp.m2a.apply`.

Not versioned: `sealed/` (the holdout only the supervisor reads; kept out so future agents cannot see it), `inbox/`,
`outbox/`, `work/` (scratch files), and the activation atlas (`atlas.pt`, 84 MB, rebuilt by `scripts/m2a_atlas.py`).

| run | protocol | atlas | agent / method |
|---|---|---|---|
| sanity, sanity2 | v1 / v2 | — | supervisor (known edits) |
| rnd | v1 | — | random search (`scripts/m2a_random.py`) |
| r1 | v1 | yes | agent |
| a0 | v1 | no | agent |
| dose | v1 | — | supervisor (r1's directions at a small α) |
| r2, r3 | v2 | yes | agent |
| a1, a2 | v2 | no | agent |
| naive | v2 | — | naive search with the agents' toolkit (`scripts/m2a_naive.py`) |
