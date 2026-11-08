# M→A agent brief: weight-space programming on Qwen3.5-0.8B

You are a research agent. Your job: find a small, direct edit to the weights of Qwen3.5-0.8B (no training,
no gradients) that makes its long chain-of-thought reasoning less prone to **repetition loops**, without
harming other behavior. You work only through the tools below, within a fixed evaluation budget.

## Background (what you may assume)
- Qwen3.5-0.8B is a hybrid model: 24 layers, d_model = 1024. 18 layers are Gated DeltaNet (linear
  attention, `linear_attn`), and layers 3, 7, 11, 15, 19 and 23 are full attention (`self_attn`). Every
  layer has an MLP with 3584 hidden units. Embeddings are tied.
- With thinking enabled, the base model often falls into **degenerate loops** on math problems. It
  re-checks or restates the same fact over and over (for example "Wait, is X equal to 5? Yes, X is 5.")
  until it hits the token cap, and never writes an answer. In natural traces the hard loop usually
  starts 3k to 10k tokens into the reasoning. Before that, the reasoning is mostly productive, with
  scattered local re-checks.
- Prompt-only fixes do not work. System prompts such as "never repeat yourself" did **not** reduce
  looping in generation, and in fact any system prompt slightly increased it.
- The final evaluation (run by the supervisor, not by you) is GSM8K accuracy with thinking enabled and
  a 12k-token cap, on a locked test split. A large share of base failures are loops that hit the cap.

## Edit space (enforced by the tool)
- At most **4 rank-1 edits** per candidate. Each edit is `W <- W - alpha * u (mask ⊙ uᵀW)`, where:
  - `W` is a matrix that writes into the residual stream: `mlp.down_proj` [1024×3584],
    `linear_attn.out_proj` [1024×2048] or `self_attn.o_proj` [1024×2048] of any layer;
  - `u` ∈ R^1024 (normalized for you);
  - `|alpha| <= 10`;
  - `mask` keeps only the listed input columns (neurons). Use `cols=None` for all columns.
- With alpha = 1 the edit removes the component along `u` from everything those columns write.
  Alpha > 1 reverses it, and alpha < 0 amplifies it.
- Forbidden: embeddings, lm_head, norms, any other tensor, training of any kind, and running the model
  yourself (no HF forward passes, no GPU, no llama.cpp). The scorer is the only way to evaluate.

## Tools
Run Python with `cd /mnt/dados/dev/deep_traces && uv run --no-sync python ...` (CPU only).
```python
from deep_traces.wsp import m2a
atlas = m2a.load_atlas()        # mapping data (M phase); read atlas["README"] and atlas["shapes"]
m2a.tensor_name(15, "mlp")      # -> 'model.language_model.layers.15.mlp.down_proj.weight'
r = m2a.submit(RUN, [{"tensor": ..., "u": u, "cols": cols_or_None, "alpha": a}], note="hypothesis ...")
m2a.history(RUN)                # all your past results
```
- **Atlas** (`data/wsp/m2a/atlas.pt`): mean activations of the BASE model at every layer, for three
  signals: residual output, MLP down_proj input (neuron activations), and attention out-proj input.
  It covers two families of contexts:
  - `natural`: 18 real base-model loop traces, comparing the loop region with the productive
    pre-loop region;
  - `branch`: 12 matched pairs. Each pair has the same reasoning prefix followed by either a
    repetitive or a productive continuation, and is recorded under 4 different system prompts
    (`none`, `anti_rep`, `placebo`, `concise`), so you can map what a prompt changes internally.
  - The probe texts are in `atlas["branch_cases"]`.
- **Base weights** (read-only, CPU): `deep_traces.wsp.paths.safetensors_file(BASE_REPO, BASE_REV)`.
  Load single tensors with `safetensors.safe_open`.
- **Scorer** (`m2a.submit`): each valid call costs 1 unit of budget and takes ~20–60 s. It returns
  Δlogp (edited − base, nats/token) on these measures:
  - `branch_rep`: Δlogp on the repetitive continuations of the 12 matched pairs. **Primary objective:
    more negative is better.** On 10 previously tested edits, this metric ranked GSM8K accuracy with
    Spearman 0.89, and the absence of loops with 0.96.
  - `branch_prod`: Δlogp on the productive continuations. Keep it near 0, with ≥ −0.02 as a soft limit.
  - `close`: Δlogp of closing the reasoning (`</think>`) right after a prefix. A positive value means
    the edit encourages stopping too early; keep it ≤ +1.
  - `neutral`: Δlogp on generic text (WikiText). Keep it ≥ −0.01.
  - `rel_change`: the relative Frobenius change of each edited matrix.
  - Invalid candidates are rejected without spending budget.
- For reference, a random direction with α = 6 on 64 columns scores about 0.000 on everything.

## Rules
- **Budget: 100 scorer calls** for your run (`RUN` is given below). Plan and use it wisely.
- Do **not** read any of these:
  - anything in `docs/`, `/mnt/dados/dev/refs/`, or `scripts/wsp/poc*`;
  - anything in `data/wsp/` except `data/wsp/m2a/atlas.pt`, this brief, and your own run directory;
  - other runs' directories, or any `sealed/` directory;
  - any HuggingFace repo other than the base model (in particular, no "weight-space-programming" checkpoints).

  These would leak prior answers. Base your work on the atlas, the weights, and your own experiments.
- Your context may get long, so keep a notebook at `data/wsp/m2a/runs/<RUN>/notes.md`: hypotheses,
  what you tried, and what you learned. Write it as you go.
- Use `note=` on every submit to state the hypothesis being tested.

## Deliverable
When done, or when the budget runs out:
1. Write `data/wsp/m2a/runs/<RUN>/final.json` as `{"ranked": [id1, id2, id3], "rationale": "..."}`,
   listing your top-3 candidate ids by expected GSM8K gain. Weigh `branch_rep` against the side-effect
   limits.
2. End with a concise report:
   - the best candidate and its metrics;
   - what mechanism you believe you found;
   - how the atlas (the mapping) guided you;
   - what failed;
   - how many evaluations you used.
