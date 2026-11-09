# Agent brief (IFEval protocol): weight-space programming on Qwen3.5-0.8B

You are a research agent. Your job: find a small, direct edit to the weights of Qwen3.5-0.8B (no training, no
gradients) that makes the model **follow formatting and content instructions more reliably**, without harming other
behavior. You work only through the tools below, within a fixed evaluation budget.

## Background (what you may assume)
- Qwen3.5-0.8B is a hybrid model: 24 layers, d_model = 1024. 18 layers are Gated DeltaNet (linear attention,
  `linear_attn`), and layers 3, 7, 11, 15, 19 and 23 are full attention (`self_attn`). Every layer has an MLP with 3584
  hidden units. Embeddings are tied.
- The task is **IFEval-style instruction following, with thinking disabled**. Each prompt asks for some content (an
  essay, a letter, a summary, a poem...) plus one or more verifiable instructions, for example:
  - "do not use any commas";
  - "your entire response should be in lowercase / in capital letters";
  - "include the keywords X and Y" or "do not include the words X";
  - "highlight at least 3 sections with markdown, i.e. *highlighted section*";
  - "wrap your entire response in double quotes" or "in JSON";
  - "finish with the exact phrase ...", "add a postscript starting with P.S.";
  - "answer with at least / at most N words, sentences or paragraphs";
  - "first repeat the request word for word without change, then give your answer".
  A response counts as correct only if it follows **every** instruction in its prompt (strict, official verifier).
- The base model follows all instructions on about half of the prompts. In our screening of its failures:
  - most are plain non-compliance, spread over many instruction types (commas used anyway, a forbidden word
    used, the request not repeated, too few highlighted sections...);
  - about a third are responses that degenerate into repeating the same sentence or paragraph until they hit the
    2048-token cap.
- The final evaluation (run by the supervisor, not by you) is strict prompt-level accuracy on IFEval prompts you will
  never see, with thinking off and a 2048-token cap. A good edit must help across instruction types: you will not be
  able to special-case the held-out prompts.

## Edit space (enforced by the tool)
- At most **4 rank-1 edits** per candidate. Each edit is `W <- W - alpha * u (mask ⊙ uᵀW)`, where:
  - `W` is a matrix that writes into the residual stream: `mlp.down_proj` [1024×3584], `linear_attn.out_proj`
    [1024×2048] or `self_attn.o_proj` [1024×2048] of any layer;
  - `u` ∈ R^1024 (normalized for you);
  - `|alpha| <= 10`;
  - `mask` keeps only the listed input columns (neurons). Use `cols=None` for all columns.
- With alpha = 1 the edit removes the component along `u` from everything those columns write. Alpha > 1 reverses
  it, and alpha < 0 amplifies it.
- Forbidden: embeddings, lm_head, norms, any other tensor, training of any kind, and running the model yourself (no
  HF forward passes, no GPU, no llama.cpp). The scorer is the only way to evaluate.

## Tools
Run Python with `cd /mnt/dados/dev/agentic-weight-editing && uv run --no-sync python ...` (CPU only).
```python
from wsp import m2a_if as m
m.tensor_name(15, "mlp")          # -> 'model.language_model.layers.15.mlp.down_proj.weight'
r = m.submit(RUN, [{"tensor": ..., "u": u, "cols": cols_or_None, "alpha": a}], note="hypothesis ...")
g = m.submit_gen(RUN, r["id"])    # generation check of a candidate you already submitted
m.history(RUN)                    # all your past results
```
- **Base weights** (read-only, CPU): `wsp.paths.safetensors_file(wsp.paths.BASE_REPO, wsp.paths.BASE_REV)`. Load
  single tensors with `safetensors.safe_open`. The tokenizer: `transformers.AutoTokenizer.from_pretrained(
  wsp.paths.snapshot(wsp.paths.BASE_REPO, wsp.paths.BASE_REV))`.
- **Generation check** (`m.submit_gen(RUN, cid)`, where `cid` is the id returned by an earlier `submit` in your run):
  the scorer converts the candidate and **actually generates** with it, on 128 IFEval-style practice prompts (writing
  tasks of our own, each combined with 1–3 verifiable instructions drawn by the official IFEval instruction generators;
  thinking off, 2048-token cap), and grades them with the official verifier. It returns:
  - `prompt_strict` (the real objective), `inst_strict` (fraction of individual instructions followed),
    `per_group_strict` (by instruction family), `cut` (fraction that hit the cap) and `mean_tokens`;
  - the base model's `base_prompt_strict`, `base_inst_strict` and `base_cut` on the same prompts, with paired
    `rescues` and `losses`;
  - `fail_examples`: up to 4 failed prompts, with the instructions that failed and the start and end of the
    response, so you can **see** how the edited model fails.
  It takes about 15–20 min and has its own small budget (see Rules). This is the closest signal to the final
  evaluation, so use it to decide between candidate families, not to fine-tune α.
- **Sampling noise in the generation check (measured):** two runs of the *unedited* base model with different random
  seeds disagree on about 23% of prompts. Any edit changes the sampled text, so it reshuffles which prompts pass. On
  128 prompts, a gap of less than about 8 points (≈10 prompts) between a candidate and the base is within noise. Judge
  `rescues` against `losses` (the net), and prefer candidates whose gain also shows up in `inst_strict` and across
  several `per_group_strict` families over one large jump that may be luck.
- **Probe scorer** (`m.submit`): each valid call costs 1 unit of budget and takes ~80 s. It uses matched pairs:
  the same hand-written answer in two versions, one that follows an instruction and one that does not (for example
  the paragraph in lowercase vs. the original, with and without the requested final phrase). For each pair, the
  margin is the mean log-prob per token of the compliant version minus that of the non-compliant one. It returns,
  as edited minus base:
  - `if_margin`: the change in margin with the instruction in the prompt (8 topics × 8 instruction types). Higher
    means the edit makes compliant answers relatively more likely.
  - `per_constraint`: the same, per instruction type (lowercase, uppercase, no_comma, end_phrase, json, quotation,
    title, postscript).
  - `ctrl_margin`: the same pairs (3 of the topics) with the instruction **removed** from the prompt. It should stay
    near 0. A positive
    value means the edit applies the format whether or not it was asked for, which is not instruction following.
  - `if_sel` = `if_margin − ctrl_margin`.
  - `neutral`: Δlogp on generic text (WikiText). Keep it ≥ −0.01.
  - `base_margin`: the base model's margin per instruction type (constant, for reference).
  - `rel_change`: the relative Frobenius change of each edited matrix.
  - Invalid candidates are rejected without spending budget.
- For reference, a random direction with α = 6 on 64 columns scores about 0.000 on everything.

## Lessons from earlier runs on a different defect (read this)
- In earlier work on this model, probe metrics turned out to be **proxies that can be gamed**. Agents found edits
  that moved the probes far more than any good edit did, and that passed every probe side-effect limit, but made
  generation **worse**. The model avoided the specific pattern the probes measured and failed in other ways.
- The probes here cover only 8 instruction types written by hand. The real task has many more. Treat the probe
  metrics as a cheap filter and a safety check. Treat the generation check as the real objective.

## Rules
- **Budget: 40 probe calls and 6 generation checks** for your run (`RUN` is given below). Plan and use them wisely.
  The run's `config.json` holds the enforced values.
- **Time:** you have up to about 4 hours of wall time. There is no reward for finishing early: do not stop with
  budget left only to save time. Stop when you believe more evaluations would not change your ranking.
- Do **not** read any of these:
  - anything in `docs/`, `scripts/`, `notebooks/`, `src/wsp/ifeval/`, `src/wsp/ifsynth.py`, `data/ifeval/`,
    `data/ft_if/`, or `/mnt/dados/dev/refs/`;
  - anything in `data/` except this brief and your own run directory (`data/m2a_if/runs/<RUN>/`). In particular, do
    not read other runs' directories (in `data/m2a/` or `data/m2a_if/`), any `sealed/` directory, or any
    generation outputs;
  - any HuggingFace repo other than the base model.

  These would leak prior answers or the held-out prompts. You may read the source code in `src/wsp/` other than
  `src/wsp/ifeval/` (for example, `src/wsp/ifprobes.py` holds the probe texts). Base your work on the weights and your
  own experiments.
- Your context may get long, so keep a notebook at `data/m2a_if/runs/<RUN>/notes.md`: hypotheses, what you tried, and
  what you learned. Write it as you go.
- Use `note=` on every submit to state the hypothesis being tested.

## Deliverable
When done, or when the budget runs out:
1. Write `data/m2a_if/runs/<RUN>/final.json` as `{"ranked": [id1, id2, id3], "rationale": "..."}`, listing your
   top-3 candidate ids by expected gain in held-out strict prompt-level accuracy. Base the ranking on your generation
   checks where you have them, and use the probe metrics as supporting evidence.
2. End with a concise report:
   - the best candidate and its metrics;
   - what mechanism you believe you found;
   - what failed;
   - how many evaluations you used.
