# Reproducing

Everything here ran on one machine: GTX 1660 SUPER (6 GB, sm_75, no bf16, no tensor cores), 15 GB RAM, Linux,
CUDA 13 driver. The commands assume the repository root as the working directory.

## Environment

```bash
uv sync          # Python 3.12; pinned: torch 2.14.1+cu130, transformers 5.22.1, flash-linear-attention 0.6.0,
                 # peft 0.21.2, datasets 5.1.0, safetensors 0.8.0 (see pyproject.toml / uv.lock)
```

- `causal_conv1d` is not installed (it needs nvcc), so transformers uses its PyTorch fallback. That is correct, just
  slower.
- The Gated DeltaNet layers run through fla's Triton kernels, which work on sm_75 but only on CUDA tensors. CPU-only
  tests of the HF path fail.
- Models come from the HF cache: `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17` and
  `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (the Astra edit). See `src/wsp/paths.py`.

### External tools

| tool | used for | where it is configured |
|---|---|---|
| llama.cpp `convert_hf_to_gguf.py` | edited safetensors → f16 GGUF | `CONVERT` in `scripts/poc7_build.py`, `scripts/ft_export.py` |
| `llama-server` shipped with Ollama | parallel generation (`-np 16`) | `BIN` in `scripts/llama_serve.sh` |
| Paper logs (`lm_understand-logs/exp8_math`) | reference texts for the PoC probes | `REFS` in `src/wsp/paths.py` |

We use Ollama's `llama-server` directly because Ollama forces `-np 1` for the `qwen35` architecture.
`GGML_BACKEND_PATH` must point to the CUDA backend; without it the server silently runs on CPU at ~8 tok/s.
`llama_serve.sh` handles this.

## Running jobs

Long jobs run as systemd user units, so they survive the terminal and an OOM kills only the job:

```bash
scripts/run_bg.sh <name> <script.py|script.sh> [args...]     # unit wsp-<name>, log data/logs/<name>.log
WSP_MEM_MAX=12G scripts/run_bg.sh ...                         # raise the default 6G memory cap
```

Only one model fits on the GPU at a time. The queues stop llama-server before any HF stage (`scripts/llama_serve.sh stop`).

## Building an edited model

An edit is a list of rank-1 operations `{layer, kind, u, cols, alpha}` (see `src/wsp/m2a.py: validate/apply`).

```bash
uv run python scripts/poc7_build.py astra_check                              # rebuild Astra from base + recovered edit, compare, convert
uv run python scripts/poc7_build.py m2a_a1_062959_102 data/m2a/runs/a1/done/062959_102.pt   # an agent's candidate -> data/gguf/
```

Every candidate an agent submitted is versioned as `data/m2a/runs/<run>/done/<id>.pt` (267 files); the PoC edit
families are in `data/poc7_dirs.pt` and `data/poc8_cells.pt`. The Astra edit itself is recovered from the public
checkpoints by `scripts/poc1_diff.py` (writes `data/edit_dirs.pt`).

## Evaluation

```bash
scripts/llama_serve.sh start <model.gguf|ollama-name> 16 13312         # usually done by the eval scripts themselves
uv run python scripts/gen_eval.py dev48 48 12288 base,astra,m2a_a1_062959_102 --np 16 --split train
WSP_UNLOCK_TEST=1 uv run python scripts/gen_eval.py t12k 32 12288 base,<arm> --np 16     # locked test (logged in data/ledger.jsonl)
uv run python scripts/math_eval.py math100b 100 12288 base,<arm> --np 16 --offset 48
```

- An arm is `base`, `astra`, any `data/gguf/<name>.gguf`, a decoding arm, or `<model>+<decoding>`:
  - `lb_w<A>_t<B>`: logit bias −A on "Wait"/" Wait" and +B on `</think>`. `lbx_...` also biases "wait"/" wait".
  - `bf<N>`: budget forcing. Generate up to N tokens; if `</think>` is missing, inject `\n</think>\n\n` and continue to
    the cap.
  - Example combo arm: `m2a_a1_062959_102+bf4096`.
- Generations are checkpointed per problem in `data/gen_<tag>/<arm>.jsonl`; re-running resumes. The summaries
  (`data/gen_<tag>.json`) hold per-problem correctness and the paired comparisons.
- Sampling: THINK35 = temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 over the whole output
  (`repeat_last_n = max_new + 512`).

## Agent runs (scorer + brief)

```bash
scripts/run_bg.sh m2a scripts/m2a_scorer.py        # scorer daemon: HF fp16 model resident; v2 adds generation checks on port 8091
```

1. Create `data/m2a/runs/<run>/config.json` with `{"budget": 40, "gen_budget": 6}`.
2. Give the agent `data/m2a/BRIEF_v2.md` (rewrite its absolute paths for this layout) and point it at its run
   directory. The agent submits with `wsp.m2a.submit(run, edits)` and `wsp.m2a.submit_gen(run, cid)`, and reads its
   history with `wsp.m2a.history(run)`.
3. The agent finishes by writing `final.json` (ranked candidates + rationale).
4. The supervisor confirms on dev48 with `scripts/queue_confirm.sh <run>[:<id>] ...` and audits the transcript for
   forbidden reads.

Controls: `scripts/m2a_random.py` (random search, v1) and `scripts/m2a_naive.py` (naive search with the winning
agents' toolkit and the generation signal, E3).

## Fine-tuning baseline (E4)

```bash
WSP_MEM_MAX=12G scripts/run_bg.sh e4 scripts/queue_e4.sh
# = ft_spike.py (feasibility) -> ft_data.py 600 (self-distillation + loop excision) -> ft_train.py sft --epochs 3 --max-len 8192
#   -> ft_export.py (merge LoRA -> GGUF) per epoch -> pick the epoch on the 16 gen-check problems -> dev48
```

## Results notebook

```bash
uv sync --group analysis                      # plotly, kaleido, jupyter kernel
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
```

The notebook reads only `data/` (small versioned summaries). Each chart is saved twice: as an interactive plotly
figure with tooltips (JupyterLab, VS Code) and as a static PNG (GitHub). The PNG needs kaleido and a Chrome or
Chromium. With a snap-confined Chromium, the notebook points kaleido at `~/snaptmp`, because snaps can't read `/tmp`
or hidden folders in `$HOME`. Without a browser, the PNGs are skipped and the interactive charts still work.

## Gotchas we hit (in order of pain)

1. **HF gradient checkpointing only acts in train mode.** `from_pretrained` returns the model in eval mode, and LoRA
   then OOMs at T = 1024 on 6 GB. Call `model.train()`; dropout is 0 here, so nothing else changes.
2. **The 248k vocabulary.** Never materialize full logits for long sequences. The loss is computed in checkpointed
   blocks of 256 positions (`scripts/ft_common.py: chunked_loss`).
3. **llama-server prompt cache.** The default `--cache-ram 8192` led to an OOM kill of the server. Use
   `--cache-ram 0 --ctx-checkpoints 1` on the server and `cache_prompt: false` in the client.
4. **Presence-penalty window.** llama.cpp defaults to the last 64 tokens, while vLLM uses the whole output. Set
   `repeat_last_n` explicitly.
5. **llama-server feeds prompt tokens to the sampler.** With a presence penalty this punishes copying arguments from the
   prompt, so tool-calling evals use no penalty.
6. **RAM.** With 15 GB, two agents each loading fp32 embeddings (~1 GB) plus a browser triggered systemd-oomd, which
   killed the whole terminal scope. Agent briefs now cap memory per process.
7. **Power loss.** Generation and training checkpoint per problem and per epoch. Keep the data disk's fstab entries
   (including bind mounts) `nofail`.
