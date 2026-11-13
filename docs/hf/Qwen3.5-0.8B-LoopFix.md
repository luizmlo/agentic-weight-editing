---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: finetune
language:
- en
pipeline_tag: text-generation
tags:
- weight-editing
- weight-space-programming
- rank-1-edit
- reasoning
- gguf
---

# Qwen3.5-0.8B-LoopFix

Qwen3.5-0.8B with **two rank-1 weight edits** that stop most of its thinking loops ("Wait, … Yes, … Wait, …").
No training was involved: a Claude agent found the edits by reading and testing the weights, and we then removed the
parts of its edit that made no measurable difference.

With thinking on and a 12,288-token budget, the original model runs out of space on about 85% of grade-school math
problems before answering. This version finishes most of them, and on 200 locked GSM8K test problems it solves 67.5%
with about 4,200 tokens per answer.

**Read this first.** On this model, **switching thinking off is a strong and much cheaper alternative**: on the same
200 problems the unedited model solves 58.0% with thinking off, using about 1,400 tokens per answer. This edit is
for when you want the thinking mode to work. It is a research artifact from a single-GPU study, evaluated on
48–200 problems per set; see the limitations below.

## What was changed

Each edit changes one output projection $W$ (what a set of neurons writes into the residual stream), only along one
direction $u$:

$$W \leftarrow W - \alpha\, u\, (\text{mask} \odot u^\top W)$$

| tensor | columns | α | effect |
|---|---|---|---|
| `model.language_model.layers.16.mlp.down_proj.weight` | neuron 998 | +10 | flips this neuron's write along the "Wait" direction (−9×) |
| `model.language_model.layers.22.linear_attn.out_proj.weight` | all | −4 | amplifies the layer's write along the `</think>` direction (5×) |

Every other tensor and file is identical to [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B). The
edits are in `edits/` (safetensors and JSON, no pickle), with `edits/apply_edits.py` to re-apply them to the
original weights yourself.

In the study, a second agent's 4-edit fix ("a2") also edited L16 neuron 3422, L17 neuron 1915 and L22 neuron 573.
Removing those three made no measurable difference on any set we tested (table below), so this release keeps the
2-edit version. Both edits are needed: the layer-16 edit alone lowers "Wait" but leaves 25 of 48 answers cut off.

## Results

Thinking on, 12,288-token budget, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 over the whole output,
llama.cpp (F16). Accuracy is exact match on the final answer; "cut off" = still running at the budget.

| set | original | original, thinking off | full 4-edit a2 | **this model (2 edits)** |
|---|---|---|---|---|
| GSM8K train[0:48] (practice set, used to choose this variant) | 14.6% (85% cut off) | — | 64.6% | **68.8%** (12.5% cut off) |
| GSM8K test[400:600], locked, pre-registered | — | 58.0% (1,365 tokens) | 64.0% (4,701 tokens) | **67.5%** (4,217 tokens, 12% cut off) |
| MATH-500[0:148] | 8.1% (92% cut off) | 43.9% (3,842 tokens) | 48.6% | **47.3%** (7,649 tokens, 43% cut off) |

- Thinking off used an 8,192-token cap. On another 200 locked GSM8K problems, the original model with thinking on
  solved 15.5%.
- Against the full a2, the differences are within noise: 28 problems rescued and 21 lost on the locked test
  (p = 0.39), 12 and 14 on MATH.
- Against thinking off, it is +9.5 points on the locked GSM8K slice (39 rescued, 20 lost; p = 0.02, uncorrected, not
  pre-registered) and +3.4 on MATH (a tie).
- On MATH-500 with the same budget, the edit published in *Sparks of Weight-Space Programming* solves 20.9%; the full
  a2 beats it on 43 problems and loses 2. That edit is still improving at 12k tokens, so this comparison is about
  efficiency at this budget, not a ceiling. On the first locked GSM8K slice with a 20,480-token budget, it catches up
  (65.0%, against 63.5% for the full a2), using about twice the tokens.

## How to use

The edit targets the thinking mode, so leave thinking on and give it room (we used 12,288 new tokens). Use the
sampling settings above; the presence penalty matters for this model.

**llama.cpp / Ollama / LM Studio:** `Qwen3.5-0.8B-LoopFix-F16.gguf`.

```bash
llama-server -m Qwen3.5-0.8B-LoopFix-F16.gguf -c 16384 --temp 0.6 --top-p 0.95 --top-k 20 \
  --presence-penalty 1.5 --repeat-last-n -1
```

**transformers / vLLM:** this folder loads like the original model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "<hf-user>/Qwen3.5-0.8B-LoopFix"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")
```

## Limitations

- **One model, one defect.** The edit fixes thinking loops in this 0.8B model. It does not add knowledge or skills;
  it lets the model finish reasoning it could already do.
- **Small evaluations.** 48–200 problems per set (±3–4 points). The locked test above is a single run.
- **Thinking off is cheaper.** See the first section. On MATH it ties this edit at about half the tokens.
- **Finished-but-wrong answers.** Among answers that finish, the edited models are somewhat less accurate than the
  rare original answers that finish; the gain comes from finishing far more often. A post hoc analysis suggests this
  is partly premature conclusion, strongest for edits with a large `</think>` push; for the full a2 the effect was
  small and not significant.
- **Side effects: none detected, on small panels.** Paired with the original model on the same items and seeds:
  - tool calling with thinking off, 300 requests: 90.3% vs 89.7%;
  - grade-school math with thinking off, 200 problems: 46.0% vs 48.5% (17 gained, 22 lost; p = 0.52);
  - tool calling with thinking on, 100 requests: 94% vs 93%.

  All are within one standard error. These panels only detect changes of about 5–7 points. No safety evaluation
  beyond the original model's was done.
- **Sampler.** Numbers are from llama.cpp with the settings above; other samplers shift absolute numbers.

## How it was made

From the pre-paper *Agentic Weight-Space Programming as an Alternative to Fine-Tuning in Small Language Models*
(Luiz Felipe Melo, Unisul, 2026). Claude Opus 5.5 agents received the weights, a scorer with a small generation budget and
at most 4 rank-1 edits; the second agent's pick ("a2") was later reduced to these 2 edits by ablation, with
predictions written down before each test. Code, lab notebook and every submitted edit: <https://github.com/luizmlo/agentic-weight-editing>. Pre-paper: <https://doi.org/10.5281/zenodo.24208535>.

## License

Apache 2.0, inherited from Qwen3.5-0.8B (Copyright 2026 Alibaba Cloud). See `LICENSE` and `NOTICE` (what was
changed).

## Citation

```bibtex
@misc{melo2026agenticwsp,
  title  = {Agentic Weight-Space Programming as an Alternative to Fine-Tuning in Small Language Models},
  author = {Melo, Luiz Felipe},
  year   = {2026},
  note   = {Pre-paper},
  doi    = {10.5281/zenodo.24208535}
}
```
