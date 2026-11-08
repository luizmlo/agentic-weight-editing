"""PoC-2: viabilidade na GTX 1660 Super (sem bf16).

1. Numérica: logits fp16 vs fp32 (GPU) no mesmo prompt — overflow/NaN? diferença de top-1?
2. Edição em memória: aplicar a edição recuperada no fp16 reproduz o checkpoint editado?
3. Throughput de geração (tok/s) para batch 1, 8, 32.
"""

import json
import time

import torch

from wsp import modelio
from wsp.paths import DATA, EDITED_REPO

out = {}
msgs = [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nPlease reason step by step, and put your final answer within \\boxed{}."}]

# --- 1. fp32 referência
m32, tok = modelio.load(torch.float32)
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
ids = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    ref = m32(**ids).logits[0].float().cpu()
del m32
torch.cuda.empty_cache()

m16, _ = modelio.load(torch.float16)
with torch.no_grad():
    l16 = m16(**ids).logits[0].float().cpu()
out["fp16_nan"] = bool(torch.isnan(l16).any() or torch.isinf(l16).any())
out["fp16_max_abs_logit_diff"] = (l16 - ref).abs().max().item()
out["fp16_top1_agree"] = (l16.argmax(-1) == ref.argmax(-1)).float().mean().item()
lp16, lp32 = l16.log_softmax(-1), ref.log_softmax(-1)
out["fp16_kl_mean"] = (lp32.exp() * (lp32 - lp16)).sum(-1).mean().item()
print(json.dumps(out))

# --- 2. edição em memória vs checkpoint editado
dirs = torch.load(DATA / "edit_dirs.pt")
undo = modelio.apply_published_edit(m16, dirs)
with torch.no_grad():
    le_mem = m16(**ids).logits[0].float().cpu()
undo()
me, _ = modelio.load(torch.float16, repo=EDITED_REPO, revision=None)
with torch.no_grad():
    le_ck = me(**ids).logits[0].float().cpu()
del me
torch.cuda.empty_cache()
out["edit_mem_vs_ckpt_max_diff"] = (le_mem - le_ck).abs().max().item()
out["edit_mem_vs_ckpt_top1_agree"] = (le_mem.argmax(-1) == le_ck.argmax(-1)).float().mean().item()
print(json.dumps({k: out[k] for k in out if k.startswith("edit")}))

# --- 3. throughput
tok.padding_side = "left"
thr = {}
for bs in (1, 8, 32):
    batch = tok([prompt] * bs, return_tensors="pt", padding=True).to("cuda")
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        g = m16.generate(**batch, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_new_tokens=256)
    torch.cuda.synchronize()
    dt = time.time() - t
    n = (g.shape[1] - batch["input_ids"].shape[1]) * bs
    thr[bs] = {"tok_s": round(n / dt, 1), "secs": round(dt, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print(bs, thr[bs])
out["throughput_256new"] = thr
(DATA / "poc2_hw.json").write_text(json.dumps(out, indent=1))
