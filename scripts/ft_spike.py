"""E4.0 — viabilidade do LoRA-SFT na GTX 1660 SUPER (6 GB, sm_75, sem bf16).

Para cada comprimento de sequência: 3 passos completos (forward + backward + AdamW), mede pico de VRAM,
segundos por passo (média dos passos 2–3), perda e norma do gradiente finitas. Para no primeiro OOM.
Usa um texto real do base (dev48) para o conteúdo da sequência.
Grava data/ft/spike.json com o max_len recomendado (maior que coube com <= 45 s/passo).

uso: scripts/run_bg.sh ft-spike scripts/ft_spike.py
"""

import json
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from ft_common import chunked_loss, deltanet_path, load_lora  # noqa: E402

from wsp.paths import DATA  # noqa: E402

OUT = DATA / "ft"
OUT.mkdir(parents=True, exist_ok=True)
model, tok = load_lora()
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("LoRA params", n_train, "deltanet:", deltanet_path(model), flush=True)

texts = [json.loads(l)["text"] for l in (DATA / "gen_dev48" / "base.jsonl").read_text().splitlines()]
ids_all = tok("\n".join(texts), add_special_tokens=False)["input_ids"]
# correção da perda em blocos: compara com a CE direta nos logits completos (T=512)
with torch.no_grad():
    ids = torch.tensor(ids_all[:512], device="cuda")
    ref = torch.nn.functional.cross_entropy(model(input_ids=ids[None]).logits[0, :-1].float()[199:], ids[200:])
    mine, _ = chunked_loss(model, ids, n_prompt=200, chunk=128)
    print("check perda: direta", float(ref), "em blocos", float(mine), flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
scaler = torch.amp.GradScaler("cuda")
res = {"loss_check": [float(ref), float(mine)], "deltanet": deltanet_path(model), "lora_params": n_train, "runs": []}
for T in (1024, 2048, 4096, 6144, 8192):
    ids = torch.tensor(ids_all[:T], device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times, row = [], {"T": T}
    try:
        for step in range(3):
            torch.cuda.synchronize()
            t = time.time()
            loss, n = chunked_loss(model, ids, n_prompt=200)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t)
            row.update(loss=float(loss), grad_norm=float(gn), finite=bool(torch.isfinite(loss) and torch.isfinite(gn)))
        row.update(ok=True, s_per_step=round(sum(times[1:]) / 2, 2), first_step=round(times[0], 2),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    except torch.OutOfMemoryError as e:
        row.update(ok=False, error="OOM", detail=str(e)[:200])
    except Exception as e:  # noqa: BLE001
        row.update(ok=False, error=repr(e)[:300])
    print(json.dumps(row), flush=True)
    res["runs"].append(row)
    opt.zero_grad(set_to_none=True)
    if not row["ok"]:
        break
good = [r["T"] for r in res["runs"] if r.get("ok") and r.get("finite") and r["s_per_step"] <= 45]
res["recommended_max_len"] = max(good) if good else None
(OUT / "spike.json").write_text(json.dumps(res, indent=1))
print("recomendado", res["recommended_max_len"], flush=True)
