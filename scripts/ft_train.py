"""E4.2 — LoRA-SFT do base nos traços autodestilados (data/ft/sft.jsonl).

LoRA r=16, alpha=32, em todas as projeções lineares (atenção, MLP, DeltaNet); AdamW lr 1e-4 com aquecimento
linear de 20 passos e decaimento cosseno; acumulação de 8 sequências; fp16 com GradScaler; perda só na resposta.
Sequências mais longas que max_len (de data/ft/spike.json, ou --max-len) são descartadas, não truncadas
(truncar ensinaria a cortar o raciocínio no meio). Salva o adaptador a cada época em data/ft/<nome>/ep<k>.

uso: scripts/run_bg.sh ft-train scripts/ft_train.py sft [--epochs 3] [--max-len N] [--lr 1e-4]
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ft_common import chunked_loss, load_lora  # noqa: E402

from wsp.paths import DATA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--data", default=str(DATA / "ft" / "sft.jsonl"))
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--max-len", type=int, default=None)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--r", type=int, default=16)
A = ap.parse_args()
OUT = DATA / "ft" / A.name
OUT.mkdir(parents=True, exist_ok=True)
max_len = A.max_len or json.loads((DATA / "ft" / "spike.json").read_text())["recommended_max_len"]
if not max_len:
    sys.exit("spike.json sem max_len recomendado: LoRA não coube na GPU")

model, tok = load_lora(r=A.r, alpha=2 * A.r)
EOS = "<|im_end|>"
data = []
for line in Path(A.data).read_text().splitlines():
    r = json.loads(line)
    p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    c = tok(r["completion"] + EOS, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) <= max_len:
        data.append((torch.tensor(p + c), len(p), r["kind"]))
kinds = {k: sum(1 for d in data if d[2] == k) for k in ("natural", "excised")}
print(f"{len(data)} sequências <= {max_len} tokens {kinds}", flush=True)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
steps = math.ceil(len(data) / A.accum) * A.epochs
warm = min(20, steps // 10 + 1)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
rng = random.Random(0)
log = (OUT / "log.jsonl").open("a")
cfg = {**vars(A), "max_len": max_len, "n_seq": len(data), "kinds": kinds, "opt_steps": steps}
(OUT / "config.json").write_text(json.dumps(cfg, indent=1))
step, t0 = 0, time.time()
for ep in range(1, A.epochs + 1):
    order = list(range(len(data)))
    rng.shuffle(order)
    run_loss, run_tok = 0.0, 0
    for k, j in enumerate(order):
        ids, n_prompt, _ = data[j]
        loss, n = chunked_loss(model, ids.cuda(), n_prompt)
        scaler.scale(loss / A.accum).backward()
        run_loss += float(loss) * n
        run_tok += n
        if (k + 1) % A.accum == 0 or k + 1 == len(order):
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            row = {"ep": ep, "step": step, "loss": round(run_loss / max(run_tok, 1), 4), "grad_norm": round(gn, 3),
                   "lr": sched.get_last_lr()[0], "scale": scaler.get_scale(), "min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            run_loss, run_tok = 0.0, 0
    model.save_pretrained(OUT / f"ep{ep}")
    print(f"época {ep} salva ({(time.time() - t0) / 60:.0f} min)", flush=True)
