"""E4.2 — mescla um adaptador LoRA nos pesos do base e converte para GGUF (mesma cadeia do poc7_build).

W' = W + (alpha/r) · B·A, em fp32, sobre os pesos originais do safetensors; os tensores mesclados são gravados
em fp32 e o conversor arredonda para f16 (o mesmo formato de todos os outros braços).
Diagnóstico de precisão: quanto da atualização sobrevive ao arredondamento para f16
(rel_err = ||f16(W+Δ) − (W+Δ)|| / ||Δ||), por tensor e agregado.

uso: ft_export.py <nome_gguf> <dir_do_adaptador>     -> data/gguf/<nome_gguf>.gguf (+ .json)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from wsp.paths import BASE_REPO, BASE_REV, DATA, safetensors_file, snapshot

CONVERT = "/mnt/dados/dev/refs/llama.cpp/convert_hf_to_gguf.py"
name, adir = sys.argv[1], Path(sys.argv[2])
acfg = json.loads((adir / "adapter_config.json").read_text())
scale = acfg["lora_alpha"] / acfg["r"]
ad = load_file(adir / "adapter_model.safetensors")

src = snapshot(BASE_REPO, BASE_REV)
shard = safetensors_file(BASE_REPO, BASE_REV)
tmp = DATA / "snap" / name
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
for f in src.iterdir():
    if f.name != shard.name:
        shutil.copy(f, tmp / f.name)
with safe_open(shard, "pt") as fh:
    meta = fh.metadata()
sd = load_file(shard)

info = {"name": name, "adapter": str(adir), "scale": scale, "tensors": 0}
num, den, rel = 0.0, 0.0, []
for k in [k for k in ad if ".lora_A." in k]:
    mod = k.split(".lora_A.")[0]
    key = mod.replace("base_model.model.model.", "model.language_model.") + ".weight"
    if key not in sd:
        sys.exit(f"chave não encontrada no checkpoint: {key} (de {k})")
    A = ad[k].float()
    B = ad[k.replace(".lora_A.", ".lora_B.")].float()
    W = sd[key].float()
    D = scale * (B @ A)
    W2 = W + D
    e = (W2.half().float() - W2).norm().item()
    num += e ** 2
    den += D.norm().item() ** 2
    rel.append(D.norm().item() / W.norm().item())
    sd[key] = W2.contiguous()
    info["tensors"] += 1
info["f16_rel_err_of_update"] = (num / den) ** 0.5 if den else None
info["mean_rel_change"] = sum(rel) / len(rel)
info["max_rel_change"] = max(rel)
print(json.dumps(info), flush=True)

save_file(sd, tmp / shard.name, metadata=meta)
del sd
out = DATA / "gguf" / f"{name}.gguf"
subprocess.run([sys.executable, CONVERT, str(tmp), "--outtype", "f16", "--outfile", str(out)], check=True)
shutil.rmtree(tmp)
info["gguf"] = str(out)
(DATA / "gguf" / f"{name}.json").write_text(json.dumps(info, indent=1))
print("ok", out, flush=True)
