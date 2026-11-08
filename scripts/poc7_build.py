"""PoC-7b: materializar uma edição rank-1 como checkpoint e converter para GGUF (para o llama-server).

uso: poc7_build.py <nome> [arquivo.pt | data/m2a/runs/<run>/done/<id>.pt]
     poc7_build.py <método> [arquivo.pt]   método de data/poc7_dirs.pt (ou do arquivo dado,
                                           ex. poc8_cells.pt)  -> data/gguf/<método>.gguf
     poc7_build.py astra_check   reconstrói o Astra a partir de base + edit_dirs.pt, compara com o
                                 checkpoint publicado e converte (testa a cadeia inteira).

A conta é a mesma de modelio.apply_rank1, feita em fp32 sobre os pesos do safetensors original
(bf16) e gravada de volta no dtype original. O snapshot temporário é apagado depois da conversão.
"""

import json
import shutil
import subprocess
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from wsp.paths import BASE_REPO, BASE_REV, DATA, EDITED_REPO, safetensors_file, snapshot

CONVERT = "/mnt/dados/dev/refs/llama.cpp/convert_hf_to_gguf.py"
name = sys.argv[1]
src_pt = sys.argv[2] if len(sys.argv) > 2 else "poc7_dirs.pt"
if name == "astra_check":
    dirs = torch.load(DATA / "edit_dirs.pt")
elif src_pt.endswith(".pt") and "/" in src_pt:  # candidato do M→A: runs/<run>/done/<id>.pt {"edits": [...]}
    dirs = dict(enumerate(torch.load(src_pt)["edits"]))
else:
    dirs = torch.load(DATA / src_pt)[name]

src = snapshot(BASE_REPO, BASE_REV)
shard = safetensors_file(BASE_REPO, BASE_REV)
tmp = DATA / "snap" / name
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
for f in src.iterdir():
    if f.name != shard.name:
        shutil.copy(f, tmp / f.name)  # segue o symlink do cache do HF

with safe_open(shard, "pt") as fh:
    meta = fh.metadata()
sd = load_file(shard)
info = {"method": name, "layers": {}}
for L, e in dirs.items():
    key = e["tensor"]
    W = sd[key].float()
    u = e["u"].float()
    u = u / u.norm()
    proj = u @ W
    mask = torch.zeros_like(proj)
    mask[torch.as_tensor(e["cols"]) if e["cols"] is not None else slice(None)] = 1.0
    W2 = W - float(e["alpha"]) * torch.outer(u, proj * mask)
    sd[key] = W2.to(sd[key].dtype).contiguous()
    info["layers"][L] = {"tensor": key, "alpha": float(e["alpha"]), "n_cols": None if e["cols"] is None else int(len(e["cols"])),
                         "rel_change": ((W2 - W).norm() / W.norm()).item()}

if name == "astra_check":
    with safe_open(safetensors_file(EDITED_REPO), "pt") as ref:
        for L, e in dirs.items():
            k = e["tensor"]
            r = ref.get_tensor(k).float()
            info["layers"][L]["vs_published_max_abs"] = (sd[k].float() - r).abs().max().item()
            info["layers"][L]["vs_published_rel"] = ((sd[k].float() - r).norm() / r.norm()).item()

save_file(sd, tmp / shard.name, metadata=meta)
del sd
out = DATA / "gguf" / f"{name}.gguf"
out.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, CONVERT, str(tmp), "--outtype", "f16", "--outfile", str(out)], check=True)
shutil.rmtree(tmp)
info["gguf"] = str(out)
info["gguf_bytes"] = out.stat().st_size
(DATA / "gguf" / f"{name}.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
