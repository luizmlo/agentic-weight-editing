"""Re-apply the rank-1 edits to the original Qwen3.5-0.8B weights (no pickle, no project code needed).

usage: python apply_edits.py <original_model_dir> <output_dir>
  original_model_dir: a local copy of Qwen/Qwen3.5-0.8B (e.g. `hf download Qwen/Qwen3.5-0.8B --local-dir orig`)
  output_dir:         receives the edited model (all other files copied unchanged)

Each edit changes one output projection W (d x k): W <- W - alpha * outer(u, mask * (u @ W)), computed in fp32 and
cast back to the original dtype. u is a unit vector in the residual stream; mask selects the edited columns
(neurons), or all of them. Requires torch and safetensors.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

here = Path(__file__).resolve().parent
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
spec = json.loads((here / "edits.json").read_text())
vec = load_file(here / "edits.safetensors")

dst.mkdir(parents=True, exist_ok=True)
shards = sorted(src.glob("*.safetensors"))
for f in src.iterdir():
    if f.is_file() and f not in shards:
        shutil.copy(f, dst / f.name)
for shard in shards:
    with safe_open(shard, "pt") as fh:
        meta = fh.metadata()
    sd = load_file(shard)
    for e in spec["edits"]:
        if e["tensor"] not in sd:
            continue
        W = sd[e["tensor"]].float()
        u = vec[f"edit{e['index']}.u"].float()
        proj = u @ W
        mask = torch.zeros_like(proj)
        mask[vec[f"edit{e['index']}.cols"] if e["cols"] is not None else slice(None)] = 1.0
        sd[e["tensor"]] = (W - e["alpha"] * torch.outer(u, proj * mask)).to(sd[e["tensor"]].dtype).contiguous()
        print("edited", e["tensor"])
    save_file(sd, dst / shard.name, metadata=meta)
print("done:", dst)
