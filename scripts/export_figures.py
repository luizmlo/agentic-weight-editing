"""Save the static PNG of every chart in notebooks/results.ipynb to docs/figures/<name>.png (run after executing it)."""
import base64, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["defect", "headline", "paired", "why", "difficulty", "goodhart", "atlas", "search", "finetune", "cost",
         "astra-tokens", "side-effects"]  # notebook chart order
nb = json.load(open(ROOT / "notebooks" / "results.ipynb"))
pngs = [o["data"]["image/png"] for c in nb["cells"] for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
assert len(pngs) == len(NAMES), (len(pngs), len(NAMES))
out = ROOT / "docs" / "figures"
out.mkdir(exist_ok=True)
for name, b64 in zip(NAMES, pngs):
    (out / f"{name}.png").write_bytes(base64.b64decode(b64))
print("wrote", len(pngs), "figures to", out)
