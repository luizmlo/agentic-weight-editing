"""Re-executa o código de notebooks/results.ipynb e salva as figuras com textos em pt-BR em docs/figures/pt-BR/
(usadas pelo PDF em português). uso: uv run python scripts/export_figures_pt_br.py"""
import json, sys, io, contextlib, re
from pathlib import Path
import plotly.graph_objects as go
sys.path.insert(0, str(Path(__file__).parent))
from figures_pt_br import TR
NAMES = ["defect", "headline", "paired", "why", "difficulty", "goodhart", "atlas", "search", "finetune", "cost",
         "astra-tokens", "side-effects"]
nb = json.load(open("notebooks/results.ipynb"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
out = Path("docs/figures/pt-BR"); out.mkdir(parents=True, exist_ok=True)
missing = set()
def tr(o, key=""):
    if isinstance(o, dict): return {k: (v if k in ("hovertemplate",) else tr(v, k)) for k, v in o.items()}
    if isinstance(o, list): return [tr(v, key) for v in o]
    if isinstance(o, str):
        if o in TR: return TR[o]
        if re.fullmatch(r"\d{1,3}(,\d{3})+", o): return o.replace(",", ".")
        if any(c.isalpha() for c in o) and key in ("text", "name", "ticktext", "x", "y") and not o.startswith(("#", "rgb")):
            missing.add(o)
    return o
n = [0]
def show(fig, width=960):
    d = tr(json.loads(fig.to_json()))
    f = go.Figure(d); f.update_layout(separators=",.")
    f.write_image(out / f"{NAMES[n[0]]}.png", width=width, height=f.layout.height or 420, scale=1.5)
    n[0] += 1
ns = {}
for i, c in enumerate(code):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(c, ns)
    if i == 0: ns["show"] = show
print(n[0], "figuras; sem tradução:", sorted(missing)[:40])
