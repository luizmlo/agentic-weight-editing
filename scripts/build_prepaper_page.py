"""Render docs/paper-draft.md + key notebook figures into a self-contained HTML page for sharing."""
import base64, json, re, html
from pathlib import Path
import mistune

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "prepaper.html")
OUT.parent.mkdir(parents=True, exist_ok=True)

src = (REPO / "docs/paper-draft.md").read_text()

# figures from the executed notebook, in output order
nb = json.load(open(REPO / "notebooks/results.ipynb"))
pngs = [o["data"]["image/png"] for c in nb["cells"] for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
FIG = {  # notebook figure index (1-based) -> caption; inserted after the paragraph/table that starts with the anchor
    1: "F1. Length of each of the 48 dev answers. Base model: 85% hit the 12,288-token cap without answering. Agent edit a1: 17%.",
    2: "T1 (chart). Accuracy per arm on the three evaluation sets (±1 SE).",
    3: "T2 (chart). Paired comparison: problems solved only by a1 (right) vs only by the competitor (left), same problems and seeds.",
    4: "F2. Accuracy vs share of generations cut at the cap, all 20 arms evaluated on dev48 (r = −0.99).",
    6: "F4. Goodhart: the probe score each agent's chosen edit achieved vs its real dev48 accuracy.",
    7: "F5. Atlas vs no-atlas agents, paired by protocol.",
    8: "F6. Agent search timelines: each generation check (16 problems) and the best so far.",
    10: "F8. Cost to produce each fix (GPU-hours) and tokens generated per answer.",
}
ANCHOR = {  # figure -> unique text that ends the block after which the figure goes
    1: "both slightly *raise* the loop rate (0.31 / 0.30 vs 0.23).",
    2: "\\* MATH-500[0:48] only.",
    3: "+15.5 ± 3.6 (27/4).",
    4: "Excised traces teach the model what to write after a cut, not when to cut.",
    6: "Probes are\nuseful as constraints, not as targets.",
    7: "Three pairs make this suggestive, not conclusive.",
    8: "The tools and the generation signal are not sufficient; the agent's choices are what\nmatter.",
    10: "On MATH-148, a1 generates 29% fewer tokens than the base. This saving recurs on every query.",
}
for k, anchor in ANCHOR.items():
    assert src.count(anchor) == 1, anchor
    src = src.replace(anchor, anchor + f"\n\n@@FIG{k}@@\n")

# protect math from markdown
maths = []
def keep(m):
    maths.append(m.group(0)); return f"@@MATH{len(maths)-1}@@"
src = re.sub(r"\$\$.*?\$\$", keep, src, flags=re.S)
src = re.sub(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$", keep, src)

md = mistune.create_markdown(plugins=["table", "strikethrough", "task_lists"], escape=False)
body = md(src)
body = re.sub(r"\$(?=\d)", '<span class="usd">$</span>', body)  # money, so KaTeX doesn't read it as math
for i, m in enumerate(maths):
    body = body.replace(f"@@MATH{i}@@", html.escape(m, quote=False))
for k, cap in FIG.items():
    img = f'<figure><img src="data:image/png;base64,{pngs[k-1]}" alt="{html.escape(cap)}" loading="lazy"><figcaption>{html.escape(cap)}</figcaption></figure>'
    body = re.sub(rf"<p>@@FIG{k}@@</p>", img, body)
assert "@@" not in body, re.findall(r"@@\w+@@", body)[:5]

# split off the H1 + status line into a header; wrap tables for horizontal scroll
h1 = re.search(r"<h1>(.*?)</h1>", body, re.S)
title = h1.group(1)
body = body.replace(h1.group(0), "", 1)
byline = re.search(r"<p><em>(Pre-paper draft.*?)</em></p>", body, re.S)
body = body.replace(byline.group(0), "", 1)
body = re.sub(r"<table>", '<div class="tw"><table>', body).replace("</table>", "</table></div>")
body = body.replace('<li class="task-list-item"><input class="task-list-item-checkbox" type="checkbox" disabled checked/>', '<li class="task done">')
body = body.replace('<li class="task-list-item"><input class="task-list-item-checkbox" type="checkbox" disabled/>', '<li class="task">')

# table of contents from h2
toc, n = [], 0
def anchor_h2(m):
    global n
    n += 1; t = re.sub("<.*?>", "", m.group(1)); slug = f"s{n}"
    toc.append((slug, t)); return f'<h2 id="{slug}">{m.group(1)}</h2>'
body = re.sub(r"<h2>(.*?)</h2>", anchor_h2, body)
toc_html = "".join(f'<li><a href="#{s}">{html.escape(t)}</a></li>' for s, t in toc)

page = f"""<title>Agents as Weight-Space Programmers</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,600;1,7..72,400&family=IBM+Plex+Sans:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single preprint column (~68ch) with a sticky contents rail on wide screens. */
:root {{
  --paper: #fbfbfa; --ink: #16181d; --ink-2: #4b4f58; --rule: #dcdde1; --tint: #eef3fa; --accent: #1f5fae; --fig: #fcfcfb;
  --serif: "Literata", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #15171b; --ink: #e9eaee; --ink-2: #a9adb7; --rule: #30333a; --tint: #1c2533; --accent: #7fb0ee; --fig: #fcfcfb; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --paper: #15171b; --ink: #e9eaee; --ink-2: #a9adb7; --rule: #30333a; --tint: #1c2533; --accent: #7fb0ee; --fig: #fcfcfb; color-scheme: dark; }}
body {{ background: var(--paper); color: var(--ink); font: 400 17px/1.62 var(--serif); }}
.wrap {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 48px; max-width: 1120px; margin: 0 auto;
  padding-block: 40px 96px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
@media (min-width: 1000px) {{ .wrap {{ grid-template-columns: 220px minmax(0, 1fr); }} }}
nav {{ display: none; }}
@media (min-width: 1000px) {{ nav {{ display: block; position: sticky; top: calc(24px + env(safe-area-inset-top, 0px)); align-self: start; font: 500 13px/1.4 var(--sans); }}
  nav ol {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }}
  nav a {{ color: var(--ink-2); text-decoration: none; }} nav a:hover {{ color: var(--accent); }}
  nav .lab {{ text-transform: uppercase; letter-spacing: .08em; font-size: 11px; color: var(--ink-2); margin-bottom: 10px; }} }}
main {{ min-width: 0; max-width: 72ch; }}
header {{ border-bottom: 1px solid var(--rule); padding-bottom: 20px; margin-bottom: 8px; }}
.kicker {{ font: 600 12px/1 var(--sans); letter-spacing: .1em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 600 clamp(26px, 4vw, 36px)/1.18 var(--serif); margin: 12px 0 10px; text-wrap: balance; }}
.byline {{ color: var(--ink-2); font: 400 14px/1.5 var(--sans); margin: 0; }}
h2 {{ font: 600 22px/1.3 var(--sans); margin: 48px 0 12px; text-wrap: balance; scroll-margin-top: 24px; }}
h3 {{ font: 600 17px/1.35 var(--sans); margin: 28px 0 8px; }}
p, li {{ hyphens: auto; }} a {{ color: var(--accent); }}
strong {{ font-weight: 600; }}
code {{ font: 400 .86em var(--mono); background: var(--tint); padding: .08em .3em; border-radius: 3px; overflow-wrap: anywhere; }}
blockquote {{ margin: 20px 0; padding: 14px 18px; background: var(--tint); border-radius: 6px; font: 400 15px/1.55 var(--sans); }}
blockquote p {{ margin: 0; }} blockquote ul {{ margin: 8px 0 0; padding-left: 20px; }}
ul, ol {{ padding-left: 22px; }} li + li {{ margin-top: 4px; }}
li.task {{ list-style: none; margin-left: -22px; padding-left: 26px; position: relative; }}
li.task::before {{ content: ""; position: absolute; left: 2px; top: .45em; width: 12px; height: 12px; border: 1.5px solid var(--ink-2); border-radius: 3px; }}
li.task.done::before {{ background: var(--accent); border-color: var(--accent); }}
.tw {{ overflow-x: auto; margin: 16px 0; }}
table {{ border-collapse: collapse; font: 400 14px/1.45 var(--sans); font-variant-numeric: tabular-nums; min-width: 100%; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ font-weight: 600; color: var(--ink-2); font-size: 12.5px; letter-spacing: .02em; border-bottom: 1.5px solid var(--ink-2); }}
figure {{ margin: 24px 0; }}
figure img {{ display: block; width: 100%; height: auto; background: var(--fig); border: 1px solid var(--rule); border-radius: 6px; }}
figcaption {{ font: 400 13.5px/1.5 var(--sans); color: var(--ink-2); margin-top: 8px; }}
hr {{ border: 0; border-top: 1px solid var(--rule); margin: 48px 0; }}
.katex-display {{ overflow-x: auto; overflow-y: hidden; padding: 4px 0; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
<div class="wrap">
  <nav aria-label="Contents"><div class="lab">Contents</div><ol>{toc_html}</ol></nav>
  <main>
    <header>
      <div class="kicker">Pre-paper · working draft</div>
      <h1>{title}</h1>
      <p class="byline">{byline.group(1)}</p>
    </header>
    {body}
  </main>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<script>
try {{ renderMathInElement(document.querySelector("main"), {{ delimiters: [{{left: "$$", right: "$$", display: true}}, {{left: "$", right: "$", display: false}}], throwOnError: false, output: "mathml", ignoredClasses: ["usd"] }}); }} catch (e) {{}}
</script>
"""
OUT.write_text(page)
print(OUT, len(page) // 1024, "KB", "figs", len(FIG), "toc", len(toc))
