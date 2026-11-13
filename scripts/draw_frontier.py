"""Fronteira acerto × tokens no 1º teste bloqueado do GSM8K (test[200:400], t200), todos os braços já rodados.

Acerto e média de tokens: data/gen_t200.json, data/gen_t200nt.json (raciocínio desligado, teto 8k) e
data/gen_t200L.json (Astra com teto 20.480). Acerto entre as respostas terminadas: CADERNO, 13/11 04h15.
uso: python3 scripts/draw_frontier.py (matplotlib do sistema) -> docs/figures/frontier.png e docs/figures/pt-BR/frontier.png
"""
import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data"
GRAY, ORANGE, PURPLE, GREEN, BLUE, NAVY = "#8a8985", "#eb6834", "#6551d6", "#1baf7a", "#2a78d6", "#184f95"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3"


def arm(f, a):
    rows = json.load(open(R / f))["arms"][a]["rows"]
    n = len(rows)
    acc = sum(bool(r["correct"]) for r in rows) / n
    tok = sum(r["n_tokens"] for r in rows) / n
    fin = [r for r in rows if not r["cut"]]
    return acc, math.sqrt(acc * (1 - acc) / n), tok, sum(bool(r["correct"]) for r in fin) / len(fin)


# chave, arquivo, braço, cor, deslocamento do rótulo (pontos), alinhamento
P = [
    ("base", "gen_t200.json", "base", GRAY, (-8, 6), "right"),
    ("off", "gen_t200nt.json", "base", GREEN, (8, -12), "left"),
    ("ft", "gen_t200.json", "ft_sft_ep2", PURPLE, (8, -4), "left"),
    ("ban", "gen_t200.json", "lbx_w10_t1", GREEN, (8, -12), "left"),
    ("bf", "gen_t200.json", "bf4096", GREEN, (0, 14), "center"),
    ("bfa", "gen_t200.json", "bf4096a1024", GREEN, (-6, -16), "right"),
    ("a1n1", "gen_t200.json", "m2a_a1", BLUE, (-8, 8), "right"),
    ("a1", "gen_t200.json", "m2a_a1_062959_102", BLUE, (8, -16), "left"),
    ("a2", "gen_t200.json", "m2a_a2", BLUE, (8, 2), "left"),
    ("a1bfa", "gen_t200.json", "m2a_a1_062959_102+bf4096a1024", NAVY, (-8, 6), "right"),
    ("astra", "gen_t200.json", "astra", ORANGE, (8, -4), "left"),
    ("astraL", "gen_t200L.json", "astra", ORANGE, (-8, 6), "right"),
]
TXT = {
    "en": dict(
        base="Original model", off="Thinking off", ft="Fine-tuning (LoRA)", ban="Word ban", bf="Time limit",
        bfa="Time limit, answer capped", a1n1="a1 agent's first pick", a1="Agent edit (a1)", a2="2nd agent (a2)",
        a1bfa="a1 + time limit, capped", astra="Published edit (Astra), 12k", astraL="Astra, 20k limit",
        unreached="Unreached: more accuracy for fewer tokens.\nThis is where an efficiency-targeted search would aim.",
        callout=("On the frontier, but by thinking less: only {fin} of its\nfinished answers are right "
                 "(time limit: {bf};\nthinking off: {off}). It concludes before it is ready."),
        xlabel="Mean tokens per answer (log scale): the cost of every future answer", ylabel="Share solved",
        title="Agent edits sit near the efficiency frontier without being asked to",
        sub=("First locked test (GSM8K, 200 problems). Dashed line: best accuracy reachable at each cost. "
             "Bars: ±1 standard error."),
        pct=lambda v: f"{v:.0%}", k=["1k", "2k", "4k", "8k", "12k"],
    ),
    "pt-BR": dict(
        base="Modelo original", off="Sem raciocínio", ft="Ajuste fino (LoRA)", ban="Proibir “Wait”",
        bf="Limite de tempo", bfa="Limite, resposta com teto", a1n1="1ª escolha do agente a1",
        a1="Agente a1", a2="Agente a2", a1bfa="a1 + limite, com teto",
        astra="Edição publicada (Astra), 12k", astraL="Astra, teto de 20k",
        unreached=("Não alcançado: mais acerto com menos tokens.\n"
                   "É aqui que uma busca voltada à eficiência miraria."),
        callout=("Na fronteira, mas porque pensa menos: só {fin} das\nrespostas que terminam estão certas "
                 "(limite de tempo: {bf};\nsem raciocínio: {off}). Conclui antes da hora."),
        xlabel="Média de tokens por resposta (escala log): o custo de cada resposta futura",
        ylabel="Problemas resolvidos",
        title="As edições dos agentes ficam perto da fronteira de eficiência sem que isso fosse pedido",
        sub=("1º teste bloqueado (GSM8K, 200 problemas). Tracejado: o melhor acerto alcançável a cada custo. "
             "Barras: ±1 erro-padrão."),
        pct=lambda v: f"{v:.0%}".replace(".", ","), k=["1k", "2k", "4k", "8k", "12k"],
    ),
}
pts = {key: (*arm(f, a), c, off, ha) for key, f, a, c, off, ha in P}


def draw(lang):
    t = TXT[lang]
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.set_xscale("log")

    # fronteira: pontos não dominados (menos tokens e mais acerto)
    s = sorted(pts.values(), key=lambda p: p[2])
    front, best = [], -1
    for p in s:
        if p[0] > best:
            front.append(p); best = p[0]
    fx, fy = [front[0][2]], [front[0][0]]
    for p in front[1:]:
        fx += [p[2], p[2]]; fy += [fy[-1], p[0]]
    fx.append(20000); fy.append(fy[-1])
    ax.fill_between(fx, fy, 1, color="#dfeaf8", alpha=0.6, lw=0, zorder=0)
    ax.plot(fx, fy, color=NAVY, lw=1.2, ls="--", zorder=1)
    ax.text(1150, 0.83, t["unreached"], fontsize=9.5, color=NAVY, va="top")

    a12, a20 = pts["astra"], pts["astraL"]
    ax.annotate("", (a20[2], a20[0]), (a12[2], a12[0]), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))

    for key, (acc, se, tok, fin, c, off, ha) in pts.items():
        ax.errorbar(tok, acc, yerr=se, fmt="o", ms=8, color=c, mec=SURF, mew=2, elinewidth=1, capsize=0, zorder=3)
        ax.annotate(t[key], (tok, acc), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=9,
                    color=INK2)

    # a 1ª escolha da a1: na fronteira, mas conclui cedo demais
    n1 = pts["a1n1"]
    ax.scatter([n1[2]], [n1[0]], s=330, facecolors="none", edgecolors=BLUE, lw=1.3, zorder=4)
    ax.annotate(t["callout"].format(fin=t["pct"](n1[3]), bf=t["pct"](pts["bf"][3]), off=t["pct"](pts["off"][3])), (n1[2], n1[0]),
                xytext=(1700, 0.30), textcoords="data", fontsize=9, color=INK, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=BLUE, lw=1),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1, shrinkB=9, connectionstyle="arc3,rad=0.25"))

    ax.minorticks_off(); ax.set_xlim(1000, 16000); ax.set_ylim(0, 0.85)
    ax.set_xticks([1000, 2000, 4000, 8000, 12000]); ax.set_xticklabels(t["k"])
    ax.set_yticks([0, .2, .4, .6, .8]); ax.set_yticklabels([t["pct"](v) for v in (0, .2, .4, .6, .8)])
    ax.set_xlabel(t["xlabel"], color=INK2)
    ax.set_ylabel(t["ylabel"], color=INK2)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c4")
    ax.tick_params(colors=INK2)
    fig.text(0.07, 0.965, t["title"], fontsize=14 if lang == "en" else 13, weight="bold", color=INK)
    fig.text(0.07, 0.925, t["sub"], fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.1)
    out = ROOT / "docs" / "figures" / ("frontier.png" if lang == "en" else "pt-BR/frontier.png")
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    print(out)


for k, p in pts.items():
    print(f"{k:7s} acc {p[0]:.3f} tok {p[2]:7.0f} finished-correct {p[3]:.2f}")
for lang in TXT:
    draw(lang)
