"""Diagrama "anatomia da edição a1" para o pre-paper: (A) o que é uma edição de posto 1, (B) onde a a1 mexe.

Esquemático (sem números inventados): os dados vêm do registro do agente a1 (data/m2a/runs/a1/done/062959_102.pt):
L17n1915, L18n881, L22n573 com u = direção de "Wait" e α = −6 (a parte "Wait" desses neurônios fica 7×), e o MLP
inteiro da L23 com u = direção de </think> e α = −0,8 (1,8×).

uso: python3 scripts/draw_edit_anatomy.py   (matplotlib do sistema) -> docs/figures/edit-anatomy.png e pt-BR/
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, NAVY, GRAY, LIGHT = "#2a78d6", "#eb6834", "#184f95", "#8a8985", "#e8e7e3"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

TXT = {
    "en": dict(
        A="A · One rank-1 edit", B="B · The a1 edit: 4 rank-1 edits",
        matrix="output matrix of one layer's MLP\n1024 rows × 3,584 columns",
        neuron="one column =\none neuron\n(what it writes)",
        u="direction u\n(pushes toward “Wait”)", w="neuron's\nwrite w", wnew="after the edit",
        rule="w ← w − α (uᵀw) u\nα = −6: the part of w along u becomes 7×;\nthe rest of the neuron is untouched.\nThese neurons fire negative while thinking,\nso their brake on “Wait” gets 7× stronger.",
        stream="residual stream\n(the model's\nworking memory)", inp="text read so far", out="next word",
        layer="layer", wait_note="3 neurons that brake “Wait”\n(L17 n1915, L18 n881, L22 n573):\nbrake made 7× stronger",
        think_note="whole MLP of layer 23:\n“</think>” signal 1.8×",
        effect="“Wait” less likely  ·  “</think>” more likely",
        result="The loop loses its way back (“Wait, let me check…”) and gains an exit (end thinking, answer).",
    ),
    "pt-BR": dict(
        A="A · Uma edição de posto 1", B="B · A edição a1: 4 edições de posto 1",
        matrix="matriz de saída do MLP de uma camada\n1024 linhas × 3.584 colunas",
        neuron="uma coluna =\num neurônio\n(o que ele escreve)",
        u="direção u\n(empurra para “Wait”)", w="o que o\nneurônio\nescreve (w)", wnew="depois da edição",
        rule="w ← w − α (uᵀw) u\nα = −6: a parte de w na direção u fica 7×;\no resto do neurônio não muda.\nEsses neurônios disparam negativos no raciocínio,\nentão o freio deles sobre “Wait” fica 7× mais forte.",
        stream="fluxo residual\n(a memória de\ntrabalho do modelo)", inp="texto lido até aqui", out="próxima palavra",
        layer="camada", wait_note="3 neurônios que freiam “Wait”\n(L17 n1915, L18 n881, L22 n573):\nfreio 7× mais forte",
        think_note="MLP inteiro da camada 23:\nsinal de “</think>” 1,8×",
        effect="“Wait” menos provável  ·  “</think>” mais provável",
        result="O loop perde o caminho de volta (“Espera, deixa eu conferir…”) e ganha uma saída (fechar o raciocínio e responder).",
    ),
}


def arrow(ax, p, q, color=INK2, lw=1.4, style="-|>", ms=10, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def draw(lang):
    t = TXT[lang]
    fig = plt.figure(figsize=(11, 6.4), dpi=200, facecolor=SURF)
    plt.rcParams["font.family"] = "DejaVu Sans"

    # ---------------- A: rank-1 edit ----------------
    ax = fig.add_axes([0.02, 0.10, 0.45, 0.80]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 9.75, t["A"], fontsize=16.9, weight="bold", color=INK, va="top")
    # matrix
    x0, y0, W, H, ncol = 0.3, 4.2, 4.2, 3.2, 28
    ax.add_patch(Rectangle((x0, y0), W, H, fc=LIGHT, ec=GRAY, lw=1))
    for i in range(1, ncol):
        ax.plot([x0 + i * W / ncol] * 2, [y0, y0 + H], color="white", lw=0.8)
    hc = 17
    ax.add_patch(Rectangle((x0 + hc * W / ncol, y0), W / ncol, H, fc=BLUE, ec=BLUE))
    ax.text(x0 + W / 2, y0 - 0.25, t["matrix"], ha="center", va="top", fontsize=11.1, color=INK2)
    ax.text(x0 + (hc + 0.5) * W / ncol, y0 + H + 0.15, t["neuron"], ha="center", va="bottom", fontsize=11.1, color=BLUE)
    # vector picture
    ox, oy = 5.9, 4.4
    arrow(ax, (ox - 0.2, oy), (9.9, oy), color=GRAY, lw=1, ms=8)           # u axis
    ax.text(ox + 0.1, oy - 0.3, t["u"], ha="left", va="top", fontsize=11.1, color=ORANGE)
    wx, wy = 1.0, 2.6                                                    # w = (along u, orthogonal)
    arrow(ax, (ox, oy), (ox + wx, oy + wy), color=BLUE, lw=2)
    ax.text(ox + wx - 0.45, oy + wy + 0.1, t["w"], ha="right", va="bottom", fontsize=11.1, color=BLUE)
    ax.plot([ox + wx, ox + wx], [oy, oy + wy], color=GRAY, lw=0.8, ls=":")
    arrow(ax, (ox, oy - 0.05), (ox + wx, oy - 0.05), color=ORANGE, lw=2.4, style="-", ms=1)
    nx = 7 * wx * 0.5                                                    # 7× along u (drawn at half scale)
    arrow(ax, (ox, oy), (ox + nx, oy + wy), color=NAVY, lw=2, ls="--")
    ax.plot([ox + nx, ox + nx], [oy, oy + wy], color=GRAY, lw=0.8, ls=":")
    arrow(ax, (ox, oy + 0.05), (ox + nx, oy + 0.05), color=ORANGE, lw=1, style="-", ms=1, ls="--")
    ax.text(ox + nx, oy + wy + 0.15, t["wnew"], ha="right", va="bottom", fontsize=11.1, color=NAVY)
    ax.text(0.3, 2.6, t["rule"], ha="left", va="top", fontsize=11.7, color=INK, family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=LIGHT))

    # ---------------- B: where a1 edits ----------------
    bx = fig.add_axes([0.52, 0.10, 0.47, 0.80]); bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis("off")
    bx.text(0, 9.75, t["B"], fontsize=16.9, weight="bold", color=INK, va="top")
    lx, lw_, ly0, lh = 2.3, 1.7, 1.1, 0.25
    hi = {17: BLUE, 18: BLUE, 22: BLUE, 23: NAVY}
    for k in range(24):
        y = ly0 + k * lh
        bx.add_patch(Rectangle((lx, y), lw_, lh * 0.82, fc=hi.get(k, LIGHT), ec="none"))
        if k in (0, 23) or k in hi:
            bx.text(lx + lw_ + 0.12, y + lh * 0.4, f"{t['layer']} {k}", va="center", fontsize=9.8,
                    color=hi.get(k, INK2), weight="bold" if k in hi else "normal")
    top = ly0 + 24 * lh
    arrow(bx, (1.5, ly0 - 0.1), (1.5, top + 0.1), color=GRAY, lw=1.6)
    bx.text(1.25, (ly0 + top) / 2, t["stream"].replace("\n", " "), ha="right", va="center", fontsize=10.4, color=INK2, rotation=90)
    bx.text(lx + lw_ / 2, ly0 - 0.2, t["inp"], ha="center", va="top", fontsize=11.1, color=INK2)
    bx.text(lx + lw_ / 2, top + 0.6, t["out"], ha="center", va="bottom", fontsize=11.1, color=INK2)
    arrow(bx, (lx + lw_ / 2, top + 0.05), (lx + lw_ / 2, top + 0.55), color=GRAY, lw=1.2, ms=8)
    # callouts: a bracket over the 3 "Wait" layers, a leader line for layer 23
    yc = lambda k: ly0 + k * lh + lh * 0.4
    xb = 6.15
    bx.plot([xb, xb], [yc(17), yc(22)], color=BLUE, lw=1.2)
    for k in (17, 18, 22):
        bx.plot([xb - 0.12, xb], [yc(k), yc(k)], color=BLUE, lw=1.2)
    yw = (yc(17) + yc(22)) / 2
    bx.plot([xb, xb + 0.25], [yw, yw], color=BLUE, lw=1.2)
    bx.text(xb + 0.3, yw, t["wait_note"], va="center", fontsize=11.1, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=BLUE, lw=1))
    yt = top + 0.3
    bx.plot([xb - 0.12, xb, xb, xb + 0.25], [yc(23), yc(23), yt, yt], color=NAVY, lw=1.2)
    bx.text(xb + 0.3, yt, t["think_note"], va="center", fontsize=11.1, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=NAVY, lw=1))
    bx.text(0, 8.85, t["effect"], ha="left", va="top", fontsize=12.3, color=INK2, weight="bold")

    fig.text(0.02, 0.06, t["result"], fontsize=13.0, color=INK, weight="bold")
    out = ROOT / "docs" / "figures" / ("edit-anatomy.png" if lang == "en" else "pt-BR/edit-anatomy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", out)


for lang in TXT:
    draw(lang)
