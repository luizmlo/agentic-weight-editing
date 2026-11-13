"""Diagrama "anatomia da edição a1" para o pre-paper: (A) o que é uma edição de posto 1, (B) onde a a1 mexe.

Esquemático (sem números inventados): os dados vêm do registro do agente a1 (data/m2a/runs/a1/done/062959_102.pt):
L17n1915, L18n881, L22n573 com u = direção de "Wait" e α = −6 (a parte "Wait" desses neurônios fica 7×), e o MLP
inteiro da L23 com u = direção de </think> e α = −0,8 (1,8×).

Versão de 13/11: tela estreita (7,6 pol.) e poucos elementos, para o texto continuar legível na largura da coluna.
uso: python3 scripts/draw_edit_anatomy.py   (matplotlib do sistema) -> docs/figures/edit-anatomy.png e pt-BR/
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, NAVY, GRAY, LIGHT = "#2a78d6", "#eb6834", "#184f95", "#8a8985", "#e8e7e3"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

TXT = {
    "en": dict(
        A="A · One rank-1 edit", B="B · Where a1 edits",
        u="“Wait” direction u", w="what one neuron\nwrites (w)", wnew="after the edit",
        part="part along u",
        rule="α = −6: the part along u becomes 7×;\nthe rest of the neuron is untouched.",
        layer="layer", first="layer 0", last="layer 23",
        wait_note="one neuron each in layers\n17, 18 and 22: their brake\non “Wait” made 7× stronger",
        think_note="whole MLP of layer 23:\n“</think>” signal 1.8×",
    ),
    "pt-BR": dict(
        A="A · Uma edição de posto 1", B="B · Onde a a1 edita",
        u="direção de “Wait” (u)", w="o que um\nneurônio escreve (w)", wnew="depois da edição",
        part="parte em u",
        rule="α = −6: a parte em u fica 7×;\no resto do neurônio não muda.",
        layer="camada", first="camada 0", last="camada 23",
        wait_note="um neurônio em cada uma das\ncamadas 17, 18 e 22: o freio\ndeles sobre “Wait” fica 7×",
        think_note="MLP inteiro da camada 23:\nsinal de “</think>” 1,8×",
    ),
}


def arrow(ax, p, q, color=INK2, lw=1.4, style="-|>", ms=10, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def draw(lang):
    t = TXT[lang]
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(7.6, 3.7), dpi=250, facecolor=SURF)

    # ---------------- A: uma edição de posto 1 ----------------
    ax = fig.add_axes([0.0, 0.0, 0.47, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.2, 9.6, t["A"], fontsize=12.5, weight="bold", color=INK, va="top")
    ox, oy = 0.8, 3.6
    arrow(ax, (ox - 0.3, oy), (9.8, oy), color=GRAY, lw=1, ms=8)
    ax.text(9.8, oy - 0.35, t["u"], ha="right", va="top", fontsize=10, color=ORANGE, weight="bold")
    wx, wy = 1.1, 3.6
    arrow(ax, (ox, oy), (ox + wx, oy + wy), color=BLUE, lw=2.2, ms=12)
    ax.text(ox + wx + 0.15, oy + wy - 0.1, t["w"], ha="left", va="top", fontsize=10, color=BLUE)
    ax.plot([ox + wx] * 2, [oy, oy + wy], color=GRAY, lw=0.8, ls=":")
    arrow(ax, (ox, oy - 0.08), (ox + wx, oy - 0.08), color=ORANGE, lw=3, style="-", ms=1)
    nx = 7 * wx  # 7× ao longo de u (em escala)
    arrow(ax, (ox, oy), (ox + nx, oy + wy), color=NAVY, lw=2.2, ms=12, ls="--")
    ax.plot([ox + nx] * 2, [oy, oy + wy], color=GRAY, lw=0.8, ls=":")
    arrow(ax, (ox, oy + 0.08), (ox + nx, oy + 0.08), color=ORANGE, lw=1.2, style="-", ms=1, ls="--")
    ax.text(ox + nx, oy + wy + 0.2, t["wnew"], ha="right", va="bottom", fontsize=10, color=NAVY, weight="bold")
    ax.text(ox + 0.1, oy - 0.35, t["part"], ha="left", va="top", fontsize=8.5, color=ORANGE)
    ax.text(0.2, 1.75, "w ← w − α (uᵀw) u", fontsize=11.5, color=INK, family="DejaVu Sans Mono", va="center")
    ax.text(0.2, 0.75, t["rule"], fontsize=9.5, color=INK2, va="center")

    # ---------------- B: onde a a1 edita ----------------
    bx = fig.add_axes([0.5, 0.0, 0.5, 1.0]); bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis("off")
    bx.text(0.2, 9.6, t["B"], fontsize=12.5, weight="bold", color=INK, va="top")
    lx, lw_, ly0, lh = 0.4, 1.5, 0.5, 0.30
    hi = {17: BLUE, 18: BLUE, 22: BLUE, 23: NAVY}
    for k in range(24):
        bx.add_patch(Rectangle((lx, ly0 + k * lh), lw_, lh * 0.78, fc=hi.get(k, LIGHT), ec="none"))
    yc = lambda k: ly0 + k * lh + lh * 0.39
    bx.text(lx + lw_ + 0.15, yc(0), t["first"], va="center", fontsize=8.5, color=INK2)
    for k in hi:
        bx.text(lx + lw_ + 0.15, yc(k), str(k), va="center", fontsize=9, color=hi[k], weight="bold")
    xb = 2.75
    bx.plot([xb, xb], [yc(17), yc(22)], color=BLUE, lw=1.2)
    for k in (17, 18, 22):
        bx.plot([xb - 0.12, xb], [yc(k), yc(k)], color=BLUE, lw=1.2)
    yw = (yc(17) + yc(22)) / 2 - 1.2
    bx.plot([xb, xb + 0.3], [(yc(17) + yc(22)) / 2] * 2, color=BLUE, lw=1.2)
    bx.plot([xb + 0.3, xb + 0.3, xb + 0.45], [(yc(17) + yc(22)) / 2, yw, yw], color=BLUE, lw=1.2)
    bx.text(xb + 0.55, yw, t["wait_note"], va="center", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=BLUE, lw=1))
    yt = yc(23) + 0.35
    bx.plot([xb - 0.12, xb + 0.3, xb + 0.3, xb + 0.45], [yc(23), yc(23), yt, yt], color=NAVY, lw=1.2)
    bx.text(xb + 0.55, yt, t["think_note"], va="center", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=NAVY, lw=1))

    out = ROOT / "docs" / "figures" / ("edit-anatomy.png" if lang == "en" else "pt-BR/edit-anatomy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", out)


for lang in TXT:
    draw(lang)
