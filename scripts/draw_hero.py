"""Figura de abertura: antes / o agente edita os pesos / depois.

Exemplo real: GSM8K train[14] (problema 14 do dev48), original × a2 reduzida (m2a_a2_red), mesma semente.
Trechos literais de data/gen_dev48/{base,m2a_a2_red}.jsonl (sem os $ do LaTeX; "Method B" encurtado).
Números: original no 1º teste bloqueado (t200: 15,5%, 84,5% cortados); a2 reduzida e raciocínio desligado no 2º
(t400: 67,5%, 12% cortados; 58,0%).
uso: python3 scripts/draw_hero.py -> docs/figures/hero.png e docs/figures/pt-BR/hero.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
GRAY, GREEN, BLUE, NAVY, RED = "#8a8985", "#1baf7a", "#2a78d6", "#184f95", "#c8402f"
INK, INK2, SURF, CARD, LIGHT = "#0b0b0b", "#52514e", "#fcfcfb", "#ffffff", "#e8e7e3"
MONO = "DejaVu Sans Mono"

TXT = {
    "en": dict(
        title="A Claude agent repairs a thinking loop by editing the model's weights directly",
        problem="Problem: “Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?”",
        before_h="BEFORE · original Qwen3.5-0.8B",
        before_note="… “Wait, 120 / 8 = 15.” 328 times",
        before_bar="✗  cut off at 12,288 tokens · no answer",
        before_stat="Locked test: 15.5% solved",
        before_sub="85% of answers never finish thinking",
        mid_h="THE AGENT EDITS THE WEIGHTS",
        agent="Claude agent (Claude Code)",
        steps=[("read the weights", "which neurons push toward “Wait”?"),
               ("propose a rank-1 edit", "W ← W − α·u (uᵀW)"),
               ("test it", "quick probes + 16 practice problems")],
        repeat="repeat",
        m1="layer 16: one neuron", m2="layer 22: whole layer",
        m1b="flips its “Wait” push", m2b="boosts “</think>”",
        mid_stat="2 rank-1 edits",
        mid_sub="no training data, no gradient descent",
        after_h="AFTER · same model, edited weights",
        after_note="← checks once,\n    stops, answers",
        after_bar="✓  correct answer in 1,299 tokens",
        after_stat="Locked test: 67.5% solved",
        after_sub="12% cut off · thinking off: 58%",
        after_lines=["Method B: pages per minute", "  8 / 20 = 0.4 pages/min", "  120 / 0.4 = 300 minutes",
                     "  300 / 60 = 5 hours.", "Format the output.", "</think>", "",
                     "Hours = 300 / 60 = 5 hours", "", "\\boxed{5}"],
    ),
    "pt-BR": dict(
        title="Um agente Claude conserta um loop de raciocínio editando os pesos do modelo diretamente",
        problem="Problema: “Joy lê 8 páginas de um livro em 20 minutos. Quantas horas ela leva para ler 120 páginas?”",
        before_h="ANTES · Qwen3.5-0.8B original",
        before_note="… “Wait, 120 / 8 = 15.” 328 vezes",
        before_bar="✗  cortado em 12.288 tokens · sem resposta",
        before_stat="Teste bloqueado: 15,5% resolvidos",
        before_sub="85% das respostas não terminam de pensar",
        mid_h="O AGENTE EDITA OS PESOS",
        agent="Agente Claude (Claude Code)",
        steps=[("lê os pesos", "que neurônios empurram “Wait”?"),
               ("propõe uma edição de posto 1", "W ← W − α·u (uᵀW)"),
               ("testa", "sondas rápidas + 16 problemas de treino")],
        repeat="repete",
        m1="camada 16: 1 neurônio", m2="camada 22: inteira",
        m1b="inverte seu “Wait”", m2b="reforça “</think>”",
        mid_stat="2 edições de posto 1",
        mid_sub="sem dados de treino, sem gradiente",
        after_h="DEPOIS · mesmo modelo, pesos editados",
        after_note="← confere uma vez,\n    para e responde",
        after_bar="✓  resposta certa em 1.299 tokens",
        after_stat="Teste bloqueado: 67,5% resolvidos",
        after_sub="12% cortados · sem raciocínio: 58%",
    ),
}
TXT["pt-BR"]["after_lines"] = TXT["en"]["after_lines"]  # o texto do modelo fica no original
BEFORE = ["300 / 60 = 5 hours.", "This confirms 5 hours again.", "Wait, why did I calculate 15", "  earlier?",
          "Wait, 120 / 8 = 15.", "Wait, 120 / 8 = 15.", "Wait, 120 / 8 = 15.", "Wait, 120 / 8 = 15.",
          "Wait, 120 / 8 = 15.", "Wait, 120 / 8 = 15."]


def draw(lang):
    t = TXT[lang]
    fig = plt.figure(figsize=(12, 5.6), dpi=200, facecolor=SURF)
    plt.rcParams["font.family"] = "DejaVu Sans"
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 120); ax.set_ylim(0, 56); ax.axis("off")

    def card(x, y, w, h, edge, lw=1.4, fc=CARD):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=edge, lw=lw))

    def arrow(p, q, color=INK2, lw=1.6, ms=14, cs="arc3"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                     connectionstyle=cs))

    ax.text(3, 53, t["title"], fontsize=15, weight="bold", color=INK, va="center")
    ax.text(3, 49.2, t["problem"], fontsize=10.5, color=INK2, va="center", style="italic")

    # antes
    ax.text(3, 45, t["before_h"], fontsize=10.5, weight="bold", color=GRAY)
    card(3, 9, 36, 34, GRAY)
    for k, line in enumerate(BEFORE):
        a = 1.0 if k < 7 else 1.0 - (k - 6) * 0.24
        ax.text(5, 40.5 - k * 2.45, line, fontsize=9.2, family=MONO, color=INK, alpha=a, va="center")
    ax.text(21, 14.2, t["before_note"], fontsize=9, color=INK2, ha="center", style="italic")
    ax.add_patch(Rectangle((3.6, 9.6), 34.8, 1.8, fc="#fbe3df", ec="none"))
    ax.text(21, 10.5, t["before_bar"], fontsize=9.5, color=RED, ha="center", va="center", weight="bold")
    ax.text(21, 5.6, t["before_stat"], fontsize=11, ha="center", color=INK, weight="bold")
    ax.text(21, 2.8, t["before_sub"], fontsize=9, ha="center", color=INK2)

    # agente
    ax.text(60, 45, t["mid_h"], fontsize=10.5, weight="bold", color=BLUE, ha="center")
    card(43, 27, 34, 16, BLUE, lw=1.6, fc="#f2f7fd")
    ax.text(60, 40.2, t["agent"], fontsize=10.5, weight="bold", color=NAVY, ha="center")
    for k, (a, b) in enumerate(t["steps"]):
        y = 36.6 - k * 3.4
        ax.add_patch(plt.Circle((46.2, y), 1.05, fc=BLUE, ec="none"))
        ax.text(46.2, y, str(k + 1), fontsize=8, color="white", ha="center", va="center", weight="bold")
        ax.text(48.2, y + 0.55, a, fontsize=9.3, color=INK, va="center", weight="bold")
        ax.text(48.2, y - 1.05, b, fontsize=8.2, color=INK2, va="center", family=MONO if "←" in b else None)
    arrow((75, 29.5), (75, 37.2), color=BLUE, lw=1.3, ms=10, cs="arc3,rad=0.6")
    ax.text(77.7, 33.3, t["repeat"], fontsize=7.8, color=BLUE, rotation=90, va="center")

    # duas matrizes: L16 (uma coluna) e L22 (todas as colunas, ao longo de uma direção)
    nc, mw, mh, my = 14, 12.5, 8.5, 13.6
    for k, (mx, lab, sub) in enumerate([(45.5, t["m1"], t["m1b"]), (62, t["m2"], t["m2b"])]):
        ax.add_patch(Rectangle((mx, my), mw, mh, fc=LIGHT, ec=GRAY, lw=1))
        if k == 1:
            ax.add_patch(Rectangle((mx, my), mw, mh, fc=BLUE, alpha=0.38, ec="none"))
        for c in range(1, nc):
            ax.plot([mx + c * mw / nc] * 2, [my, my + mh], color=SURF, lw=0.8)
        if k == 0:
            ax.add_patch(Rectangle((mx + 8 * mw / nc, my), mw / nc, mh, fc=BLUE, ec="none"))
        ax.text(mx + mw / 2, my + mh + 1.3, lab, fontsize=8, color=INK, ha="center", weight="bold")
        ax.text(mx + mw / 2, my - 1.3, sub, fontsize=7.8, color=INK2, ha="center")
    arrow((60, 26.6), (60, 24.6), color=BLUE)
    ax.text(60, 5.6, t["mid_stat"], fontsize=11, ha="center", color=INK, weight="bold")
    ax.text(60, 2.8, t["mid_sub"], fontsize=9, ha="center", color=INK2)

    arrow((39.6, 26), (42.4, 26), color=GRAY, lw=2, ms=16)
    arrow((77.6, 26), (80.4, 26), color=BLUE, lw=2, ms=16)

    # depois
    ax.text(81, 45, t["after_h"], fontsize=10.5, weight="bold", color=GREEN)
    card(81, 9, 36, 34, GREEN)
    for k, line in enumerate(t["after_lines"]):
        c = BLUE if line == "</think>" else INK
        ax.text(83, 40.5 - k * 2.45, line, fontsize=9.2, family=MONO, color=c, va="center",
                weight="bold" if line in ("</think>", "\\boxed{5}") else "normal")
    ax.text(100.5, 28.25, t["after_note"], fontsize=8.5, color=BLUE, va="center")
    ax.add_patch(Rectangle((81.6, 9.6), 34.8, 1.8, fc="#ddf3ea", ec="none"))
    ax.text(99, 10.5, t["after_bar"], fontsize=9.5, color="#117a55", ha="center", va="center", weight="bold")
    ax.text(99, 5.6, t["after_stat"], fontsize=11, ha="center", color=INK, weight="bold")
    ax.text(99, 2.8, t["after_sub"], fontsize=9, ha="center", color=INK2)

    out = ROOT / "docs" / "figures" / ("hero.png" if lang == "en" else "pt-BR/hero.png")
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    print(out)


for lang in TXT:
    draw(lang)
