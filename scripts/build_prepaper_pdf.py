"""Typeset docs/paper-draft.md as a PDF with pandoc + pdfLaTeX (floating figures, proper page breaks).

usage: uv run python scripts/build_prepaper_pdf.py [--lang en|pt-BR] [--print] [out.pdf]
       --print: review copy for paper (wide right margin for notes, line numbers, "page X of Y", a final notes page)
                -> docs/pre-paper[.pt-BR].print.pdf
       (default: docs/pre-paper.pdf from docs/paper-draft.md; pt-BR: docs/pre-paper.pt-BR.pdf from docs/paper-draft.pt-BR.md)
needs: uv sync --group analysis (bundled pandoc), pdflatex with tcolorbox/titlesec/caption/newunicodechar, tgpagella, tgheros, mathpazo.
Figures come from docs/figures/ (scripts/export_figures.py).
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
LANG = args.pop(args.index("--lang") + 1) if "--lang" in args else "en"
args = [a for a in args if a != "--lang"]
PRINT = "--print" in args
args = [a for a in args if a != "--print"]
L = {  # every language-specific marker the markdown uses, and the labels typeset around it
    "en": dict(src="paper-draft.md", out="pre-paper.pdf", authors="Authors:", note="Working draft", fig="Figure",
               exp="Experiment", setup="Setup", after_tldr="Everything ran on one gaming", pandoc_lang="en",
               kicker="PRE-PAPER · WORKING DRAFT", header="Agentic Weight-Space Programming · pre-paper draft",
               figdir="figures", page="page", of="of", notes="Notes, questions and suggestions"),
    "pt-BR": dict(src="paper-draft.pt-BR.md", out="pre-paper.pt-BR.pdf", authors="Autores:", note="Rascunho de trabalho",
                  fig="Figura", exp="Experimento", setup="Configuração", after_tldr="Tudo rodou em uma única placa",
                  pandoc_lang="pt-BR", kicker="PRÉ-PAPER · RASCUNHO",
                  header="Programação Agêntica no Espaço de Pesos · rascunho",
                  figdir="figures/pt-BR", page="página", of="de", notes="Anotações, dúvidas e sugestões"),
}[LANG]
OUT = Path(args[0]).resolve() if args else ROOT / "docs" / (L["out"].replace(".pdf", ".print.pdf") if PRINT else L["out"])


def figpath(rel):  # translated copy of the figure when there is one (docs/figures/pt-BR/, scripts/export_figures_pt_br.py)
    alt = ROOT / "docs" / L["figdir"] / Path(rel).name
    return alt if alt.exists() else ROOT / "docs" / rel
md = (ROOT / "docs" / L["src"]).read_text()

# --- title block: taken out of the body, typeset by hand
title = re.search(r"^# (.+)$", md, re.M).group(1)
authors = re.search(rf"^\*\*{L['authors']}\*\* (.+)$", md, re.M).group(1)
note = re.search(rf"^\*({L['note']}.+?)\*$", md, re.M | re.S).group(1).replace("\n", " ")
md = md[md.index("**TL;DR**"):]

# --- figures: image + "**Figure N:** caption" paragraph -> one captioned float
md, nfig = re.subn(rf"!\[[^\]]*\]\((figures/[\w-]+\.png)\)\n\n\*\*{L['fig']} \d+:\*\* (.+?)\n\n",
                   lambda m: f"![{m.group(2).replace(chr(10), ' ')}]({figpath(m.group(1))}){{width=94%}}\n\n", md, flags=re.S)
# --- "Experiment N" labels, TL;DR box, money signs, decorative rules
md = re.sub(rf"^\*{L['exp']} (\d+)\*\n\n### (.+)$", rf"### \\texorpdfstring{{\\expkicker{{{L['exp'].upper()} \1}}}}{{}} \2", md, flags=re.M)
assert f"*{L['exp']}" not in md
md = md.replace("**TL;DR**\n\n", "\\tldrbegin\n\n", 1)
md = md.replace(f"\n\n{L['after_tldr']}", f"\n\n\\tldrend\n\n{L['after_tldr']}", 1)
assert "\\tldrend" in md
md = re.sub(r"(?<![\\$])\$(?=\d)", r"\\$", md)
md = re.sub(r"^---$", "", md, flags=re.M)
md = re.sub(r"^(## (References|Referências).*)$", r"\1\n\n\\begingroup\\small", md, flags=re.M) + "\n\n\\endgroup\n"
md = re.sub(rf"^\*\*{L['setup']}\*\*$", r"\\setuplabel", md, flags=re.M)  # keep "Setup" with its list

HEADER = r"""
\newcommand{\setupname}{@SETUP@}
\usepackage{etoolbox}
\usepackage{mathpazo}\renewcommand{\rmdefault}{qpl}\usepackage[scale=0.9]{tgheros}
\usepackage{newunicodechar}
\newunicodechar{−}{\ensuremath{-}}\newunicodechar{≤}{\ensuremath{\leq}}\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≈}{\ensuremath{\approx}}\newunicodechar{×}{\ensuremath{\times}}\newunicodechar{±}{\ensuremath{\pm}}
\usepackage{xcolor}
\definecolor{accent}{HTML}{1F5FAE}\definecolor{ink2}{HTML}{4B4F58}\definecolor{tint}{HTML}{EEF3FA}
\usepackage[most]{tcolorbox}
\newcommand{\tldrbegin}{\begin{tcolorbox}[enhanced,breakable,fontupper=\small,colback=tint,colframe=tint,arc=2mm,boxrule=0pt,left=4mm,right=4mm,top=3mm,bottom=2mm,before skip=6pt,after skip=10pt]{\sffamily\footnotesize\bfseries\color{accent}TL;DR}\par\smallskip}
\newcommand{\tldrend}{\end{tcolorbox}}
\makeatletter\@beginparpenalty=10000 \@secpenalty=0\makeatother
\newcommand{\setuplabel}{\par\addvspace{4pt}\noindent\textbf{\setupname}\par\nopagebreak[4]}
\DeclareRobustCommand{\expkicker}[1]{{\scriptsize\bfseries\color{accent}#1}\\*}
\usepackage{titlesec}
\titleformat{\section}{\sffamily\Large\bfseries\interlinepenalty=10000}{}{0pt}{}
\titleformat{\subsection}{\sffamily\large\bfseries\interlinepenalty=10000}{}{0pt}{}
\titlespacing*{\section}{0pt}{20pt plus 4pt minus 2pt}{6pt}
\titlespacing*{\subsection}{0pt}{16pt plus 3pt minus 2pt}{4pt}
\usepackage{needspace}
\newif\ifskipneed
\preto\section{\needspace{4\baselineskip}}
\preto\subsection{\ifskipneed\global\skipneedfalse\else\needspace{5\baselineskip}\fi}
\newcommand{\sectionwithsub}{\needspace{12\baselineskip}\global\skipneedtrue}
\usepackage{caption}
\captionsetup{font={small,sf},labelfont={bf,color=ink2},textfont={color=ink2},labelsep=colon,justification=raggedright,singlelinecheck=false,skip=6pt}
\captionsetup[table]{position=top}
\makeatletter\def\fps@figure{!htbp}\makeatother
\renewcommand{\topfraction}{0.85}\renewcommand{\bottomfraction}{0.6}\renewcommand{\textfraction}{0.12}
\renewcommand{\floatpagefraction}{0.8}\setcounter{topnumber}{2}\setcounter{bottomnumber}{1}\setcounter{totalnumber}{3}
\setlength{\textfloatsep}{14pt plus 4pt minus 4pt}\setlength{\floatsep}{12pt plus 3pt minus 3pt}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}
\fancyhead[R]{\sffamily\scriptsize\color{ink2}@HEADER@}
\fancyfoot[C]{\sffamily\footnotesize\color{ink2}\thepage}
\renewcommand{\headrulewidth}{0pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\sffamily\footnotesize\color{ink2}\thepage}}
\usepackage{enumitem}\setlist{itemsep=2pt,topsep=4pt}
\setlength{\emergencystretch}{3em}
\clubpenalty=2000 \widowpenalty=10000 \displaywidowpenalty=3000 \raggedbottom
\renewcommand{\arraystretch}{1.25}
\AtBeginEnvironment{tabular}{\sffamily\small}
"""

HEADER = HEADER.replace("@SETUP@", L["setup"]).replace("@HEADER@", L["header"])
GEOMETRY = "top=24mm,bottom=24mm,left=26mm,right=26mm,headsep=8mm"
if PRINT:  # review copy: room to write, line numbers to point at, page X of Y, and a ruled notes page at the end
    GEOMETRY = "top=22mm,bottom=24mm,left=22mm,right=46mm,headsep=8mm"
    HEADER += r"""
\usepackage{lastpage}
\fancyfoot[C]{\sffamily\footnotesize\color{ink2}@PAGE@ \thepage\ @OF@ \pageref*{LastPage}}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\sffamily\footnotesize\color{ink2}@PAGE@ \thepage\ @OF@ \pageref*{LastPage}}}
\usepackage[pagewise]{lineno}\renewcommand{\linenumberfont}{\normalfont\tiny\sffamily\color{ink2!60}}
\setlength{\linenumbersep}{6mm}
\usepackage{setspace}\setstretch{1.12}
\AtBeginDocument{\linenumbers}
\DeclareRobustCommand{\expkicker}[1]{{\scriptsize\bfseries\color{accent}#1}\enspace}
""".replace("@PAGE@", L["page"]).replace("@OF@", L["of"])
    md += ("\n\n\\clearpage\\nolinenumbers\n\n## " + L["notes"] + "\n\n" +
           "\\vspace{6mm}" + "".join("\\noindent{\\color{ink2!35}\\rule{\\linewidth}{0.4pt}}\\par\\vspace{4.5mm}" for _ in range(18)) + "\n")


def tex_escape(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"@B{\1}", s)
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    return s.replace("@B{", r"\textbf{")

BEFORE = rf"""
\thispagestyle{{plain}}
{{\sffamily\scriptsize\bfseries\color{{accent}}{L['kicker']}}}\par\medskip
{{\raggedright\fontsize{{20}}{{25}}\selectfont\bfseries {tex_escape(title)}\par}}\medskip
{{\sffamily\small\color{{ink2}} \textbf{{{L['authors']}}} {tex_escape(authors)}\par}}\smallskip
{{\small\itshape\color{{ink2}} {tex_escape(note)}\par}}
\medskip\noindent{{\color{{ink2!30}}\rule{{\linewidth}}{{0.4pt}}}}\par
"""

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "header.tex").write_text(HEADER)
    (td / "before.tex").write_text(BEFORE)
    tex = pypandoc.convert_text(
        md, "latex", format="markdown+raw_tex+lists_without_preceding_blankline",
        extra_args=["--standalone", "--shift-heading-level-by=-1",
                    f"--include-in-header={td / 'header.tex'}", f"--include-before-body={td / 'before.tex'}",
                    "-V", "documentclass=article", "-V", "papersize=a4", "-V", "fontsize=11pt",
                    "-V", f"geometry={GEOMETRY}",
                    "-V", "fontfamily=tgpagella",
                    "-V", "colorlinks=true", "-V", "linkcolor=accent", "-V", "urlcolor=accent",
                    "-V", f"lang={L['pandoc_lang']}", "--pdf-engine=pdflatex"])
    # short tables: pandoc's longtable mis-measures pages that also hold floats -> plain, unbreakable tabular
    tex = re.sub(r"\\bottomrule\\noalign\{\}\s*\\endlastfoot\s*", "", tex)
    tex = re.sub(r"\\end(first)?head\s*|\\endfoot\s*", "", tex)
    tex = tex.replace("\\begin{longtable}[]{", "\\par\\medskip\\noindent\\begin{tabular}{")
    tex = tex.replace("\\end{longtable}", "\\bottomrule\n\\end{tabular}\\par\\medskip")
    assert "longtable}" not in tex.split("\\begin{document}")[1]

    # tables: smaller type and tighter columns (footnotesize in the appendices), and keep each table on the same page
    # as the paragraph or heading that introduces it and the note that follows it
    app = re.search(r"\\section\{(Appendix|Ap\\^\{e\}ndice|Apêndice)", tex)
    cut = app.start() if app else len(tex)
    def tabfont(part, size):
        part = part.replace("\\par\\medskip\\noindent\\begin{tabular}{",
                            f"\\par\\medskip\\noindent{{{size}\\setlength{{\\tabcolsep}}{{4pt}}\\begin{{tabular}}{{")
        return part.replace("\\end{tabular}\\par\\medskip", "\\end{tabular}}\\par\\medskip")
    tex = tabfont(tex[:cut], "\\small") + tabfont(tex[cut:], "\\footnotesize")
    blocks = tex.split("\n\n")
    for i in range(1, len(blocks)):
        if "\\begin{tabular}" not in blocks[i]:
            continue
        body = blocks[i][blocks[i].index("\\begin{tabular}"):]
        rows = body.count("\\\\")
        wrap = 1.6 if "p{" in body else 1.1
        lead = max(1, len(re.sub(r"\\[a-zA-Z]+|[{}]", "", blocks[i - 1])) // 85 + 1)
        need = int(rows * wrap + lead + 3)
        j = i - 1  # start the reserved space at the heading(s) right above the lead-in, so no heading is left behind
        while j > 0 and re.search(r"\\(sub)?section\{", blocks[j - 1]):
            j -= 1; need += 3
        blocks[j] = f"\\needspace{{{need}\\baselineskip}}\n" + blocks[j]
    tex = "\n\n".join(blocks)

    # a short section (heading plus up to ~14 lines of text, e.g. "Data, code and logs"): never split it across pages
    secs = re.split(r"(?=\\section\{)", tex)
    for k in range(1, len(secs)):
        body = secs[k].split("\\end{document}")[0]
        if "\\subsection{" in body or "\\begin{tabular}" in body or "includegraphics" in body:
            continue
        paras = [b for b in body.split("\n\n")[1:] if b.strip()]
        lines = sum(len(re.sub(r"\\[a-zA-Z]+|[{}]", "", b)) // 85 + 1 for b in paras)
        if lines <= 14:
            secs[k] = f"\\needspace{{{lines + 4}\\baselineskip}}\n" + secs[k]
    tex = "".join(secs)

    # a section followed directly by a subsection: reserve room for both headings plus a few lines, once
    blocks = tex.split("\n\n")
    for i in range(len(blocks) - 1):
        if "\\section{" in blocks[i] and "\\subsection{" in blocks[i + 1]:
            blocks[i] = "\\sectionwithsub\n" + blocks[i]
    tex = "\n\n".join(blocks)
    (td / "paper.tex").write_text(tex)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=td,
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:]); sys.exit(1)
    log = (td / "paper.log").read_text(errors="ignore")
    over = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt", log)
    for m in re.finditer(r"Overfull \\hbox \((\d+\.\d+)pt[^\n]*\n(.{0,160})", log):
        if float(m.group(1)) > 5: print("  overfull", m.group(1), "pt:", m.group(2).replace("\n", " ")[:120])
    shutil.copy(td / "paper.pdf", OUT)
    pages = re.search(r"Output written on paper\.pdf \((\d+) pages", log)
print(OUT, pages.group(1) if pages else "?", "pages;", nfig, "figures;", f"{len(over)} overfull boxes (max {max(map(float, over), default=0):.1f}pt)")
