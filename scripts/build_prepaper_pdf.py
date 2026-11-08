"""Typeset docs/paper-draft.md as a PDF with pandoc + pdfLaTeX (floating figures, proper page breaks).

usage: uv run python scripts/build_prepaper_pdf.py [out.pdf]      (default: docs/pre-paper.pdf)
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
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "docs" / "pre-paper.pdf"
md = (ROOT / "docs" / "paper-draft.md").read_text()

# --- title block: taken out of the body, typeset by hand
title = re.search(r"^# (.+)$", md, re.M).group(1)
authors = re.search(r"^\*\*Authors:\*\* (.+)$", md, re.M).group(1)
note = re.search(r"^\*(Working draft.+?)\*$", md, re.M | re.S).group(1).replace("\n", " ")
md = md[md.index("**TL;DR**"):]

# --- figures: image + "**Figure N:** caption" paragraph -> one captioned float
md, nfig = re.subn(r"!\[[^\]]*\]\((figures/[\w-]+\.png)\)\n\n\*\*Figure \d+:\*\* (.+?)\n\n",
                   lambda m: f"![{m.group(2).replace(chr(10), ' ')}]({ROOT / 'docs' / m.group(1)}){{width=94%}}\n\n", md, flags=re.S)
# --- "Experiment N" labels, TL;DR box, money signs, decorative rules
md = re.sub(r"^\*Experiment (\d+)\*\n\n### (.+)$", r"### \\texorpdfstring{\\expkicker{EXPERIMENT \1}}{} \2", md, flags=re.M)
assert "*Experiment" not in md
md = md.replace("**TL;DR**\n\n", "\\tldrbegin\n\n", 1)
md = md.replace("\n\nEverything ran on one gaming", "\n\n\\tldrend\n\nEverything ran on one gaming", 1)
md = re.sub(r"(?<![\\$])\$(?=\d)", r"\\$", md)
md = re.sub(r"^---$", "", md, flags=re.M)
md = re.sub(r"^\*\*Setup\*\*$", r"\\setuplabel", md, flags=re.M)  # keep "Setup" with its list

HEADER = r"""
\usepackage{etoolbox}
\usepackage{mathpazo}\renewcommand{\rmdefault}{qpl}\usepackage[scale=0.9]{tgheros}
\usepackage{newunicodechar}
\newunicodechar{−}{\ensuremath{-}}\newunicodechar{≤}{\ensuremath{\leq}}\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≈}{\ensuremath{\approx}}\newunicodechar{×}{\ensuremath{\times}}\newunicodechar{±}{\ensuremath{\pm}}
\usepackage{xcolor}
\definecolor{accent}{HTML}{1F5FAE}\definecolor{ink2}{HTML}{4B4F58}\definecolor{tint}{HTML}{EEF3FA}
\usepackage[most]{tcolorbox}
\newcommand{\tldrbegin}{\begin{tcolorbox}[enhanced,colback=tint,colframe=tint,arc=2mm,boxrule=0pt,left=4mm,right=4mm,top=3mm,bottom=2mm,before skip=6pt,after skip=10pt]{\sffamily\footnotesize\bfseries\color{accent}TL;DR}\par\smallskip}
\newcommand{\tldrend}{\end{tcolorbox}}
\makeatletter\@beginparpenalty=10000 \@secpenalty=0\makeatother
\newcommand{\setuplabel}{\par\addvspace{4pt}\noindent\textbf{Setup}\par\nopagebreak[4]}
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
\fancyhead[R]{\sffamily\scriptsize\color{ink2}Agentic Weight-Space Programming · pre-paper draft}
\fancyfoot[C]{\sffamily\footnotesize\color{ink2}\thepage}
\renewcommand{\headrulewidth}{0pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\sffamily\footnotesize\color{ink2}\thepage}}
\usepackage{enumitem}\setlist{itemsep=2pt,topsep=4pt}
\setlength{\emergencystretch}{3em}
\clubpenalty=2000 \widowpenalty=10000 \displaywidowpenalty=3000 \raggedbottom
\renewcommand{\arraystretch}{1.25}
\AtBeginEnvironment{tabular}{\sffamily\small}
"""

def tex_escape(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"@B{\1}", s)
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    return s.replace("@B{", r"\textbf{")

BEFORE = rf"""
\thispagestyle{{plain}}
{{\sffamily\scriptsize\bfseries\color{{accent}}PRE-PAPER · WORKING DRAFT}}\par\medskip
{{\raggedright\fontsize{{20}}{{25}}\selectfont\bfseries {tex_escape(title)}\par}}\medskip
{{\sffamily\small\color{{ink2}} \textbf{{Authors:}} {tex_escape(authors)}\par}}\smallskip
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
                    "-V", "geometry=top=24mm,bottom=24mm,left=26mm,right=26mm,headsep=8mm",
                    "-V", "fontfamily=tgpagella",
                    "-V", "colorlinks=true", "-V", "linkcolor=accent", "-V", "urlcolor=accent",
                    "-V", "lang=en", "--pdf-engine=pdflatex"])
    # short tables: pandoc's longtable mis-measures pages that also hold floats -> plain, unbreakable tabular
    tex = re.sub(r"\\bottomrule\\noalign\{\}\s*\\endlastfoot\s*", "", tex)
    tex = re.sub(r"\\end(first)?head\s*|\\endfoot\s*", "", tex)
    tex = tex.replace("\\begin{longtable}[]{", "\\par\\medskip\\noindent\\begin{tabular}{")
    tex = tex.replace("\\end{longtable}", "\\bottomrule\n\\end{tabular}\\par\\medskip")
    assert "longtable}" not in tex.split("\\begin{document}")[1]

    # a section followed directly by a subsection: reserve room for both headings plus a few lines, once
    blocks = tex.split("\n\n")
    for i in range(len(blocks) - 1):
        if "\\section{" in blocks[i] and "\\subsection{" in blocks[i + 1]:
            blocks[i] = "\\sectionwithsub\n" + blocks[i]
    tex = "\n\n".join(blocks)
    (td / "paper.tex").write_text(tex)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=td,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); sys.exit(1)
    log = (td / "paper.log").read_text(errors="ignore")
    over = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt", log)
    shutil.copy(td / "paper.pdf", OUT)
    pages = re.search(r"Output written on paper\.pdf \((\d+) pages", log)
print(OUT, pages.group(1) if pages else "?", "pages;", nfig, "figures;", f"{len(over)} overfull boxes (max {max(map(float, over), default=0):.1f}pt)")
