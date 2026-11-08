# Weight-Space Programming local — plano do experimento

Origem: "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab — MIT CSAIL, 02/11/2026,
<https://rikiyat.github.io/blog/weight-space-programming/>). No experimento 8, o GPT-6 Astra editou
0,016% dos pesos do Qwen3.5-0.8B (sem gradiente) e levou o MATH-500 de 7,9% para 48,1%, removendo
uma direção que sustenta loops de repetição no raciocínio. Logs completos:
<https://github.com/RikiyaT/lm_understand-logs> (`exp8_math/`), clonados em `/mnt/dados/dev/refs/`.

## Pergunta central

Com o **mesmo orçamento de avaliações** no modelo pequeno, um agente frontier (aqui: Claude Opus 5.5
via Claude Code) que raciocina sobre pesos e ativações melhora o modelo mais do que (a) busca cega,
(b) gradiente, (c) só prompt?

## Hipóteses (escritas antes de rodar)

- **H1.** Falhas de *elicitação* (formato de tool call, loops, parar cedo, JSON quebrado) são
  corrigíveis com edições de rank ≤ 1, e o agente as encontra com menos avaliações que busca cega.
- **H2.** Falhas de *capacidade* (errar o raciocínio em si) não melhoram com edição; GRPO também
  ganha pouco nessa escala (Yue et al., NeurIPS 2025).
- **H3.** Boa parte do ganho em tool calling sai só com prompt; a edição precisa superar esse baseline.
- **H4 (mapeamento por prompt).** Variantes de system prompt + monitoramento de ativações revelam as
  direções que a edição deve atingir: o delta de ativação induzido por um prompt que corrige o
  comportamento aponta na mesma direção que uma edição eficaz, e pode ser "gravado" nos pesos.

## Braços (mesmo orçamento de avaliações)

| Braço | O que faz | Pergunta |
|---|---|---|
| **P** | só otimiza o system prompt | quanto o prompt resolve sozinho |
| **M→A** | mapeia com variantes de prompt + ativações, grava a direção nos pesos, depois edita guiado pelo mapa | H4 |
| **A0** | agente edita sem o mapa | quanto o mapa vale |
| **R** | busca cega (ES/aleatória) no mesmo subespaço de edições | raciocínio > mutação aleatória? |
| **G** | GRPO + TinyLoRA/LoRA rank-1 | baseline de gradiente |

Métrica extra do M→A: **fidelidade da gravação** = fração do ganho do prompt que sobrevive nos pesos,
sem o prompt.

## Tarefas (verificadores determinísticos)

| Tarefa | Fonte | Verificador |
|---|---|---|
| Tool calling | gerador sintético próprio + fatia do BFCL simple/multiple | igualdade AST/JSON |
| Math | GSM8K (prática = train, avaliação = test) | resposta numérica exata |
| Coding | MBPP sanitized | testes executados em subprocess com timeout |

## Regras de validade (o agente sou eu, então o harness impõe)

1. **Orçamento no ledger**: `eval` debita do orçamento e recusa ao esgotar.
2. **Teste trancado**: só o harness lê o split de teste; um único `final` por run, registrado.
3. **Caderno em arquivo** (`docs/CADERNO.md` + sqlite): hipóteses, edições, resultados.
   O contexto do agente é compactado em sessões longas; o caderno é a memória.
4. **Réplicas**: ≥ 3 runs independentes (sessões ou subagentes sem acesso às notas uns dos outros).
5. **Custo**: sem dólares de API; registrar wall-time, nº de tool calls, nº de avaliações, GPU-hora.
6. Proibido no braço A (como no paper): gradiente/treino, rodar o modelo fora da ferramenta de
   avaliação, editar embedding/lm_head/norma final, escalar um tensor inteiro por constante.

## Hardware e stack

- GTX 1660 Super 6 GB (Turing, sm_75: **sem bf16** → fp16; KL fp16 vs fp32 ≈ 1e-5, verificado),
  16 GB RAM, 4 núcleos. Tudo em `/mnt/dados` (SSD cheio).
- `uv sync`: torch 2.14.1+cu130, transformers 5.22, flash-linear-attention 0.6.
  Cache uv em `/mnt/dados/cache/uv` (`UV_CACHE_DIR`), HF em `/mnt/dados/cache/huggingface`.
- Modelo: `Qwen/Qwen3.5-0.8B` @ `2fc0636` (mesma revisão do paper), híbrido: 18 camadas Gated DeltaNet
  + 6 de atenção completa (3, 7, 11, 15, 19, 23), MLP SwiGLU 3584, d=1024.
- **Geração: `llama-server` do Ollama** (`/usr/local/lib/ollama/llama-server`, via `scripts/llama_serve.sh`, `-np` 16–32, `--cache-ram 0`). O Ollama em si força `-np 1` para `qwen35`; fica só como registro dos modelos `wsp-base`/`wsp-astra` (f16 via `convert_hf_to_gguf.py` do llama.cpp em `/mnt/dados/dev/refs/llama.cpp`). Edições novas viram GGUF com `scripts/poc7_build.py`. Jobs longos sempre por `scripts/run_bg.sh` (unit do systemd --user com teto de memória). Presence penalty: passar `repeat_last_n >= max_new` (padrão do llama.cpp é 64).
- **Gargalo original: geração no HF.** HF `generate` faz ~21 tok/s (batch 1), ~222 tok/s (batch 64). O Qwen editado
  tem mediana de ~9k tokens por resposta correta → geração longa precisa de llama.cpp/Ollama
  (conversão para GGUF pendente) ou de tetos curtos. Medições teacher-forced são rápidas e densas.

## Fases

| Fase | Conteúdo | Status |
|---|---|---|
| F0 | stack, numérica fp16, throughput | feito (ver CADERNO) |
| F1 | engenharia reversa da edição publicada + medições teacher-forced (PoC 1–5) | em andamento |
| F2 | caixa de ferramentas do agente + ledger + braço P | — |
| F3 | experimento principal: 5 braços × 3 tarefas, começando por tool calling | — |
| F4 | escada de capacidade com redes de brinquedo (MinAtar, transformer de caractere) | — |

## Código

```
src/wsp/paths.py         caminhos, revisões dos modelos
src/wsp/modelio.py       carregar (fp16, só texto), edição rank-1 em memória com undo
src/wsp/tf.py            leitura teacher-forced (logp por token, residual), rótulo dup-8-grama
scripts/poc1_diff.py     diff base × editado → u por camada, α, colunas
scripts/poc2_hw.py       numérica fp16, edição em memória, throughput
scripts/poc3_tf.py       dose-resposta, direção por diferença de médias, system prompts
data/                    saídas (gitignored)
```
