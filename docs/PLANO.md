# Weight-Space Programming local — plano do experimento

Origem: "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab — MIT CSAIL, 02/11/2026,
<https://rikiyat.github.io/blog/weight-space-programming/>). No experimento 8, o GPT-6 Astra editou
0,016% dos pesos do Qwen3.5-0.8B (sem gradiente) e levou o MATH-500 de 7,9% para 48,1%, removendo
uma direção que sustenta loops de repetição no raciocínio. Logs completos:
<https://github.com/RikiyaT/lm_understand-logs> (`exp8_math/`), clonados localmente (`REFS` em `src/wsp/paths.py`).

Este documento mantém o plano original (hipóteses e braços escritos antes de rodar) e o estado atual de cada item.
Resultados e evidências: [CADERNO.md](CADERNO.md) (registro primário) e [research-notes.md](research-notes.md).
Atualizado em 08/11/2026.

## Pergunta central

Com o **mesmo orçamento de avaliações** no modelo pequeno, um agente frontier (subagentes Claude Opus 5.5,
supervisionados pelo Claude Code) que raciocina sobre pesos e ativações melhora o modelo mais do que (a) busca cega,
(b) gradiente, (c) só prompt?

## Hipóteses (escritas antes de rodar) e estado

| | hipótese | estado |
|---|---|---|
| **H1** | Falhas de *elicitação* (formato de tool call, loops, parar cedo, JSON quebrado) são corrigíveis com edições de rank ≤ 1, e o agente as encontra com menos avaliações que busca cega. | **Confirmada para loops**: ≤ 4 edições rank-1 levam o GSM8K dev de 0,146 a 0,667 e o MATH-500 de 0,081 a 0,466; busca aleatória (rnd) e busca ingênua com as mesmas ferramentas (E3, 0,271) ficam muito abaixo. Outras falhas de elicitação: não testadas. |
| **H2** | Falhas de *capacidade* (errar o raciocínio em si) não melhoram com edição; GRPO também ganha pouco nessa escala. | Não testada diretamente. Sem raciocínio, nenhuma edição muda o GSM8K (efeito só com thinking ligado). |
| **H3** | Boa parte do ganho em tool calling sai só com prompt; a edição precisa superar esse baseline. | Para loops, prompt **não** ajuda (anti-repetição ≈ placebo). Tool calling ficou só como painel de efeito colateral (sem dano). |
| **H4** | Variantes de system prompt + monitoramento de ativações (atlas) revelam as direções que a edição deve atingir. | **Refutada** em 3 de 3 pares de réplicas: com atlas r1/r2/r3 (0,042 / 0,354 / 0,396 no dev48), sem atlas a0/a1/a2 (0,375 / 0,667 / 0,646). |

## Braços

| braço | plano original | o que foi feito |
|---|---|---|
| **P** | só otimiza o system prompt | PoC-5/6: prompts anti-repetição e placebo, em teacher forcing e em geração; nenhum resolve o loop |
| **M→A** | mapeia com variantes de prompt + ativações (atlas), edita guiado pelo mapa | r1 (protocolo v1), r2 e r3 (v2) |
| **A0** | agente edita sem o mapa | a0 (v1), a1 e a2 (v2); a1 é a edição principal |
| **R** | busca cega no mesmo subespaço de edições | rnd (aleatória, v1) e naive (E3: ferramentas dos agentes vencedores + sinal de geração, v2) |
| **G** | GRPO + TinyLoRA/LoRA rank-1 | substituído por **LoRA-SFT em dados autodestilados** (E4, em andamento): mais barato na 1660S e um baseline direto para o mesmo defeito |
| — | (não planejado) | baselines de decodificação (E1: logit bias, budget forcing) e a combinação edição + budget forcing |

Protocolos do avaliador:
- **v1:** sondas teacher-forced (branch_rep como objetivo); 60 chamadas.
- **v2:** 40 sondas + 6 checagens por geração (GSM8K train[200:216], teto 6144), porque a sonda sozinha sofreu
  Goodhart.

## Tarefas

| tarefa | plano | estado |
|---|---|---|
| Math | GSM8K (prática = train, avaliação = test) | **feito**: dev48 = train[0:48], teste trancado = test[0:32]; MATH-500[0:148] como transferência fora da distribuição de busca |
| Tool calling | gerador sintético próprio + fatia do BFCL | benchmark sintético próprio (`src/wsp/toolbench.py`) usado como painel de efeito colateral; é fácil demais para o 0.8B servir de alvo de edição |
| Coding | MBPP sanitized | não iniciado |

## Regras de validade

1. **Orçamento no ledger:** cada avaliação do avaliador debita do orçamento da run (`data/m2a/runs/<run>/ledger.jsonl`,
   `gen_ledger.jsonl`) e é recusada ao esgotar.
2. **Teste trancado:** o split de teste só abre com `WSP_UNLOCK_TEST=1`; toda avaliação é registrada em
   `data/ledger.jsonl`. Um candidato por braço, escolhido no dev antes de abrir o teste.
3. **Caderno versionado:** `docs/CADERNO.md` no git, com previsões registradas antes dos resultados e desvios
   declarados.
4. **Réplicas:** subagentes independentes, sem acesso às notas uns dos outros (3 pares com/sem atlas até agora).
5. **Custo:** sem dólares de API; registrar wall-time, nº de chamadas, nº de avaliações e GPU-hora (contabilidade
   completa junto com o E4).
6. **Restrições dos agentes** (como no paper): sem gradiente/treino, sem rodar o modelo fora do avaliador, sem editar
   embedding/lm_head/norma final, sem escalar um tensor inteiro. Proibido ler os artefatos do Astra, `docs/`,
   referências, outras runs, o holdout lacrado e o código do avaliador. Conformidade por honra, com auditoria das
   chamadas de ferramenta de cada transcrito (nenhuma violação até agora).

## Hardware e stack

- GTX 1660 SUPER 6 GB (Turing, sm_75: **sem bf16** → fp16; KL fp16 vs fp32 ≈ 1e-5, verificado), 16 GB RAM
  (~15 GB utilizáveis), 4 núcleos. Projeto, dados e caches no disco de dados (o SSD do sistema está quase cheio).
- `uv sync` com versões fixadas no `pyproject.toml`: torch 2.14.1+cu130, transformers 5.22.1,
  flash-linear-attention 0.6.0, peft 0.21.2. Cache do uv no disco de dados (`UV_CACHE_DIR`); cache do HF idem.
- Modelo: `Qwen/Qwen3.5-0.8B` @ `2fc0636` (mesma revisão do paper), híbrido: 18 camadas Gated DeltaNet
  + 6 de atenção completa (3, 7, 11, 15, 19, 23), MLP SwiGLU 3584, d=1024, vocabulário 248k.
- **Geração:** `llama-server` do Ollama, usado direto (`scripts/llama_serve.sh`, `-np 16`, `--cache-ram 0`), porque o
  Ollama força `-np 1` para `qwen35`. ~150 tok/s agregados com teto de 12k. Edições viram GGUF f16 com
  `scripts/poc7_build.py` (conversor do llama.cpp, `convert_hf_to_gguf.py`). Presence penalty na saída inteira
  (`repeat_last_n = max_new + 512`; o padrão do llama.cpp é 64).
- **Avaliador dos agentes** em HF fp16 (edição em memória, ~52 s por sonda); checagem por geração via GGUF +
  llama-server (~10 min).
- Jobs longos sempre por `scripts/run_bg.sh` (unit do systemd --user com teto de memória).

## Fases

| fase | conteúdo | estado |
|---|---|---|
| F0 | stack, numérica fp16, throughput | feito |
| F1 | engenharia reversa da edição publicada + medições teacher-forced (PoC 1–8) | feito |
| F2 | caixa de ferramentas do agente (avaliador v1/v2, ledger, atlas, holdout lacrado, teste trancado) | feito |
| F3 | experimento principal: braços × tarefas | loops/math feito (paper 1); tool calling e coding como alvo de edição: pendentes |
| F4 | escada de capacidade com redes de brinquedo (MinAtar, transformer de caractere) | fora do escopo do paper 1 |

## Paper 1: estado e próximos passos

| | conteúdo | estado |
|---|---|---|
| E1 | baselines de decodificação (logit bias, budget forcing) | feito |
| E2 | réplicas com e sem atlas (a2, r3) | feito |
| combo | edição a1 + budget forcing | feito |
| E3 | busca ingênua com as ferramentas dos agentes | feito |
| MATH-148 | MATH-500[0:148] para todos os braços principais | feito |
| E4 | baseline LoRA-SFT (dados autodestilados com excisão do loop) | em andamento |

Próximos passos (detalhes em [paper-draft.md](paper-draft.md)):
1. concluir o E4 e fazer a contabilidade de custo (agente vs fine-tuning);
2. aumentar o n no GSM8K test (≥ 200);
3. mais réplicas do protocolo vencedor e do braço com atlas;
4. um segundo defeito sem marcador lexical (formato de tool call, parada precoce, JSON) e, se der, um segundo modelo.

## Código

```
src/wsp/paths.py         caminhos, revisões dos modelos
src/wsp/modelio.py       carregar (fp16, só texto), edição rank-1 em memória com undo
src/wsp/tf.py            leitura teacher-forced (logp por token, residual), rótulo dup-8-grama
src/wsp/probes.py        sondas de ramificação (pares casados), onset de loop
src/wsp/m2a.py           API do agente: validate / apply / submit / submit_gen / history / atlas
src/wsp/evalrun.py       avaliação por geração com checkpoint por item, braços de decodificação, ledger, pareamento
src/wsp/llama_gen.py     cliente do llama-server (amostragem THINK35, budget forcing)
src/wsp/gen.py           extração e checagem de respostas (\boxed{})
src/wsp/toolbench.py     benchmark sintético de tool calling
scripts/poc*.py          PoC 1–8 (diff do Astra, numérica, teacher forcing, geração, fatorial)
scripts/m2a_*.py         atlas, avaliador (daemon), busca aleatória, busca ingênua, dose
scripts/gen_eval.py      GSM8K por geração; math_eval.py (MATH-500); tool_eval.py (tool calling)
scripts/ft_*.py          E4: viabilidade, dados, treino LoRA, exportação para GGUF
scripts/queue_*.sh       filas de cada experimento; run_bg.sh, llama_serve.sh
data/                    saídas (só registros pequenos versionados; ver .gitignore)
```
