# Caderno de laboratório — WSP local

Entradas em ordem cronológica. Números brutos em `data/*.json`.

---

## 2026-11-03 — F0: stack e hardware

- `uv sync` no venv do projeto (em `/mnt/dados`): torch 2.14.1+cu130 (inclui sm_75),
  transformers 5.22.1, flash-linear-attention 0.6.0. `causal_conv1d` não instalado (não há nvcc);
  o fallback PyTorch dele é usado.
- Baixados `Qwen/Qwen3.5-0.8B` @ `2fc0636` e `RikiyaT/Qwen3.5-0.8B-weight-space-programming`
  (1,7 GB cada, no cache HF do disco de dados).
- Logs do paper (exp8) clonados em `/mnt/dados/dev/refs/lm_understand-logs/exp8_math`: prompt da
  tarefa, as 158 turns do Astra, tool calls, caderno e relatório final dele.

**Numérica fp16 (PoC-2)** — Turing não tem bf16. Logits fp16 vs fp32 no mesmo prompt:
sem NaN/inf, top-1 100% igual, KL médio 9,5e-6, diferença máxima de logit 0,085. **fp16 é seguro.**

**Edição em memória** — aplicar a edição recuperada (ver PoC-1) no modelo base fp16 reproduz o
checkpoint editado: top-1 100% igual, diferença máxima de logit 0,047 (arredondamento fp16 vs bf16).

**Throughput HF `generate`** (256 tokens novos, fp16):

| batch | sem fla | com fla |
|---|---|---|
| 1 | 18 tok/s | 21 tok/s |
| 8 | 39 tok/s | — |
| 32 | 125 tok/s | 135 tok/s |
| 64 | — | 222 tok/s (4,0 GB) |

Gargalo é overhead de Python por passo de decodificação, não a GPU. Para o protocolo do paper
(32k tokens, mediana de 9k nas respostas corretas do editado) isso é inviável: 64 problemas × 9k
tokens ≈ 45 min por braço. Saídas: (a) llama.cpp/Ollama (o Ollama 0.40.6 já roda a arquitetura
`qwen35`, mas não importa safetensors sem MLX → precisa converter para GGUF com o
`convert_hf_to_gguf.py` do llama.cpp; adotado a partir do PoC-7);
(b) medições teacher-forced (rápidas); (c) tetos curtos de geração.

## 2026-11-03 — PoC-1: engenharia reversa da edição do Astra

Só dois tensores diferem do base: `layers.15.mlp.down_proj.weight` e `layers.19.mlp.down_proj.weight`
([1024, 3584]).

| camada | colunas alteradas | energia no 1º valor singular | α estimado | erro de reconstrução | mudança relativa |
|---|---|---|---|---|---|
| 15 | 64 | 0,999994 | 6,0 | 0,23% | 10,6% |
| 19 | 64 | 0,999991 | 6,0 | 0,29% | 8,1% |

- A edição é **exatamente rank-1** por camada: `W' = W - 6·u (mask64 ⊙ uᵀW)`, como o relatório do
  Astra diz. Com α = 6 não é uma ablação: o componente de cada uma das 64 colunas na direção u passa
  a ser **-5×** o original (inverte e amplifica).
- **|cos(u15, u19)| = 0,849**: as duas camadas atacam praticamente a mesma direção do residual
  stream. É uma única "direção de repetição", escrita por neurônios diferentes (só 3 índices em comum).
- Contexto do relatório do Astra: u veio de 4 pares de contextos "produtivo vs repetitivo" do próprio
  modelo, lidos nas saídas do residual 16/20; os 64 neurônios são os que mais escrevem em u.
  Controle aleatório com norma igual: +1,56 pp (SE 3,52), contra +49,8 pp (SE 1,5) da edição, em
  1152 problemas de prática held-out.
- Nas outras runs: com 10 passes, ele removeu uma direção de 30 matrizes (camadas 8–22, α = 1,5);
  com 25, desligou a head 7 da camada 15. As duas tiveram ganhos pequenos. Só a run de 50 passes
  achou a edição forte (20 h de parede, 158 turns, 80 execuções de Python).

## 2026-11-03 — PoC-3/4/5: medições teacher-forced (20 textos em loop do base, MATH-500)

Setup: os 20 textos publicados do modelo base (todos em loop até 32k), lidos pelo base em fp16 na
janela de tokens 512–4096 do assistente. Rótulo mecânico por token: **repetitivo** se o 8-grama que
termina nele já apareceu antes no texto (média de ~1050 tokens repetitivos e ~2530 produtivos por texto).
35 min, ~280 tok/s de prefill no HF, incluindo cópias das ativações. Saída: `data/poc3_tf_w4096.json`.

**PoC-3: dose-resposta da edição do Astra** (Δ log-prob médio por token, editado − base):

| α | repetitivos | produtivos | razão |
|---|---|---|---|
| 1 | −0,0038 | −0,0016 | 2,3× |
| 2 | −0,0079 | −0,0039 | 2,0× |
| 4 | −0,0172 | −0,0100 | 1,7× |
| 6 | −0,0281 | −0,0183 | 1,5× |

Monotônica e seletiva, mas **bem mais fraca** que a assinatura do Astra (−0,103 contra −0,009, ~11×).
O mais provável é que as medidas sejam diferentes. Dentro de um loop já estabelecido, o base já tem
log-prob de −0,11 nos tokens repetitivos (quase determinístico), e a edição quase não mexe nisso.
O Astra mediu no **ponto de bifurcação**, com prefixos iguais e duas continuações (produtiva ou
repetitiva). Leitura: a edição age na **entrada** do loop, não na manutenção. Próxima métrica:
log-prob no início do loop (primeira repetição de uma sentença).

**PoC-4: o "método trivial" recupera a direção?** A direção r = média(residual | repetitivo) −
média(residual | produtivo), sem LLM nenhum:

| camada do residual | cos(r, u) por texto (média) | cos(r, u) com r agregado dos 20 | acaso |
|---|---|---|---|
| 16 (após L15) | −0,39 | **−0,52** | 0,024 |
| 20 (após L19) | −0,39 | **−0,52** | 0,024 |

O sinal é irrelevante (o sinal de u no SVD é arbitrário, e a edição u uᵀ não depende dele).
**|cos| = 0,52 fica cerca de 20× acima do acaso**, a partir de 20 textos naturais e uma diferença de
médias. A direção em si é largamente recuperável de forma mecânica. O que ainda não sabemos: se r,
usado com a mesma receita (top-64 escritores, α = 6), produz o mesmo efeito de comportamento →
**PoC-7**. Se produzir, o mérito do Astra está no diagnóstico (o loop é a falha dominante; quem
escreve são os MLPs das camadas 15 e 19; a receita esparsa; a disciplina de validação), não na
álgebra da direção.

**PoC-5: a ideia do system prompt** (mesmo texto do assistente lido sob system prompts diferentes):

| prompt | Δlogp repetitivos | Δlogp produtivos | cos(Δativação, u), camadas 16 / 20 (agregado) |
|---|---|---|---|
| anti-repetição | +0,0005 | −0,0011 | −0,09 / −0,15 |
| placebo (neutro, mesmo tamanho) | +0,0009 | −0,0008 | −0,19 / −0,30 |
| **anti-rep − placebo** | — | — | **+0,12 / +0,21** |

- Em teacher forcing, nenhum prompt muda a probabilidade de continuar o loop. Dentro de um loop,
  o contexto repetido domina o system prompt, que fica milhares de tokens atrás.
- **Qualquer** system prompt desloca as ativações ao longo de u, com o mesmo sinal da diferença
  (repetitivo − produtivo). É um efeito genérico, maior no placebo.
- O componente **específico** da instrução anti-repetição (descontado o placebo) aponta para o lado
  oposto, o de "não repetir", com cos de 0,12 a 0,21 (5 a 9× o acaso). Ou seja: a instrução deixa uma
  pegada fraca e de sinal correto no eixo de repetição, mas fraca demais para alterar o comportamento
  dentro de um loop. O teste de comportamento (gerar do zero com o prompt) é o PoC-6.
- Lição de método para o braço M→A: **sempre** ter o placebo. Sem ele, o efeito genérico de "existe
  um system prompt" teria sido lido como efeito da instrução, e com o sinal errado.

## 2026-11-03 — Motor de geração

| motor | config | tok/s agregado |
|---|---|---|
| HF `generate` fp16 | batch 64 | 222 (desperdiça passos com sequências que já terminaram) |
| Ollama 0.40.6 | `OLLAMA_NUM_PARALLEL=16` | 69 |
| llama-server (binário do Ollama) | `-np 16` | 168 |
| idem | `-np 32` | 181 |
| idem | `-np 48` | 188 |

- O Ollama **ignora** o paralelismo para a arquitetura `qwen35` (log: "model architecture does not
  currently support parallel requests") e força `-np 1`. Por isso usamos o `llama-server` que vem com
  ele (`/usr/local/lib/ollama/llama-server`), direto: `scripts/llama_serve.sh`.
- Atenção: sem `GGML_BACKEND_PATH=/usr/local/lib/ollama/cuda_v13/libggml-cuda.so`, o binário não
  acha o backend CUDA e roda **na CPU** (~8 tok/s) sem dar erro.
- Platô em ~185 tok/s: o gargalo provável são os kernels Gated DeltaNet com várias sequências, não a
  banda de memória (o teórico para 1,6 GB a 336 GB/s seria ~210 tok/s **por sequência**). Ainda não
  testados: Q8_0 e ubatch maior.
- HF e llama-server ficam na mesma ordem de grandeza. O HF tem uma vantagem estrutural para o loop
  do agente: aplica a edição **em memória**, sem converter para GGUF (~1 min e 1,6 GB por variante).
  O llama-server tem continuous batching e protocolo idêntico ao do vLLM do paper.

## 2026-11-04 — Estabilidade do harness

- O cache de prompts em RAM do `llama-server`, ligado por padrão (`--cache-ram 8192` MiB), leva o processo a
  7,4–8 GB de RSS e ao OOM. Configuração adotada: `--cache-ram 0 --ctx-checkpoints 1` no servidor e
  `cache_prompt: false` no cliente.
- Jobs longos rodam por `scripts/run_bg.sh` (`systemd-run --user`, `MemoryMax=6G`), isolados do terminal: um
  OOM afeta só o experimento.
- **Desvio de protocolo descoberto:** no llama.cpp, a presence penalty vale só para os últimos
  `repeat_last_n` tokens (padrão 64). No vLLM do paper, ela vale para a saída inteira. O PoC-6
  rodou com janela de 64, então os números absolutos não são comparáveis aos do paper; a
  comparação entre braços continua válida. A partir do PoC-7, `gen_eval.py` usa
  `repeat_last_n = max_new + 512`, e o servidor confirma isso em `generation_settings`.
  Diferença que sobra: o llama-server também alimenta os tokens do prompt no sampler, então eles
  entram na penalidade. O efeito é menor, porque os dígitos e as palavras comuns aparecem na saída
  logo nas primeiras centenas de tokens.

## 2026-11-04 — PoC-6 (parcial): geração GSM8K, 64 problemas, teto 4096, janela de penalty 64

| braço | acc | cut | fechou `</think>` | loop no teto | tok/s |
|---|---|---|---|---|---|
| base | 0,031 (2/64) | 0,98 | 0,016 | 0,23 | 107 |
| anti_rep | 0,047 (3/64) | 1,00 | 0,000 | **0,31** | 112 |
| placebo | 0,031 (2/64) | 0,97 | 0,031 | **0,30** | 110 |
| astra | ver resultado final abaixo | | | | |

- Com teto de 4096, o base quase nunca termina, e na maioria dos casos o que acontece é raciocínio
  longo demais ("Wait..."), não loop duro. A acurácia não discrimina nada aqui; os conjuntos de
  acerto do base ({1, 55}) e do anti_rep ({23, 25, 26}) nem se sobrepõem.
- O system prompt anti-repetição **não ajudou** e talvez tenha piorado a repetição no fim
  (quartil superior do `dup_tail`: 0,44 → 0,60). Isso é coerente com o PoC-5: a instrução deixa
  uma pegada fraca no eixo u e não muda o comportamento.
- **O placebo dá o mesmo resultado que o anti_rep** (loop no teto: 0,30 contra 0,31; base 0,23). O aumento
  de loop vem de *ter um system prompt*, não da instrução. Isso bate com o PoC-5: qualquer system prompt
  desloca o residual ao longo de u **no sentido da repetição**, e o deslocamento é maior no placebo. É a
  primeira ligação entre uma medição de ativação (PoC-5) e uma mudança de comportamento (PoC-6). A
  diferença ainda é pequena para N=64 e precisa de réplica.

## 2026-11-04 — PoC-7: desenho (`scripts/queue_night.sh`)

**Pergunta:** a direção recuperada **sem** o Astra (diferença de médias do PoC-4, |cos| ≈ 0,5 com u),
transformada numa edição no mesmo formato (64 colunas, α=6), reproduz o ganho comportamental do
Astra? É o teste de ponta a ponta do braço M (mapear → editar) sem modelo de fronteira.

- `poc7_dirs.py`:
  - recalcula a direção de diferença de médias nas camadas 15/19;
  - escolhe as colunas do down_proj por top-64 de |dᵀW|;
  - cria um controle com direção aleatória;
  - checa se essa regra de seleção reproduz as colunas que o Astra escolheu;
  - mede o Δlogp teacher-forced das três edições.
- `poc7_build.py`: materializa a edição em safetensors e converte para GGUF. **Validado** com
  `astra_check`, que reconstrói o Astra a partir do base mais `edit_dirs.pt`: a diferença para o
  checkpoint publicado é de 1,2e-4 relativo e 4,9e-4 absoluto no máximo, ou seja, só o
  arredondamento bf16.
- `gen_eval.py`: GSM8K com 32 problemas, teto 12.288, `np=16`, penalty na saída inteira, braços
  base → astra → meandiff → random. Grava checkpoint por problema
  (`data/gen_t12k/<braço>.jsonl`), e se o servidor não subir tenta de novo com metade dos slots.
- Cadeia validada de ponta a ponta numa instância só de CPU (`WSP_LLAMA_NGL=0`).
- Resultados em `data/gen_t12k.json` e `data/poc7_dirs.json`.

## 2026-11-04 — Painel de efeito colateral e tool calling: hipóteses

Hipóteses (`queue_night.sh`), todas com os braços base, astra, meandiff e random:

- **H-especificidade:** a edição anti-loop não deve prejudicar capacidades fora do alvo. Medidas:
  - Δlogp em texto neutro (WikiText-2, dentro do `poc7_dirs`);
  - tool calling sem raciocínio;
  - GSM8K sem raciocínio, em que não há loop longo para remover; o esperado é ganho de ~0.
- **H-mecanismo:** se a edição atua no raciocínio longo, o ganho deve aparecer **só** com raciocínio
  ligado. Compara GSM8K com e sem raciocínio e tool calling com e sem raciocínio.
- **H-M (central):** meandiff ≈ astra > random no GSM8K com raciocínio e teto de 12k.

**Benchmark de tool calling** (`src/wsp/toolbench.py`, sintético e determinístico):
- 11 ferramentas com esquema JSON e 4 ferramentas oferecidas por tarefa.
- Tipos de tarefa: single (60%), parallel (20%, duas chamadas) e irrelevance (20%, o certo é não chamar).
- Formato: o XML do chat template do Qwen3.5.
- Validação do verificador: as respostas certas renderizadas pelo template pontuam 300/300, e os
  controles negativos (argumento errado ou nenhuma chamada) dão erro.
- Splits: dev com sementes 1000+i; teste com sementes 900000+i, **trancado**. Só abre com
  `--test` e `WSP_UNLOCK_TEST=1`, e toda avaliação vai para `data/ledger.jsonl`, que é o começo
  do ledger de orçamento.
- Amostragem sem presence penalty, porque o llama-server põe o prompt no sampler e a penalidade
  puniria copiar argumentos da mensagem do usuário.
- Teste de fumaça (astra, 8 tarefas, CPU): 7/8 exatas. **Risco de teto:** este benchmark é fácil
  para o 0.8B. Para servir como alvo de edição (F3), vai precisar de uma versão difícil: multi-turn,
  muitas ferramentas, argumentos aninhados e conversão implícita de unidades e datas.

## 2026-11-04 (manhã) — Resultados: PoC-6, PoC-7 e painel de efeito colateral

Geração com teto de 12k e `np=16`: ~150 tok/s agregados.

### PoC-6 final (GSM8K, 64 problemas, teto 4096)
| braço | acc | loop no teto | ganho pareado |
|---|---|---|---|
| base | 0,031 | 0,23 | — |
| anti_rep | 0,047 | 0,31 | +1,6 ± 3,5 pp |
| placebo | 0,031 | 0,30 | 0 ± 3,1 pp |
| astra | 0,062 | **0,00** | +3,1 ± 3,1 pp |

Com 4k tokens, o Astra elimina o loop duro: nenhuma resposta acaba em loop no teto, contra 23% no
base. Mesmo assim ele ainda precisa de mais de 4k tokens para fechar o raciocínio, então a acurácia
ainda não aparece.

### PoC-7a — edições (teacher-forced nos 20 textos em loop, janela 512–4096; WikiText com 16×1024 tokens)
| edição | Δlogp repetitivo | Δlogp produtivo | Δlogp neutro | mudança relativa L15/L19 | sobreposição de colunas com o Astra |
|---|---|---|---|---|---|
| astra | −0,028 | −0,018 | −0,0063 | 10,6% / 8,1% | — |
| meandiff | **−0,033** | **−0,011** | −0,0035 | 9,3% / 6,9% | 44% / 19% (\|cos u\| = 0,52) |
| random | −0,0002 | −0,0003 | −0,0004 | 7,3% / 6,7% | 0% / 2% |

- A NLL do base no WikiText é 2,84. As edições custam 0,1 a 0,2% dela.
- Pela métrica teacher-forced, a meandiff é **mais forte e mais seletiva** que o Astra.
- A regra "top-64 por |uᵀW|" **não** reproduz as colunas do Astra: a sobreposição é de 48% em L15 e
  28% em L19. O Astra escolhe os neurônios por outro critério.

### PoC-7b — comportamento (GSM8K, 32 problemas, raciocínio ligado, teto 12.288, penalty na saída inteira)
| braço | acc | fechou `</think>` | loop no teto | mediana de tokens (terminadas) | ganho pareado |
|---|---|---|---|---|---|
| base | 0,125 | 0,125 | 0,875 | 4109 | — |
| **astra** | **0,406** | **0,625** | **0,22** | 6854 | **+28 ± 10 pp (11 resgates / 2 perdas)** |
| meandiff | 0,188 | 0,31 | 0,81 | 5788 | +6 ± 9 pp (5/3) |
| random | 0,062 | 0,06 | 0,91 | 4760 | −6 ± 4 pp (0/2) |

- **O ganho do Astra se reproduz localmente.** O efeito é grande, de ~2,7 erros-padrão, e a
  mediana de tokens das respostas corretas fica em ~7k, como no paper.
- **A meandiff não reproduz.** Ela leva algumas respostas a fechar o raciocínio (0,31 contra 0,125),
  mas 81% continuam em loop. Isso dá **dissociação**: a métrica teacher-forced dizia meandiff ≥
  astra, e o comportamento diz astra ≫ meandiff.
- **Lição de método:** Δlogp médio em tokens de loop **não serve** de proxy para o efeito no
  comportamento. O relatório do Astra usava a métrica de ponto de ramificação (a entrada no loop),
  não a média. O loop interno do braço M precisa de uma métrica de entrada no loop ou de geração curta.
- **Suspeita principal: as colunas.** A direção da meandiff é parecida com o u do Astra
  (|cos| = 0,52), mas as colunas coincidem pouco, principalmente em L19 (19%). O próximo passo é um
  fatorial 2×2: {u do Astra, meandiff} × {colunas do Astra, top-64}.

### Painel de efeito colateral
| teste | base | astra | meandiff | random |
|---|---|---|---|---|
| tool calling sem raciocínio (300) | 0,897 | 0,870 (−2,7 ± 1,9) | 0,930 (+3,3 ± 1,4) | 0,900 |
| GSM8K sem raciocínio (200, teto 1024) | 0,485 | 0,480 (−0,5 ± 3,6) | 0,500 (+1,5 ± 3,8) | 0,475 |
| tool calling com raciocínio (100, teto 3072) | 0,93 | **0,98 (+5 ± 2,2; 5/0)** | 0,94 | 0,92 |

- **H-especificidade:** sem raciocínio, nenhuma edição muda nada de forma significativa no GSM8K.
  No tool calling sem raciocínio, as diferenças ficam em ±3 pp, dentro de ~2 erros-padrão e sem
  correção para comparações múltiplas. A edição do Astra é cirúrgica.
- **H-mecanismo:** confirmado. O efeito só aparece com o raciocínio ligado.
- **Achado novo:** com raciocínio, o Astra melhora o tool calling (+5 pp, 5 resgates e 0 perdas).
  O ganho vem todo das tarefas *irrelevance*: o base acerta 11/16 e o Astra 16/16. Nas falhas, o base
  raciocina e depois **inventa uma chamada** (por exemplo, "17 vezes 23" vira
  `convert_currency(17, 23)`) ou fica preso contando sílabas. Com menos tokens em média (138 contra
  174), a edição anti-loop também reduz chamadas alucinadas. Ainda é um efeito pequeno (N=16
  tarefas irrelevantes) e precisa de réplica.

## 2026-11-04 — PoC-8a: fatorial direção × colunas, métricas teacher-forced (registrado às 10:27, ANTES da geração)

Direções: astra, md (8-gramas na janela 512–4096, em geral antes do loop) e mdloop (região do loop
[onset, onset+256) contra o pré-loop produtivo). Colunas: astra, topw (|uᵀW|) e contrast (|Δa ⊙ uᵀW|).
α=6. O onset do loop nos textos base do paper fica entre 3k e 10k tokens (18 de 20 textos).

- |cos| com o u do Astra: md 0,52; **mdloop 0,74**. Medir na região do loop real aproxima bem mais.
- Sobreposição de colunas com as do Astra (L15/L19), com u = astra: topw 48/28%, **contrast 67/45%**.

| célula | branch Astra rep / prod | branch próprio rep / prod | onset sel. | neutro |
|---|---|---|---|---|
| astra_astra (= Astra) | **−0,101 / −0,010** | **−0,097 / −0,000** | −0,015 | −0,002 |
| astra_topw | −0,069 / +0,008 | −0,054 / +0,002 | −0,008 | −0,007 |
| astra_contrast | −0,058 / −0,009 | −0,052 / −0,002 | −0,012 | −0,003 |
| mdloop_astra | −0,065 / −0,011 | −0,056 / −0,005 | −0,013 | −0,003 |
| mdloop_contrast | −0,035 / −0,016 | −0,031 / −0,010 | −0,012 | −0,005 |
| mdloop_topw | −0,033 / −0,010 | −0,029 / −0,010 | −0,005 | −0,010 |
| md_astra | −0,024 / −0,005 | −0,033 / −0,002 | −0,027 | +0,001 |
| md_contrast | −0,010 / −0,008 | −0,021 / −0,002 | −0,036 | +0,001 |
| md_topw (= meandiff) | −0,006 / −0,001 | −0,014 / +0,002 | −0,026 | −0,003 |
| random_topw | −0,000 / +0,001 | −0,001 / +0,002 | 0,000 | −0,000 |

- **A métrica de ramificação replica o Astra.** Nos 6 casos dele temos −0,101 / −0,010; o relatório
  dele dá −0,104 / +0,005 com α=4 e direção cheia. Também **generaliza** para os 6 casos novos que
  escritos no estilo GSM8K (−0,097 / −0,000).
- **Ela separa astra de meandiff** (−0,10 contra −0,01), o que bate com o comportamento (+28 contra
  +6 pp). As métricas de onset natural e de média por token **não separam**, e chegam a ordenar ao
  contrário (meandiff > astra). Logo, o proxy válido para o loop do agente é o par casado.
- **Leitura fatorial (repetitivo, média das duas sondas):**
  - direção: astra ≫ mdloop > md;
  - colunas: as do Astra são as melhores para **toda** direção; contrast ≈ topw.
  - Direção e colunas contam as duas, e nenhuma das regras derivadas localmente chega ao Astra.

**Previsão registrada para o PoC-8b** (ganho no GSM8K com raciocínio e teto de 12k, contra o base):
1. astra (+28, medido) > {mdloop_astra, astra_topw, astra_contrast} > {md_astra, mdloop_contrast,
   mdloop_topw} > md_contrast > meandiff (+6, medido) > random.
2. Se a ordem por "branch rep" prever a ordem dos ganhos (Spearman > 0,7 nas 10 células), a métrica
   vira o sinal barato do ciclo do agente (braço M→A).

## 2026-11-04 — PoC-8b (10 células): a previsão se confirmou

GSM8K com raciocínio, 32 problemas, teto 12k; base = 0,125.

| célula | branch rep (média das 2 sondas) | acc | fechou o raciocínio | loop no teto |
|---|---|---|---|---|
| astra | −0,099 | 0,406 | 0,62 | 0,22 |
| astra_topw | −0,062 | 0,406 | 0,66 | 0,28 |
| mdloop_astra | −0,060 | 0,250 | 0,41 | 0,53 |
| astra_contrast | −0,055 | 0,344 | 0,47 | 0,38 |
| **mdloop_contrast** (100% derivado localmente) | −0,033 | **0,281** | 0,50 | 0,56 |
| md_astra | −0,028 | 0,250 | 0,31 | 0,66 |
| md_contrast | −0,015 | 0,188 | 0,28 | 0,78 |
| meandiff | −0,010 | 0,188 | 0,31 | 0,81 |
| mdloop_topw | −0,031 | **0,312** (+19 ± 9 pp, 8/2) | 0,53 | 0,53 |
| random | −0,000 | 0,062 | 0,06 | 0,91 |

- Com as 10 células: Spearman 0,89 com acc e 0,96 com (−loop no teto). Valores com 9 células: **0,92** com acc, 0,89 com fechar o
  raciocínio e **0,98** com (−loop no teto). O critério registrado era > 0,7 e **foi atingido**.
- Spearman da métrica de onset natural: −0,19, −0,29 e −0,13. Fica **descartada**.
- O u do Astra com as colunas topw atinge a acurácia do Astra (0,406): no comportamento, a direção
  pesa mais que a escolha exata das colunas. Ressalva: o erro-padrão por célula é de ~9–10 pp com
  N=32, então diferenças entre células vizinhas não são significativas. O que é robusto é a
  correlação ao longo das células.
- A melhor edição 100% derivada localmente, sem o u do Astra (mdloop_contrast), sai de 12,5% para
  28,1%, cerca de metade do ganho do Astra.

## 2026-11-04 — Braço M→A montado (supervisor = Claude Code; agentes = subagentes independentes)

- **Fase M:** atlas de ativações do base (`scripts/m2a_atlas.py` → `data/m2a/atlas.pt`).
  - Sinais por camada: residual, entrada do down_proj e entrada do out_proj da atenção.
  - Contextos: 18 traços naturais (loop vs pré-loop) e 12 pares casados de busca, cada um sob 4
    system prompts (none, anti_rep, placebo, concise). Esta é a parte de mapeamento por prompt (H4).
- **Fase A:** avaliador daemon (`scripts/m2a_scorer.py`, unit `wsp-m2a`).
  - Fila em arquivo por run, com orçamento em `config.json` e ledger por run.
  - Espaço de edição: ≤ 4 edições rank-1 do lado da saída em down_proj, out_proj ou o_proj, com |α| ≤ 10.
  - Métricas: branch_rep (objetivo), branch_prod, close, neutral.
  - Holdout de 12 pares casados novos, **lacrado** em `runs/<run>/sealed/`, só para o supervisor.
- **Brief do agente:** `data/m2a/BRIEF.md`.
  - Orçamento: 100 avaliações.
  - Proibido ler: os artefatos do Astra, `docs/`, `refs/`, os resultados de PoC e o holdout.
  - Proibido rodar o modelo.
  - O agente recebe só o contexto genérico do problema: loops no raciocínio e o fato de que prompt
    não resolve. **Não** recebe camada, matriz nem direção.
  - A conformidade é por honra. A auditoria fica por conta do relatório e das notas do agente.
- **Validação externa** (supervisor, depois da busca):
  1. o candidato nº 1 de cada run é confirmado por geração no dev (GSM8K train[0:48], teto 12k;
     referências base e astra em `gen_dev48`);
  2. o melhor candidato vai para o teste trancado.
- **Réplicas:** primeiro um piloto (r1) para depurar o harness, depois r2 e r3 em paralelo, sem
  acesso às notas umas das outras.

## 2026-11-04 (tarde) — Referência dev48, sanidade do avaliador e lançamento do M→A

**Referência dev48** (GSM8K train[0:48], teto 12k, penalidade no output inteiro), `data/gen_dev48.json`:

| braço | acc | loop no teto | fechou o raciocínio |
|---|---|---|---|
| base | 0,146 | 0,85 | 0,15 |
| astra | **0,458** | 0,23 | 0,54 |

O astra ganha **+31 ± 7 pp** pareado (16 resgates, 1 perda). Isso replica o t12k (+28 pp) em outro split.

**Correção antes dos agentes:**
- A checagem de sanidade pegou um bug em `m2a.validate`: ele rejeitava `mlp.down_proj` nas camadas
  de atenção cheia (3, 7, …). A L15 do Astra seria proibida. Corrigido antes de qualquer agente rodar.

**Sanidade do avaliador** (run `sanity`, edições conhecidas; branch = média dos 12 pares; ~52 s/avaliação):

| edição | branch_rep | branch_prod | close | neutral | holdout_rep (lacrado) |
|---|---|---|---|---|---|
| astra | −0,099 | −0,005 | +0,15 | −0,002 | −0,115 |
| mdloop_topw | −0,031 | −0,010 | +0,52 | −0,010 | −0,040 |
| md_topw | −0,010 | +0,001 | +0,16 | −0,003 | −0,014 |
| random_topw | −0,000 | +0,002 | −0,11 | −0,000 | +0,001 |

- Os valores batem com o PoC-8 até a 3ª casa.
- O holdout lacrado preserva a ordem e a escala, então a métrica generaliza para pares novos.

**Braços** (avaliador único na GPU, compartilhado em fila):

| run | quem | orçamento | o que vê |
|---|---|---|---|
| `r1` | subagente | 60 | brief + atlas + pesos + avaliador (braço **M→A**) |
| `a0` | subagente, mesmo brief | 60 | **sem atlas**: só pesos + avaliador (controle de H4: o mapeamento ajuda?) |
| `rnd` | `scripts/m2a_random.py` | 40 | busca aleatória (1–2 rank-1, camada/tipo/α/cols aleatórios), sem agente |

Orçamento reduzido de 100 para 60 chamadas por custo de tempo (~52 s por avaliação).

**Confirmação** (`scripts/queue_confirm.sh`): o nº 1 de cada run vira GGUF e é medido no dev48, pareado com
base e astra.

**Previsões pré-registradas** (antes de ver qualquer resultado):
1. `rnd` não passa de branch_rep −0,02 dentro dos limites. Edições aleatórias são quase nulas,
   como no PoC-7.
2. `r1` acha um candidato com branch_rep ≤ −0,05 dentro dos limites (pelo menos metade do Astra).
   Ele é confirmado por geração com ganho ≥ +15 pp no dev48.
3. **H4** (mapeamento ajuda): r1 > a0 em branch_rep e no ganho no dev48. Se a0 chegar perto de r1,
   o que importa é a busca com o avaliador, não o atlas.
4. O holdout lacrado dos campeões fica a até 30% do branch_rep de busca. Se ficar muito pior, houve
   overfitting aos 12 pares de busca (Goodhart).

### M→A, resultados da busca (avaliador; ainda sem confirmação por geração)

Auditoria: os transcritos dos agentes não mostram leitura proibida. O a0 nunca abriu o atlas.

| run | chamadas | melhor branch_rep, nº 1 do final.json | holdout_rep (lacrado) | holdout_prod | close | neutral |
|---|---|---|---|---|---|---|
| astra (ref.) | — | −0,099 | −0,115 | −0,005 | +0,15 | −0,002 |
| **r1** (M→A) | 47/60 | **−2,03** (201939_974: 1 edição, L12 linear_attn.out_proj, 320 cols, α=10) | ver abaixo | — | −0,23 | +0,008 |
| a0 (sem atlas) | 48/60 | −1,14 (201707_653: 4 edições em L15/L23 self_attn + L23 mlp) | −0,44 | **−0,031** (fora do limite) | −0,02 | −0,008 |
| rnd | 40/40 | −0,014 | −0,012 | −0,016 | +0,82 | −0,008 |

Top do r1 por branch_rep, com o holdout lacrado de cada um:

| candidato | busca | holdout |
|---|---|---|
| L14 + L15mlp | −2,71 | −2,34 |
| L14, α=10, 128 cols | −2,61 | −2,24 |
| L12 + L14(−8) | −2,36 | −1,84 |

O efeito do r1 generaliza para os pares novos, com holdout em torno de 85% do valor de busca.
O do a0 cai para cerca de 40%: é overfitting ao template da busca.

**Os dois agentes acharam mecanismos diferentes, e diferentes do Astra (L15/L19 MLP, −0,1):**
- **r1, com atlas:**
  - Achou um caminho de "estado de re-checagem" na atenção linear (Gated DeltaNet) do meio da
    rede: L10 → L12 → L13 → L14, com cabeças 14, 7, 9, 12 e 11 na L14.
  - Construiu u = norm(W·Δa_branch) + norm(W·Δa_natural) a partir da decomposição exata do atlas,
    e escolheu as colunas pela contribuição para a repetição.
  - Inverteu essa componente com α=10. A direção sobe nos 18 de 18 traços naturais.
  - O controle com u aleatório nas mesmas colunas dá ≈ 0.
  - Empilhar inversões ao longo do caminho se cancela: a inversão de cima vira o sinal da de baixo.
  - Ressalva do próprio agente: parte do efeito pode ser só a quebra da cópia de tokens fáceis.
- **a0, sem atlas:**
  - Achou um **anti-cópia por token**: inverte as direções de unembedding de "Yes" e "?" no o_proj
    da L15 (α=10 duas vezes, mudança relativa de ~0,3 cada) e de "Wait" na L23.
  - É um ataque ao template da sonda, cujas frases repetem "Wait, ... Yes, ...", e não a um estado
    interno. O holdout pela metade e o holdout_prod fora do limite são coerentes com isso.
  - Também usou duas edições com o mesmo tensor para passar do α efetivo de 10.

**Previsões:**
- (1) Confirmada: o rnd ficou em −0,014.
- (2), (3) e (4) ficam para a geração no dev48: r1 nº 1, a0 nº 1 e r1 192156_139 (L14 + L15mlp,
  outra alavanca).
- Ponto em aberto para a geração: um efeito de −2 nats/token pode ser "não consigo repetir nenhuma
  frase", e não "não entro em loop". Se a edição atrapalhar o reuso legítimo de números ou de
  frases, a acurácia cai mesmo com o loop menor.

### M→A, confirmação por geração: o nº 1 do r1 FALHA (Goodhart da métrica branch)

dev48 (GSM8K train[0:48], teto 12k):

| braço | acc | loop no teto | ganho pareado vs base |
|---|---|---|---|
| base | 0,146 | 0,85 | — |
| astra | 0,458 | 0,23 | +31 ± 7 pp (16/1) |
| **m2a_r1** (L12 linear_attn, α=10, branch_rep −2,03) | **0,042** | 0,79 | **−10 ± 5 pp** (1/6) |

**Previsão (2) refutada para o nº 1 do r1.**

Diagnóstico pelos textos: a edição impede a repetição **literal**, mas o loop continua. O modelo
passa a entrar em loops que variam:
- alterna valores: "Wait, if she reads 78 pages / … 42 pages / … 78 pages";
- gera sequências que crescem: "If she sold 3072000 to 192000 / 384000 / 768000 … friends";
- especula sobre o enunciado ("Why mention Yesterday?").

A métrica branch, tanto a de busca quanto a do holdout lacrado, mede só a probabilidade de
**copiar literalmente** uma frase de re-checagem ×4. Por isso não distingue "não entra em loop" de
"não consegue copiar a frase".

**Lição metodológica:**
- A validação da métrica (Spearman 0,89) vale só dentro da família de edições em que ela foi
  calibrada: edições tipo Astra, com efeito pequeno (|Δ| ≤ 0,1).
- Um otimizador forte, o agente com 60 chamadas, sai dessa região e acha o atalho: anti-cópia
  com efeito 20× maior.
- O holdout lacrado não protege, porque tem o mesmo formato.

O que isso pede:
- (a) sondas de loop **não literal** (paráfrases, números variando);
- (b) um teto de dose;
- (c) confirmação por geração **dentro** do laço do agente, ainda que barata (poucos problemas,
  teto curto).

**Próximos testes:**
- a0 nº 1 e r1 192156_139 (L14 + L15mlp) no dev48.
- **Hipótese de dose** (`scripts/m2a_dose.py`, `queue_dose.sh`, run `dose`, 9 chamadas):
  a mesma direção do r1 (L12 ou L14), em α pequeno com branch_rep ≈ −0,15 (escala do Astra),
  ajuda na geração?
  - Se sim, a direção é boa e o problema era a dose.
  - Se não, a direção do r1 só captura cópia literal e não é a do loop.

### M→A, confirmação por geração: resultados completos (dev48)

| braço | branch_rep | holdout | acc | cortado no teto | loop no teto (8-grama) | ganho pareado |
|---|---|---|---|---|---|---|
| base | 0 | 0 | 0,146 | 0,85 | 0,85 | — |
| astra | −0,099 | −0,115 | **0,458** | 0,48 | 0,23 | +31 ± 7 pp (16/1) |
| **a0 nº 1** (anti-cópia de "Yes", "?" e "Wait"; sem atlas) | −1,14 | −0,44 | **0,375** | 0,54 | 0,50 | **+23 ± 7 pp (12/1)** |
| r1 192156_139 (L14 + L15mlp) | −2,21 | — | 0,229 | 0,77 | **0,00** | +8 ± 5 pp (5/1) |
| r1 nº 1 (L12) | −2,03 | — | 0,042 | 0,96 | 0,79 | −10 ± 5 pp (1/6) |

**Leitura (n = 1 agente por braço, então são indícios, não conclusões):**
1. **A busca com avaliador funciona.** Um agente sem atlas, só com pesos e avaliador, achou em 48
   chamadas uma edição com +23 pp no GSM8K, cerca de 3/4 do ganho do Astra. A tese central
   (um modelo de fronteira guiando edição direta de pesos substitui o fine-tuning nesta tarefa)
   ganha um segundo ponto independente, feito por outro agente, sem acesso à solução do Astra.
2. **H4 não se confirmou nesta amostra.**
   - O agente **com** atlas (r1) achou direções "internas" (estado de re-checagem na DeltaNet).
     Elas generalizam melhor nas sondas, mas pioraram (nº 1) ou ajudaram pouco (nº 3) na geração.
   - O agente **sem** atlas achou um atalho por token. Ele generaliza mal nas sondas e ajuda muito
     na geração.
   - Hipótese: os loops reais **são** feitos desses tokens ("Wait", "Yes", "?"). Uma anti-cópia
     de tokens de re-checagem desfaz o loop, e o modelo segue para a resposta em vez de variar.
     A direção do r1 barra a cópia de qualquer coisa, e o modelo contorna variando.
3. **O detector de loop (8-grama) também sofre Goodhart.**
   - O r1 nº 3 tem loop no teto 0,00, mas 77% das gerações ainda batem no teto: são loops não
     literais.
   - Daqui em diante, a métrica de geração primária é "cortado no teto", com a acurácia, e não o
     loop por 8-grama.
4. **A métrica branch ordena mal entre famílias de edição.** Entre os 4 candidatos novos, a ordem
   por branch_rep é praticamente a inversa da acurácia. Ela só servia dentro da família do Astra.

### Hipótese de dose: refutada

Direções do r1 em α=2, escolhidas para dar branch_rep na escala do Astra (dev48):

| edição | branch_rep | acc | cortado no teto | ganho pareado |
|---|---|---|---|---|
| astra | −0,099 | 0,458 | 0,48 | +31 ± 7 pp (16/1) |
| r1-L12, α=2 | −0,102 | 0,125 | 0,94 | −2 ± 6 pp (4/5) |
| r1-L14, α=2 | −0,134 | 0,208 | 0,81 | +6 ± 6 pp (6/3) |

Curvas de dose no avaliador: o branch_rep cresce de forma superlinear em α, de −0,10 (α=2) a −0,87
(α=6) na L12. O branch_prod e o neutral ficam praticamente intactos em toda a faixa.

**Com o mesmo branch_rep, as direções do r1 não fazem nada pela acurácia, enquanto a do Astra dá
+31 pp.**
- O problema não era a dose, e sim a **direção**. A direção do r1 captura "copiar a frase anterior",
  e isso não causa o loop.
- A métrica branch não identifica a direção causal entre famílias. Ela só ordenava variações em
  torno do Astra.

**Implicação para o desenho do avaliador:** as sondas branch servem como restrição e sinal de
segurança, mas não podem ser o objetivo primário. O objetivo precisa vir de geração, mesmo que
barata.

Proposta para o avaliador v2:
- a cada candidato, gerar 16 problemas difíceis do treino (os que o base corta), com teto de 4k;
- medir a fração que fecha o raciocínio e a acurácia;
- custo por avaliação: GGUF (~1 min) + geração (~3–5 min).
- Alternativa sem GGUF: geração em HF com cache, mais lenta na 1660S.

**Teste trancado do a0** (o melhor M→A confirmado no dev): `queue_test_a0.sh`.

### Teste trancado do a0 (GSM8K test[0:32], teto 12k; o mesmo conjunto do t12k)

| braço | acc | cortado no teto | ganho pareado vs base |
|---|---|---|---|
| base | 0,125 | — | — |
| astra | 0,406 | 0,48 | +28 ± 10 pp (11/2) |
| **m2a_a0** | **0,438** | **0,41** | **+31 ± 8 pp (10/0)** |

**Resultado principal do braço M→A:**
- No teste trancado, a edição achada pelo agente sem atlas **empata com a do Astra** (0,44 × 0,41;
  a diferença está dentro do erro) e não tem nenhuma perda.
- Ela foi achada por um agente com orçamento de 60 chamadas (usou 48), sem acesso à solução,
  só com pesos + avaliador, em ~2 h numa GTX 1660S.
- Ressalvas:
  - n = 32 problemas;
  - um agente só;
  - a edição é grande (mudança relativa de ~0,3 em duas matrizes da L15);
  - os colaterais (tool calling, GSM8K sem thinking, WikiText em geração) ainda não foram medidos;
  - a edição é específica de tokens ("Yes", "?" e "Wait"), então a transferência para outros
    estilos de loop e para o MATH-500 está em aberto.

### Colaterais e transferência do m2a_a0 (2026-11-05 madrugada)

| painel | base | astra | **m2a_a0** | a0 vs base (pareado) |
|---|---|---|---|---|
| tool calling sem thinking (dev300) | 0,897 | 0,870 | **0,933** | +3,7 ± 1,6 pp (17/6) |
| GSM8K sem thinking (n200) | ~0,485 | ~0,48 | 0,465 | −2 ± 4 pp (ruído) |
| tool calling com thinking (dev100) | 0,93 | 0,98 | 0,94 | +1 ± 3 pp (4/3); irrelevância 13/16 (base 11/16, astra 16/16) |
| **MATH-500** (48 problemas, teto 12k, test[0:48]) | 0,042 | 0,208 (+17 ± 5, 8/0) | **0,271** | **+23 ± 6 pp (11/0)** |

- **Nenhum dano colateral detectável.** O tool calling sem thinking até melhora um pouco: a
  anti-cópia de "Yes" e "?" talvez reduza a eco do enunciado.
- **A edição transfere para o MATH-500**, a tarefa do paper, sem ter sido buscada nele, e com zero
  perdas.
  - No nosso teto de 12k o base marca 4%. O paper reporta 7,9% com teto maior.
  - O astra marca +17 pp, e o a0 marca +23 pp.
- A ressalva de que a edição age sobre tokens específicos continua valendo, mas no MATH os loops
  também são feitos de "Wait"/"Yes"/"?", então ela pega.
- O corretor do MATH é leve (`math_eval.py`), e os textos brutos ficam em `data/gen_math48/`
  para recorreção.

### Avaliador v2 (geração no laço) e réplicas de H4 (2026-11-05, 06h)

**Código:**
- `m2a.submit_gen(run, cid)` envia o pedido.
- O ramo de geração no `m2a_scorer.py`:
  - tira o HF da GPU, monta o GGUF com o poc7_build e sobe o llama-server na porta 8091;
  - gera GSM8K train[200:216], com thinking, teto 6144;
  - devolve acc, cut, rescues/losses vs base e os finais das respostas cortadas.
- Orçamento separado (`gen_budget`).
- Brief: `BRIEF_v2.md`. A sonda deixa de ser o objetivo e ganha uma seção de lições das runs
  anteriores, igual para os dois braços.

**Sanidade v2** (run `sanity2`, candidato do a0):

| modelo | acc | cortado no teto | pareado |
|---|---|---|---|
| base (referência, `gen_ref_base.json`) | 1/16 | 100% | — |
| a0 | 7/16 | 56% | 6 resgates, 0 perdas |

~10 min por checagem.

**Réplicas:**
- r2 (com atlas) e a1 (sem atlas), em paralelo.
- Orçamento de cada uma: 40 sondas + 6 checagens por geração.

**Previsões (H4):**
- Se o mapeamento ajuda, r2 ≥ a1 no dev48.
- Se a busca com geração no laço é o que importa, os dois chegam a ≥ +20 pp, e a diferença entre
  braços fica menor que a diferença entre réplicas.
- O critério segue sendo a confirmação pelo supervisor no dev48 e no teste trancado.

### Réplicas r2 e a1: resultados da busca (checagem por geração: GSM8K train[200:216], teto 6k; base 1/16, cut 100%)

| run | nº 1 | mecanismo | checagem por geração | sondas usadas |
|---|---|---|---|---|
| **a1** (sem atlas) | 074537_659 | supressão de "Wait" por 3 neurônios de MLP que escrevem "Wait" (L17n1915, L18n881, L22n573; α=−9 no eixo do unembedding de Wait) + reforço do `</think>` na L23 mlp (α=−1,1) | **11/16**, cut 0,125, 10 resgates/0 perdas, ~2k tokens | 18/40, 6/6 checagens |
| **r2** (com atlas) | 072949_822 | 3 edições de "cabeças de loop" achadas pelo atlas (L11 h1-2, L14 h7, L15 h6-7; α=2) + reforço do `</think>` em 32 neurônios da L23 escolhidos pelo atlas (α=−6) | 6/16, cut 0,625, 5 resgates/0 perdas | 20/40, 6/6 checagens |

**Convergência (três agentes independentes):**
- O reforço do `</think>` (a saída) apareceu tanto no a1 quanto no r2.
- A supressão de "Wait"/"?"/"Yes" (a reentrada no loop) apareceu no a0 e no a1.
- A anatomia que o a1 descreve: o loop exige um caminho de volta (Wait) e nenhum de saída.
  - Sozinha, cada metade ajuda pouco: no a1, close só dá 3/16 e Wait só dá 7/16.
  - Juntas, dão 11/16.

**A checagem por geração no laço pegou Goodhart nos dois agentes:**
- No a1, a negação dos neurônios de "Wait" tinha as melhores sondas e gerou "wait wait wait…",
  com 0/16.
- No r2, as cabeças sozinhas tinham branch_rep −0,43 e deram 0/16.

**Desvio de procedimento:** o r2 gravou arquivos fora da pasta da run por um caminho relativo
(`notes.md`, `cand_*.pt`) e os moveu para `r2/work/`. O conteúdo era só do próprio agente, sem nada
pré-existente. Briefs seguintes exigem caminhos absolutos.

**Auditoria de leituras proibidas:** limpa nos dois transcritos.

**Confirmação no dev48** (`queue_confirm2.sh`): a1 nº 1, r2 nº 1 e a1 062959_102 (variante mais branda).

### Réplicas: confirmação no dev48 (GSM8K train[0:48], teto 12k)

| braço | acc | cortado | tokens médios | ganho vs base | resgates/perdas |
|---|---|---|---|---|---|
| base | 0,146 | 0,85 | 11,7k | — | — |
| astra | 0,458 | 0,48 | 9,4k | +31 ± 7 pp | 16/1 |
| m2a_a0 | 0,375 | 0,54 | 8,9k | +23 ± 7 pp | 12/1 |
| **m2a_a1** (074537_659) | **0,646** | 0,04 | 1,9k | **+50 ± 8 pp** | 25/1 |
| **m2a_a1_062959_102** (branda) | **0,667** | 0,17 | 4,7k | **+52 ± 8 pp** | 26/1 |
| m2a_r2 (072949_822) | 0,354 | 0,60 | 9,8k | +21 ± 7 pp | 12/2 |

**Leitura:**
- O a1 **supera o Astra** no dev, com cerca de +20 pp pareado (ver a linha abaixo).
- Ele também passa do GSM8K **sem** thinking (~0,48). O raciocínio volta a ajudar em vez de
  atrapalhar.
- A checagem por geração no laço (v2) mudou a qualidade do que os agentes acham: v1 a0 +23 pp,
  v2 a1 +52 pp.
- Previsão de H4 (r2 ≥ a1): **refutada** de novo.
  - Com atlas: r1 (−10 / +8 pp) e r2 (+21 pp).
  - Sem atlas: a0 (+23 pp) e a1 (+52 pp).
  - Em 2 de 2 pares, o braço sem atlas foi melhor. O n ainda é pequeno, mas a direção é
    consistente.
  - Hipótese: o atlas puxa o agente para mecanismos "internos" (estado, cabeças), enquanto a análise
    direta dos pesos pelo vocabulário (logit lens dos neurônios) acha alavancas de saída
    ("Wait", `</think>`) que controlam o comportamento mais diretamente.

**Teste trancado + MATH-500 + colaterais** da variante branda (melhor no dev e dentro de todos os
limites das sondas): `queue_a1.sh`.

## 2026-11-05 — a1 (variante branda 062959_102) no teste trancado, MATH-500 e colaterais

Pré-registrado: o melhor confirmado no dev48 vai para o teste trancado (escolhido antes de ver o teste).

**Teste trancado t12k** (GSM8K test[0:32], thinking, teto 12k):
| braço | acc | cut | tokens | vs base | vs astra | vs a0 |
|---|---|---|---|---|---|---|
| base | 0.125 | 0.88 | 11.3k | — | | |
| astra | 0.406 | 0.44 | 9.0k | +28 ± 10 | — | |
| m2a_a0 | 0.438 | 0.41 | 8.0k | +31 ± 8 | | — |
| **m2a_a1_062959_102** | **0.562** | **0.16** | **5.6k** | **+44 ± 10 (15/1)** | **+16 ± 8 (6/1)** | +12 ± 7 (5/1) |

**MATH-500[0:48]** (teto 12k; tarefa do paper, fora da distribuição de busca, que foi GSM8K):
| braço | acc | cut | vs base | vs astra |
|---|---|---|---|---|
| base | 0.042 | 0.94 | — | |
| astra | 0.208 | 0.77 | +17 | — |
| m2a_a0 | 0.271 | 0.69 | +23 ± 6 | |
| **m2a_a1** | **0.396** | 0.52 | **+35 ± 7 (17/0)** | **+19 ± 7 (11/2)** |
Por nível (1..5): a1 2/4, 6/12, 6/12, 2/7, 3/13; astra 1/4, 5/12, 4/12, 0/7, 0/13.

**Colaterais** (a1): tool nothink dev300 0.927 (+3 ± 1.6 vs base; irrelevância 0.898 vs 0.918 base);
GSM8K nothink n200 0.485 (= base, 25/25); tool think dev100 0.94 (+1; irrelevância 11/16 = base, enquanto
astra 16/16 e a0 13/16). Sem dano detectável; a1 não herda o ganho de irrelevância do astra.

**Leitura.** A edição achada pelo agente (sem atlas, scorer v2 com geração no laço; 3 neurônios anti-"Wait" +
reforço de `</think>` em L23) supera a edição do Astra com folga no teste trancado e no MATH-500, com o mesmo
espaço de edição (≤4 rank-1). Ressalvas: n pequeno (32 e 48; SE ~7–10 pp), uma réplica do braço vencedor
(a1 nº 1 ficou em 0.646 no dev, então o resultado parece replicável dentro do braço), e o agente teve
16 problemas GSM8K de checagem por geração que o Astra não teve (vantagem de protocolo, não só de agente).

## 2026-11-06 — Paper 1, semana 1: E1 (decodificação) e E2 (réplicas). PRÉ-REGISTRO

**Pergunta E1:** o ganho da edição a1 (anti-"Wait" em 3 neurônios + reforço de `</think>`) é só um logit bias?
Braços de decodificação sobre o base (código: `evalrun.decode_extra`, `llama_gen` think_budget):
- `lb_w10_t0` (bias −10 em "Wait"/" Wait", ~ NoWait), `lbx_w10_t1` (também "wait"/" wait"; +1 em `</think>`),
  `lb_w3_t1` (−3 / +1);
- `bf4096`, `bf8192`: budget forcing (no limite, injeta "\n</think>\n\n" e responde até o teto de 12k).
Ajuste no dev48 (48 problemas, mais do que os 16 de checagem que o agente teve: vantagem para o baseline).
O melhor de cada família vai para t12k e math48 junto com o a1.
**Previsões:** (1) o melhor logit bias fica abaixo do a1 no t12k e no MATH (a edição é condicional ao contexto);
se empatar dentro de 1 SE, a tese do paper muda (ver plano). (2) budget forcing melhora o base (resgata quem
já tinha a conta antes do loop), mas fica abaixo do Astra no MATH.
**E2:** réplicas a2 (sem atlas) e r3 (com atlas), mesmo protocolo v2 (40 sondas + 6 gerações), confirmação no dev48.
Previsão: a2 > r3 (H4 refutada de novo); a2 ≥ +35 pp sobre o base no dev48.
Fila: `scripts/queue_e1.sh` (unit wsp-e1).

### 2026-11-06 03:22 — a2 (sem atlas) terminou
Melhor `015739_897`: remoção da direção "Wait" (logit lens) em neurônios únicos L16n998(+n3422), L17n1915, L22n573
(α=10) + reforço de `</think>` em L22 linear_attn.out_proj (α=−4). Checagem por geração: 11/16 (base 1/16), cut 0.06,
2.9k tokens. Sondas: rep −0.33, prod +0.034, close +0.68, neutral −0.0095. Orçamento: 17/40 sondas, 6/6 gerações.
**Convergência:** a2 achou de forma independente o MESMO mecanismo do a1 (neurônios-"Wait" L17n1915 e L22n573 em
comum; L16n998 novo, que o a1 viu no scan mas não usou) + alavanca de `</think>`. Ablação em geração do a2 reproduz a
sinergia do a1 (só anti-Wait 0.50/cut 0.50; só `</think>` 0.19/cut 0.81; ambos 0.69/cut 0.06).
Falhas: suprimir "Actually" piora (é rota de saída, não marcador de loop); neurônios-"Yes" estouram as sondas em combinação.

### 2026-11-06 ~04:00 — r3 (com atlas) terminou
Melhor `034502_932`: inversão (α=4) de uma direção residual "loop/re-checagem" (empurra "?", "Yes", "Okay"; puxa
"thus/therefore") nas 256 colunas que mais a escrevem, em L13/L14 linear_attn.out_proj + L15/L19 down_proj.
Checagem por geração 5/16 (base 1/16), cut 0.69; sondas limpas (rep −0.39, prod −0.002, close +0.26, neutral +0.010).
Orçamento 23/40 sondas, 6/6 gerações. Falhas: C2 (só linear attention) teve a sonda mais forte (−0.54) e 1/16 em
geração (Goodhart de novo); atlas mostrou que o system prompt muda o interno de forma ~ortogonal à direção de loop
(cos 0.999 entre rep−prod sob os 4 prompts) — explica por que prompts não resolvem.
**H4 pela 3ª vez:** sem atlas 11/16 (a2) vs com atlas 5/16 (r3) na checagem por geração. Padrão estável: o atlas leva
a mecanismos de estado interno (DeltaNet L12–15) que reduzem a cópia literal mas não fecham o raciocínio; sem atlas o
agente acha as alavancas de saída ("Wait" + `</think>`).
**Auditoria:** varredura das chamadas de ferramenta das duas transcrições por caminhos proibidos (docs, refs,
scripts, probes.py, sealed, gen_ref_base, outras runs, ~/.claude, checkpoints do paper): nenhuma violação
(única ocorrência: r3 listando atlas.pt, permitido para o braço com atlas).

## 2026-11-06 manhã — E1 resultados (parcial: falta bf4096 no MATH e a confirmação a2/r3)
**Varredura dev48** (vs base 0.146): lb_w10_t0 0.396 (+25); **lbx_w10_t1 0.521 (+38)**; lb_w3_t1 0.396 (+25);
**bf4096 0.479 (+33)**; bf8192 0.479 (+33). a1 0.667. Escolhidos: lbx_w10_t1 e bf4096.

| a1 (062959_102) vs | dev48 | t12k (trancado) | math48 |
|---|---|---|---|
| logit bias lbx_w10_t1 | +15 ± 7 (10/3) | +9 ± 8 (5/2) | **+12.5 ± 5.7 (7/1)** |
| budget forcing bf4096 | +19 ± 8 (12/3) | **−6 ± 6 (1/3)** | ver 12:46 |
| astra | +21 ± 7 | +16 ± 8 | +19 ± 7 |
Absolutos t12k: base 0.125, astra 0.406, lbx 0.469, a1 0.562, **bf4096 0.625**. MATH: lbx 0.271, a1 0.396.

**Previsão 1 (logit bias < a1): confirmada** nos três conjuntos (significativa no MATH e no dev; no t12k dentro do ruído).
A edição não é só um logit bias.
**Previsão 2 (budget forcing < astra no MATH): pendente;** no GSM8K o budget forcing é um baseline forte (t12k 0.625,
melhor número absoluto do t12k). No GSM8K a resposta muitas vezes já foi calculada antes do loop, e fechar à força
recupera. 9/32 seguem cortados no t12k mesmo após forçar (loop na resposta).
**Leitura:** dev48 e t12k discordam sobre a1 vs bf (+19 vs −6); com n=48/32 isso é ruído plausível. Budget forcing
e a edição são ortogonais (controle de decodificação vs mudança de pesos) → testar **a1 + bf** é o passo natural.

### 2026-11-06 12:46 — E1 completo + confirmação das réplicas
**MATH48:** bf4096 0.354 (astra 0.208, lbx 0.271, a1 0.396). a1 vs bf4096 +4 ± 6 (5/3). **Previsão 2 refutada:**
budget forcing supera o Astra também no MATH.
**a1 vs bf4096 fora do dev (t12k + math48, 80 problemas): 6 resgates / 6 perdas → empate.** No dev48 (onde a1 foi
escolhido) +19. Conclusão honesta: **a edição a1 ≈ budget forcing ajustado; ambas > logit bias > Astra.**
A afirmação "a edição supera truques de decodificação" vale contra logit bias, NÃO contra budget forcing.
**Réplicas no dev48:** a2 (sem atlas) 0.646 (+50 ± 7, 24/0; vs astra +19 ± 6; vs a1 −2, empate); r3 (com atlas)
0.396 (+25 ± 8; vs a1 −27, 0/13). H4 refutada no dev48 nos 3 pares (a0>r1, a1>r2, a2>r3).
Sem-atlas v2 (a1, a2): 0.667 e 0.646 → o resultado do agente replica.
Próximos passos: a1 + bf4096 (complementares?), E3 (busca aleatória com geração), n maior.

## 2026-11-06 tarde — a1 + budget forcing (PRÉ-REGISTRO)
Braço `m2a_a1_062959_102+bf4096` (bf4096 escolhido no base; não reajustado para o a1). dev48, t12k, math48.
**Previsão:** complementares → combo ≥ max(a1, bf4096) em t12k e math48 (soma fora do dev ≥ +5 pp vs o melhor dos dois).
Risco: o a1 já fecha cedo (4.7k tokens médios no dev); o forcing só atua em quem não fechou até 4096 (inclui os ~17% que chegam ao teto), então o
ganho esperado é pequeno — se vier, vem dos loops residuais pós-edição.

### 2026-11-06 15:37 — resultado a1 + bf4096
| | dev48 | t12k | math48 |
|---|---|---|---|
| combo a1+bf4096 | **0.688** (cut 0.08, 3.8k tok) | **0.625** (cut 0.09) | **0.500** (cut 0.35) |
| a1 | 0.667 | 0.562 | 0.396 |
| bf4096 | 0.479 | 0.625 | 0.354 |
| astra | 0.458 | 0.406 | 0.208 |
Pareado combo vs: a1 dev +2 (4/3), t12k +6 (4/2), math +10 ± 6 (7/2); bf4096 dev +21 (11/1), t12k 0 (4/4),
math +15 ± 7 (9/2); astra +23 / +22 / +29 (15/1 no math).
**Fora do dev (t12k + math48, 80 problemas): combo vs a1 11/4 (+9 pp), vs bf4096 13/6 (+9 pp), vs astra 23/2.**
**Previsão (complementares, ≥ +5 pp sobre o melhor dos dois fora do dev): confirmada**, com efeito modesto
(SE ~5 pp no agregado). O ganho aparece sobretudo no MATH (problemas longos): a edição tira do loop por dentro e o
forcing recolhe quem ainda passa de 4096 sem fechar. No GSM8K o combo = bf4096 (os dois já resolvem o mesmo subconjunto).

## 2026-11-06 noite — E3 (controle de protocolo) e MATH maior. PRÉ-REGISTRO
**E3:** busca ingênua (`scripts/m2a_naive.py`, run `naive`) com as MESMAS ferramentas que os agentes vencedores
usaram (direções do logit lens de tokens frequentes nas saídas do base, colunas por |cos|, camadas 12–23), sorteadas,
e o mesmo orçamento (40 sondas; as 6 melhores sondas dentro dos limites vão para a geração). Melhor pela geração → dev48.
**Previsão:** naive < a1/a2 por ≥ 25 pp no dev48 (o mérito é do raciocínio do agente, não só do sinal de geração).
Se naive ≥ 0.55 no dev48, o mérito é do protocolo/ferramentas e o agente é dispensável.
**MATH-500[48:148]** (100 problemas novos) para combo, a1, bf4096, astra, base → total 148 com o math48.
**Previsão:** a ordem combo > a1 ≥ bf4096 > astra se mantém; combo − a1 ≥ +5 pp no agregado de 148.
Script: `scripts/queue_e3_math.sh`.

### 2026-11-07 02h — E3 e MATH-148 (parcial: falta astra e base no math100b)
**E3 (busca ingênua, mesmo orçamento e ferramentas):** melhor sonda só −0.007 de branch_rep; checagens por geração
1–3/16; escolhido `180935_606` → **dev48 0.271** (+12.5 ± 6 vs base, 8/2). a1 0.667, a2 0.646 → **−38 a −40 pp.**
**Previsão (naive ≥ 25 pp abaixo): confirmada.** O sinal de geração sozinho não basta: o mérito é do raciocínio do
agente (escolher "Wait"/`</think>`, ler a sinergia nas falhas).
**MATH-500[0:148]** (math48 + math100b): combo **0.527**, a1 **0.466**, bf4096 **0.372** (base e astra no math100b: ver a entrada final).
Pareado: combo vs a1 **+6.1 ± 3.2 (16/7)**; combo vs bf4096 **+15.5 ± 3.6 (27/4)**; a1 vs bf4096 **+9.5 ± 3.8 (23/9)**.
Previsão (combo − a1 ≥ +5 pp): confirmada (borda, ~1.9 SE). No MATH a edição SUPERA o budget forcing de forma
significativa (2.5 SE) — o empate do t12k era específico do GSM8K (respostas curtas, conta pronta antes do loop).

## 2026-11-07 — E4.0–4.2 (baseline de fine-tuning). PRÉ-REGISTRO
**E4.0** `ft_spike.py`: LoRA r16 (peft 0.21.2, instalado no venv; 10,2M parâmetros, todas as projeções lineares),
base fp16 + adaptador fp32, gradient checkpointing, perda em blocos com checkpoint (não materializa logits de 248k),
GradScaler. Mede VRAM/tempo por passo em T = 1k..8k; max_len = maior com ≤ 45 s/passo.
**E4.1** `ft_data.py`: base no GSM8K train[300:900], 1 amostra, teto 6144, amostragem da avaliação. Mantém corretos:
*natural* (fechou sozinho, sem loop) e *excised* (corta no início do loop duro, injeta `</think>`, base responde).
Não toca dev48 (0:48) nem a checagem do agente (200:216).
**E4.2** `ft_train.py`: 3 épocas, lr 1e-4 (aquecimento + cosseno), acumulação 8, perda só na resposta, sequências
> max_len descartadas. `ft_export.py`: mescla em fp32 → GGUF f16 (teste: 150 tensores, 2% da atualização perdida
no arredondamento f16). Época escolhida nos MESMOS 16 problemas de checagem do agente → dev48.
**Previsões:** (1) dados: natural raro (~5–10% das 600), excised ~25–40%. (2) LoRA-SFT melhora o base no dev48
(≥ +20 pp), mas fica abaixo do a1 (0.667). (3) custo: ≥ 5× as GPU-horas do agente.
Script: `scripts/queue_e4.sh`.

## 2026-11-07 — MATH-148 final (math48 + math100b, MATH-500[0:148], teto 12288)

| braço | acc |
|---|---|
| a1 + bf4096 (combo) | 0.527 |
| a1 (m2a_a1_062959_102) | 0.466 |
| bf4096 | 0.372 |
| astra | 0.209 |
| base | 0.081 |

Pareados: a1 vs astra +25.7 ± 4.3 (44/6); a1 vs base +38.5 ± 4.1 (58/1); combo vs base +44.6 ± 4.1 (66/0);
bf4096 vs astra +16.2 ± 3.3 (26/2); astra vs base +12.8 ± 2.8 (19/0). Com os anteriores: combo vs a1 +6.1 ± 3.2,
combo vs bf +15.5 ± 3.6, a1 vs bf +9.5 ± 3.8. Ordem estrita combo > a1 > bf > astra > base, todas as diferenças
adjacentes ≥ 1.9 SE.

## 2026-11-07 — E4.0 tentativa 1: OOM (bug, não limite)

OOM já em T=1024. Causa: o gradient checkpointing do transformers só age com `self.training=True`, e o
`from_pretrained` devolve o modelo em eval: o forward guardava 3 GB de ativações a T=1024. Correção: `model.train()`
em load_lora (dropout é 0, nada mais muda) e blocos de logits de 256 (o backward recomputa cada bloco: 1024 x 248k
x ~10 B ≈ 2.5 GB). Perfil depois da correção: T=4096 com pico 2.3 GB só no tronco. Checagem da perda em blocos: igual à direta
(0.288003 = 0.288003).

## 2026-11-07 — Queda de energia (~08:45) e retomada do E4

- Uma queda de energia interrompeu o E4.1 em 200/600 (fase A). O disco de dados ficou com setores ilegíveis só no
  journal do sistema de arquivos; o fsck isolou os blocos. Nenhum dado de pesquisa foi perdido: os JSON/JSONL do
  projeto fazem parse sem truncamento, os checkpoints do base, do Astra e o GSM8K conferem com o SHA-256, e os 21 GGUFs
  foram lidos inteiros sem erro. O único arquivo danificado foi um log do llama-server.
- O E4.1 retomou do checkpoint por problema; o E4.0 não precisou ser refeito.
- **Desvio registrado ANTES do treino:** max_len do treino 4096 → 8192. Motivo: nas 200 amostras já geradas, o
  limite de 4096 descartaria 7/17 traços naturais e 65/103 prefixos de loop (onset mediano 4186), enviesando o
  baseline para traços curtos e enfraquecendo-o. Em 8192 cabe tudo (pico 3,14 GB, 81 s por sequência).
  Parcial fase A (200): naturais 17 (8,5%, dentro da previsão de 5–10%), loops 103, cortados 183.

## 2026-11-08 — Repositório próprio do paper

- O projeto saiu do `deep_traces` e tem repositório próprio: `luizmlo/agentic-weight-editing` (privado). O pacote passou
  a ser `wsp` (antes `src/deep_traces/wsp`), os scripts ficam em `scripts/` e os dados em `data/`.
- **Os caminhos deste caderno e do PLANO foram retificados para o layout novo.** Só mudaram os caminhos, não o
  conteúdo. Os registros literais dos agentes (BRIEFs, notas, ledgers) ficam como estavam, com os caminhos antigos,
  porque são o que cada agente viu; ver `data/m2a/README.md`.
- Venv próprio com as mesmas versões fixadas (torch 2.14.1+cu130, transformers 5.22.1, fla 0.6.0, peft 0.21.2).
- Documentação em inglês (README, docs/research-notes.md, experiments.md, paper-draft.md, reproducing.md). Este
  caderno continua sendo o registro primário, em pt-BR.
- Revisão de estilo do caderno: notas operacionais (estado de filas, horários de lançamento, reinícios de servidor)
  foram removidas ou neutralizadas. Números, previsões, desvios e datas das entradas ficaram intactos; o texto
  original está no primeiro commit do repositório.

## 2026-11-08 — E4.1 dados: resultado (`data/ft/sft_stats.json`)

| fase A (600 amostras, teto 6144) | n | % |
|---|---|---|
| natural (fechou sozinho, sem loop duro, correto) | 57 | 9,5% |
| loop duro com onset ≥ 128 (vai para a excisão) | 279 | 46,5% |
| outros (cortado no teto sem loop duro detectado, onset < 128, ou fechou e errou) | 264 | 44,0% |

- Na parcial, 90% das amostras bateram no teto de 6144 e só 11% fecharam o `</think>`.
- Fase B (excisão): 118 dos 279 prefixos de loop levaram a uma resposta correta (42%), ou seja 19,7% das 600.
- **Conjunto de SFT:** 175 sequências (57 naturais + 118 excised), todas ≤ 8192 tokens, então nenhuma foi
  descartada. São 777 mil tokens de resposta: naturais com mediana de 3619 tokens, excised com mediana de 4964
  (máximo 6398).
- **Previsão (1):**
  - natural: 9,5%, dentro da faixa de 5–10%;
  - excised: 19,7%, **abaixo** dos 25–40% previstos.
  O conjunto ficou menor que o esperado (175 contra ~200–300).
- **E4.2:** o treino começou com 3 épocas e 66 passos de otimização (acumulação de 8). O tempo do E4.0 (~9,4 s por
  mil tokens) dá cerca de 2 h por época.

## 2026-11-08 — E4.2 resultado: LoRA-SFT fica muito abaixo do a1

**Treino:** 66 passos em 372 min. A perda média por época foi 0,420, 0,389 e 0,376, sem instabilidade (grad_norm entre
0,2 e 0,6, scale final 16384). Os adaptadores das três épocas foram mesclados e exportados para GGUF f16
(`data/gguf/ft_sft_ep{1,2,3}.gguf`).

**Escolha da época** nos mesmos 16 problemas de checagem do agente (GSM8K train[200:216], teto 6144):

| | acc | cut | fechou `</think>` |
|---|---|---|---|
| base | 1/16 (0,062) | 0,94 | 0,06 |
| ep1 | 4/16 (0,250) | 0,75 | 0,31 |
| **ep2** | **7/16 (0,438)** | 0,56 | 0,50 |
| ep3 | 3/16 (0,188) | 0,81 | 0,25 |

ep3 piora em relação à ep2 com perda menor. Com 16 problemas a diferença está dentro do ruído (SE ≈ 12 pp), mas é
coerente com sobreajuste a 175 sequências.

**dev48** (GSM8K train[0:48], teto 12288), pareado com as mesmas sementes:

| braço | acc | cut | ft_sft_ep2 − braço |
|---|---|---|---|
| base | 0,146 | 0,85 | **+14,6 ± 5,9** (8/1) |
| m2a_naive (E3) | 0,271 | 0,73 | +2,1 ± 5,6 (4/3) |
| **ft_sft_ep2** | **0,292** | 0,71 | — |
| astra | 0,458 | 0,48 | −16,7 ± 6,2 (1/9) |
| bf4096 | 0,479 | 0,46 | −18,8 ± 7,1 (2/11) |
| lbx_w10_t1 | 0,521 | 0,35 | −22,9 ± 7,4 (2/13) |
| a2 | 0,646 | 0,08 | −35,4 ± 7,0 (0/17) |
| **a1 (062959_102)** | **0,667** | 0,17 | **−37,5 ± 7,1 (0/18)** |
| a1 + bf4096 | 0,688 | 0,08 | −39,6 ± 7,1 (0/19) |

- A LoRA quase não tira o modelo do loop: cut 0,85 → 0,71, contra 0,17 do a1. Um conjunto de SFT em que 2/3 das
  sequências ensinam "fechar depois de um prefixo pré-loop" não ensina o modelo a reconhecer quando cortar.
- A queda de 0,438 na seleção para 0,292 no dev48 é o viés de escolher a melhor de 3 em 16 problemas. Com o a1
  acontece o mesmo: 0,688 na checagem e 0,667 no dev48.

**Previsões:**
- **(1) dados:** natural 9,5%, confirmada. Excised: a previsão "~25–40%" foi escrita sem dizer a base. Se for sobre as 600
  amostras, deu 19,7%, abaixo. Se for sobre os loops, deu 42%, um pouco acima. Fica registrada como ambígua. A entrada
  anterior a tratou como % das 600.
- **(2) melhora ≥ +20 pp e fica abaixo do a1:** parcialmente confirmada. Melhorou +14,6, abaixo do limiar de +20, e ficou
  muito abaixo do a1 (0/18).
- **(3) custo ≥ 5× as GPU-horas do agente:** confirmada (estimativa preliminar, sem contar o dev48 de nenhum dos dois).

| | GPU-h (relógio, GPU ocupada) |
|---|---|
| E4.0 viabilidade | 0,2 |
| E4.1 fase A (600 amostras, teto 6144; 90% batem no teto) | 9,0 (3,1 antes da queda de energia + 5,9 depois) |
| E4.1 fase B (279 respostas pós-excisão) | 1,2 |
| E4.2 treino + exportação | 6,3 |
| escolha da época (3 × 16 problemas) | 0,5 |
| **total E4** | **≈ 17** |
| run de agente v2 sem atlas (avaliador residente; a1: 06:03–08:04, a2: 01:23–03:21) | ≈ 2 (≈ 1,3 de GPU ativa: 17–18 sondas × 52 s + 6 checagens × ~10 min) |

A razão é de ≈ 8× em GPU-hora. A contabilidade não inclui os tokens do agente (Claude Opus 5.5 via Claude Code, sem
custo de API medido) nem o tempo humano. O custo da LoRA é dominado pela geração dos dados, que é cara justamente por
causa do defeito: 90% das amostras batem no teto.

**Decisão:** o E4.3 (avaliação completa da LoRA: t12k, MATH-148, colaterais) não é necessário para a conclusão
principal, porque no dev48 a LoRA está 37,5 pp abaixo do a1 e abaixo até dos baselines de decodificação. Fica
**opcional**: o t12k custaria uma abertura do teste trancado para um braço que já perdeu no dev.

**Limites desta comparação:** uma única configuração de SFT (r16, lr 1e-4, 3 épocas, 175 sequências, 1 amostra por
problema). Mais amostras por problema, um conjunto maior ou RL (GRPO, o braço G original) poderiam fazer melhor, a um
custo ainda maior.

## 2026-11-08 — Estudo de custo (`scripts/cost_accounting.py` → `data/cost.json`, `docs/cost.md`)

**Método:**
- Tokens: o `usage` de cada mensagem dos transcritos dos subagentes, deduplicado por id.
- Preço de referência: tabela da API para Claude Opus 5.5 (US$ 4 / 20 por milhão de tokens de entrada / saída, US$ 0,20
  por leitura de cache). A escrita de cache foi calculada a 1,25× (TTL de 5 min) e a 2× (TTL de 1 h), porque o
  transcrito não registra o TTL. Nada foi cobrado de fato: as runs usaram assinatura do Claude Code.
- GPU dos agentes: sondas × 52 s + checagens × 10,5 min. As duas taxas foram calibradas na run naive, sozinha na GPU,
  com orçamento cheio: 40 sondas e 6 checagens em 97,8 min.
- Fica de fora de todos os braços: o dev48 de confirmação, o trabalho do supervisor (contado à parte) e a construção do
  atlas, que é feita uma vez só.

| braço | dev48 | GPU-h | tokens (contexto / saída) | US$ equivalente |
|---|---|---|---|---|
| a1 | 0,667 | 1,31 | 5,39 M / 47,5 k | 5,08–6,99 |
| a2 | 0,646 | 1,30 | 6,86 M / 44,8 k | 5,95–8,26 |
| r2 / r3 (com atlas) | 0,458 / 0,396 | 1,34 / 1,38 | 10,4 M / 73 k; 10,7 M / 57 k | 9,07–12,52 / 3,88–4,25 |
| a0 / r1 (v1, só sondas) | 0,375 / 0,042 | 0,69 / 0,68 | 5,9 M / 58 k; 5,9 M / 64 k | 2,94–3,32 / 3,10–3,50 |
| naive (E3) | 0,271 | 1,63 | — | — |
| LoRA-SFT (E4) | 0,292 | 17,1 | — | — |
| ajuste da decodificação (5 configurações no dev48) | 0,479–0,521 | 4,41 | — | — |

- **Razão de GPU, E4 contra a1:** 13× pela GPU ativa; 8× contando a janela inteira do avaliador residente (2,1 h). A
  entrada anterior falava em ≈ 8×; a medida principal passa a ser a GPU ativa, e o 8× fica como limite conservador.
- **Ponto de equilíbrio em dólar:** US$ 5–7 de tokens contra ~16 GPU-h a mais na 1660S. O agente sai mais barato
  sempre que a GPU-hora custar mais de ~US$ 0,32–0,44. Em GPU própria, o custo marginal é energia: ~0,26 kWh contra
  ~3,4 kWh, supondo ~200 W na tomada.
- **Custo de servir** (tokens gerados por problema, teto 12k):

  | | dev48 | MATH-148 |
  |---|---|---|
  | base | 11.714 | 11.788 |
  | LoRA | 9.915 (−15%) | — |
  | bf4096 | 8.073 (−31%) | 9.245 (−22%) |
  | **a1** | **4.729 (−60%)** | **8.399 (−29%)** |
  | combo | 3.772 (−68%) | 7.089 (−40%) |

  É uma economia recorrente, a cada consulta.
- **Astra** (logs públicos, uso por turno): a run publicada p50c teve 158 turns, 20,3 h, 86,8 M tokens de entrada
  (93% em cache) e 2,91 M de saída. O a1 usou 16× menos entrada e 60× menos saída, em cerca de 1/10 do tempo. Não é uma
  comparação controlada: os harnesses diferem, e os passes do Astra eram gerações completas com teto de 32k.
- **Totais:**
  - todas as runs de agente, incluindo os 2 lançamentos mortos pelo OOM: 6,7 GPU-h e US$ 31–40;
  - a sessão do supervisor no projeto inteiro (PoCs, harness, briefs, auditorias, E1–E4, documentação e recuperação
    do disco): 78,2 M de contexto e 0,58 M de saída, US$ 45–56 equivalentes. É comum a todos os braços, inclusive o
    fine-tuning.

## 2026-11-08 — Pre-paper reescrito como manuscrito

- `docs/paper-draft.md` deixou de ser só um esboço e virou um rascunho completo: resumo, introdução com contribuições,
  trabalhos relacionados (citações marcadas [verify]), setup, seções de resultados, custo, especificidade,
  limitações, conclusão, plano de figuras ligado ao notebook e plano até a submissão. Será revisado pela orientadora.
- **Correção na entrada "Estudo de custo":** a tabela listava r2 com 0,458 no dev48. O valor correto é **0,354**
  (`gen_dev48.json`, `m2a_r2`), como em F7. Foi erro de transcrição; nada mais muda. O mesmo erro em `docs/cost.md`
  foi corrigido.

## 2026-11-08 (17h) — Teste trancado ampliado (t200). PRÉ-REGISTRO

Pedido do Luiz: aumentar o n do teste trancado antes de tudo. O t12k tem só 32 problemas (±8–9 pp), e é nele que o
budget forcing aparece à frente do a1 (0,625 contra 0,562).

**Conjunto:** GSM8K **test[200:400]**, 200 problemas, teto 12288, amostragem THINK35, `-np 16`, mesma semente por
problema em todos os braços (`scripts/queue_t200.sh`, tag `t200`). Escolhi [200:400], e não [32:232], porque
test[0:200] já foi usado (sem raciocínio) no painel de efeitos colaterais. Nenhum método foi escolhido nesse painel,
mas [200:400] nunca foi tocado por nenhum braço, em nenhum modo.

**Braços**, todos escolhidos antes, sem nenhuma escolha nova. Rodam nesta ordem de prioridade, para que um corte de
tempo preserve as comparações principais:
1. a1 (`m2a_a1_062959_102`, a mesma variante do t12k e do MATH);
2. bf4096 (o melhor budget forcing no dev48);
3. a1 + bf4096;
4. astra;
5. base;
6. lbx_w10_t1 (o melhor logit bias no dev48);
7. ft_sft_ep2 (a época do LoRA escolhida na checagem de geração).

**Análise primária:** acerto por braço com EP binomial; comparações pareadas (resgates e perdas, ganho ± EP pareado)
para a1 contra base, astra, bf4096 e LoRA, e para a1 + bf4096 contra a1 e contra bf4096. **Secundária:** t12k + t200
juntos (232 problemas). Nada é reescolhido depois de ver o t200.

**Previsões (antes de rodar):**
- base ≈ 0,13; astra ≈ 0,42; a1 ≈ 0,60; bf4096 ≈ 0,55; a1 + bf4096 ≈ 0,65; lbx ≈ 0,47; LoRA ≈ 0,28.
- a1 > base e a1 > astra, com folga (pareado, > 3 EP).
- a1 contra bf4096: a1 à frente por ~5 pp, mas sem certeza. Prevejo que a diferença fique dentro de 2 EP. **Se o
  bf4096 vencer por mais de 2 EP, o texto passa a dizer que, no GSM8K, o budget forcing é tão bom quanto a edição ou
  melhor**, e a vantagem do a1 fica restrita a problemas longos (MATH) e ao custo de servir.
- a1 + bf4096 ≥ max(a1, bf4096).

**Custo previsto:** ~70 s por problema no base e ~40 s no a1. São ~14,5 h para os 5 primeiros braços e ~21 h para os 7.

## 2026-11-08 (18–19h) — Triagem do segundo defeito: saída estruturada e idioma

Objetivo: achar um defeito do Qwen3.5-0.8B que o modelo base erre em 30–70% dos casos, que seja verificável
automaticamente, que não tenha uma palavra-chave óbvia para mirar e que seja uma falha de elicitação (o modelo sabe
fazer em outras condições). Benchmarks em `src/wsp/structbench.py`, avaliação com `scripts/struct_eval.py`, dev de
120 itens por família, sem raciocínio, amostragem TOOLS (sem presence penalty). O t200 foi pausado e retomado por
item para cada triagem. Custou ~15 min de GPU no total.

| família | base | a1 | o que falha |
|---|---|---|---|
| JSON por schema (extração, tipos, enum, aninhado, null) | exato 0,833; JSON válido 1,00; schema 0,95 | 0,833 | quase só valores (ex.: "No prior experience" → null em vez de 0); estrutura e tipos ~sempre certos |
| tool calling difícil (6 ferramentas, opcionais, listas, 2 chamadas) | exato 0,675; formato 1,00 | 0,692 | 21/39 falhas são parâmetros opcionais preenchidos sem pedido. Quase tudo em 2 parâmetros com enum: `currency=USD` (11/13; é o default documentado, então é artefato do verificador) e `repeat=daily` (10/12; erro real). O resto são valores ("Pay rent at 20:00" no campo de texto, músicas inventadas) |
| restrições de formato (IFEval sintético) | 0,683 | — | o benchmark tinha combinações conflitantes (minúsculas + frase final com maiúscula); "sem vírgulas" é lexical |
| idioma (pedido em pt-BR), sem raciocínio | 0,95 em português | — | 4% cortados no teto de 768 |
| idioma, com raciocínio (teto 3072, n=48) | resposta em pt ~0,8 | — | pensa em inglês, mas responde em português; as falhas são os loops (cortes) |

**Conclusão:** "JSON malformado" e "formato de tool call quebrado" não são defeitos deste modelo sem raciocínio. O
JSON sai sempre válido e o formato de chamada sempre certo. O que falha são valores e parâmetros opcionais. O único
defeito grande que vimos continua sendo o loop com raciocínio ligado. Próximo candidato: **IFEval** (Zhou et al.,
2023), o benchmark padrão de seguir instruções de formato, com o verificador oficial (vendorizado em
`src/wsp/ifeval/`) e um split fixo: dev = 200 prompts, test = 341, trancado.

## 2026-11-08 (20h) — Segundo defeito: IFEval sem raciocínio. Triagem e PRÉ-REGISTRO

**Triagem** (`scripts/ifeval_eval.py`, verificador oficial em `src/wsp/ifeval/`; split fixo dos 541 prompts por
embaralhamento com semente 0: dev = 200, test = 341, trancado). Base, dev200, sem raciocínio, teto 2048, amostragem
TOOLS:
- prompt-level strict 0,530 (loose 0,570); instruction-level strict 0,697;
- 14% das respostas batem no teto: são repetição degenerada de uma frase ou parágrafo, um loop sem raciocínio;
- das 94 falhas, 28 são esses cortes e 66 são descumprimento puro, espalhado por muitos tipos. Os mais comuns:
  sem vírgulas (11/21), repetir o pedido (8/17), palavra obrigatória (6/15), palavra proibida (6/15) e seções
  destacadas (5/19).

Por que este defeito:
- está na faixa pedida (~50% de falha);
- é verificável com o código oficial;
- não tem palavra-chave única para mirar;
- é um benchmark padrão, reconhecido pelos revisores;
- é sobretudo elicitação: o modelo sabe escrever sem vírgulas e em minúsculas, mas não o faz quando pedido.

**Protocolo** (o mesmo da run a1, só muda o defeito):
- **Agente:** Claude Opus 5.5, sem atlas, briefing `data/m2a_if/BRIEF_if.md`, mesmo espaço de edições.
- **Orçamento:** 40 sondas e 6 checagens por geração.
- **Sondas** (`src/wsp/ifprobes.py`, avaliador `scripts/m2a_scorer_if.py`): pares casados com resposta escrita à mão,
  uma versão que cumpre e outra que não cumpre a instrução (8 tipos × 8 tópicos de busca; 3 tópicos de holdout
  lacrado), mais o controle sem a instrução no prompt e WikiText.
- **Checagem por geração:** IFEval dev[0:48].
- **Desvio declarado em relação à v2:** a checagem usa 48 prompts, não 16, porque aqui as respostas são curtas (~3–5
  min por checagem).

**Avaliação, decidida antes de rodar:**
1. A escolha nº 1 do agente (`final.json`) é confirmada no IFEval dev[48:200] (152 prompts), pareada contra o base.
   No mesmo conjunto rodam também os controles: a1 (a edição dos loops) e o base com um system prompt pedindo para
   seguir as instruções à risca (braço P).
2. Teste trancado (IFEval test, 341 prompts): uma escolha por braço, rodada uma vez só, depois da confirmação.
3. Efeitos colaterais: GSM8K sem raciocínio (n=200, como no painel do paper 1) e o tool calling sintético.
4. Fine-tuning (análogo ao E4): SFT LoRA nas próprias respostas do base que passam no verificador (amostragem de
   rejeição em prompts sintéticos, não do IFEval). Fica para depois, se der tempo.

**Previsões (antes de rodar o agente):**
- A tarefa é bem mais difusa que os loops, que tinham uma alavanca lexical ("Wait") e um sinal de saída
  (`</think>`). Prevejo um ganho modesto: escolha do agente no dev152 entre 0,58 e 0,62 (+5 a +9 pp sobre ~0,53).
  Dou ~30% de chance para mais de +10 pp e ~25% para nenhum ganho significativo (< 2 EP pareados).
- Se houver ganho, a fonte mais provável é a parte de repetição degenerada (14% de cortes), não o descumprimento puro.
- O a1 (edição dos loops) não deve mudar o IFEval sem raciocínio: ele age em "Wait" e `</think>`, que não aparecem
  aqui.
- O system prompt muda pouco (no paper 1, prompts não ajudaram contra loops), mas aqui pode ajudar no descumprimento.
- As sondas (if_margin) vão sofrer Goodhart: a correlação com o prompt_strict da checagem será fraca.

## 2026-11-08 (18h40–19h25) — Sanidade do avaliador IF e run b1

**Sanidade** (run `sanity`):
- Sondas: uma edição aleatória dá ~0 em tudo. A edição a1 (loops) quase não mexe (if_margin −0,003, ctrl −0,001,
  neutral −0,002).
- O forward do Qwen3.5 neste stack faz só ~600 tok/s (fallback do `causal_conv1d`), mesmo em lote. Para manter
  ~80 s por sonda, reduzi o controle para 3 tópicos e o WikiText para 4 blocos (mudança feita antes do b1).
- Checagem por geração: base 0,500 (cortes 0,19).
- **Surpresa: a a1 fez 0,625 (8 resgates / 2 perdas; cortes 0,06).** A previsão dizia que a a1 não mudaria o IFEval
  sem raciocínio. É uma amostra de 48 prompts; a confirmação no dev152 (que já inclui a a1 como controle) decide.

**Run b1** (Claude Opus 5.5, sem atlas, briefing IF; 45 min de relógio, 23 de 40 sondas e 4 de 6 checagens):
- **Escolha nº 1** (`184512_099`): uma edição rank-1 em L23 `mlp.down_proj`. Direção `<|im_end|>` vista pela saída,
  `(1+g_final)⊙E[eos]`, nos 64 neurônios que mais empurram o fim da resposta, α = −1,5 (amplifica 2,5×). Na
  checagem: 0,625 contra 0,500 (8/2), cortes 0,125, neutral −0,0099.
- Uma variante quase idêntica (+ L22) voltou a 0,50 (6/6). **O próprio agente julga o ganho fraco**, em nível de
  ruído.
- **Mecanismo proposto:** reforçar o fim de resposta para quebrar a repetição degenerada. É a mesma família de
  alavanca do `</think>` da a1, agora no `<|im_end|>`.
- **O que falhou:** direções de "instrução" estimadas dos pesos, direções singulares das camadas de atenção,
  neurônios de entropia, cabeças de supressão de cópia e EOS em todos os neurônios (neutral −1 a −3). O
  EOS-menos-quebra-de-linha fez 0,44: o modelo termina antes de escrever o P.S. ou a frase final.
- **Sondas × geração:** as sondas não previram nada das edições de parada; só o `postscript` negativo sinalizou as
  edições que cortam o P.S. Isso confirma a previsão de Goodhart/sondas fracas.
- **Auditoria:** nenhuma chamada de ferramenta tocou caminhos proibidos (`docs/`, `scripts/`, `data/m2a/`,
  `data/ifeval/`, `src/wsp/ifeval/`, `sealed/`, refs).
- **Desvio:** o agente parou cedo "por tempo de relógio" (o prompt dizia ~2 h; ele usou 45 min). O orçamento não foi
  esgotado.

## 2026-11-08 (19h45) — Confirmação IFEval dev152: nenhum braço supera o base

IFEval dev[48:200], 152 prompts, sem raciocínio, teto 2048, pareado contra o base (mesmas sementes):

| braço | prompt strict | inst strict | cortes | tokens | vs base (ganho ± EP; resgates/perdas) |
|---|---|---|---|---|---|
| base | **0,553** | 0,686 | 0,099 | 483 | — |
| b1 (EOS ×2,5 em 64 neurônios de L23) | 0,520 | 0,657 | 0,066 | 383 | −3,3 ± 4,1 (17/22) |
| a1 (edição dos loops) | 0,513 | 0,669 | 0,118 | 478 | −3,9 ± 4,3 (18/24) |
| P (system prompt) | 0,487 | 0,653 | 0,184 | 571 | −6,6 ± 4,2 (16/26) |

**Leitura:**
- A previsão principal **falhou**. Eu previa +5 a +9 pp para o agente; ele ficou em −3,3 ± 4,1. Aconteceu o cenário
  "nenhum ganho" (25% na previsão).
- O ganho do b1 e da a1 na checagem (0,625 contra 0,500) era ruído. A previsão de que a a1 não muda o IFEval
  acabou certa.
- O b1 reduziu os cortes (0,066 contra 0,099) e encurtou as respostas, mas perdeu em cumprimento.
- O system prompt **piorou**: mais cortes (0,18). No paper 1, prompts também pioraram os loops.

**Por que a busca não teve sinal: ruído de amostragem por prompt.**
- Duas amostras do **próprio base** com sementes diferentes discordam em **35 de 152 prompts (23%)**. Com a mesma
  semente, discordam em 3 de 48.
- Qualquer edição muda a trajetória da amostragem, então cada checagem de 48 prompts vira ~10–16 prompts por puro
  acaso. Foi o que o b1 relatou.
- O EP pareado de uma checagem de 48 prompts é ~8–9 pp, maior que qualquer efeito plausível de 4 edições num defeito
  difuso.
- No defeito dos loops, os efeitos eram de +40 a +50 pp, e uma checagem de 16 problemas bastava. **O alcance do
  protocolo depende da razão sinal/ruído da checagem**, não só da capacidade do agente.

**Próximos passos (para decidir com o Luiz; nada disto roda antes do t200):**
1. **b2, réplica com checagem mais forte** (desvio declarado e motivado pela medida acima): a checagem passa de 48
   prompts do IFEval para 128 prompts sintéticos (`wsp.ifsynth`, mesmas famílias de instrução, sementes fixas). Isso
   dá EP pareado ~4–5 pp. Mantém as 40 sondas e 6 checagens. Pergunta: com sinal melhor, o agente acha uma edição
   que se confirma no dev152? Se não achar, o limite é o defeito (difuso demais para ≤ 4 edições rank-1), não o
   ruído.
2. **Baseline de fine-tuning** (`scripts/queue_if_ft.sh`, ~3 GPU-h): diz se o defeito é corrigível nesta escala. Se
   o SFT ganhar e as edições não, é um limite honesto do método, a reportar no paper.
3. O teste trancado do IFEval só abre se algum braço passar na confirmação. Hoje nenhum passou, então ele continua
   fechado.

## 2026-11-08 (21h40) — t200: primeiro braço; pré-registro do b2 e do baseline FT do IFEval

**t200, braço a1 completo:** 0,615 (cortes 0,17; loop no teto 0,165; fechou o raciocínio em 0,865). A previsão
registrada era ≈ 0,60. Os demais braços seguem (bf4096 começou às 21h30; ~1h45 por braço).

**Ordem da GPU para a madrugada (decidida antes de ver qualquer resultado novo):**
1. t200, os 5 braços principais (a1, bf4096, combo, astra, base), até ~04h40.
2. Baseline FT do IFEval (`scripts/queue_if_ft.sh`, sem mudanças): dados por rejeição em 600 prompts `ifsynth`
   (sementes 10000+, K=2) → LoRA-SFT com a receita do E4 (3 épocas) → escolha da época no IFEval dev[0:48] →
   confirmação no dev152.
3. b2 (precisa de mim acordado para lançar o agente).
4. Cauda do t200 (lbx_w10_t1, ft_sft_ep2), a menor prioridade: é a primeira a sair se faltar tempo.

**b2, desenho (desvio declarado em relação ao b1, motivado pelo ruído medido no dev152):**
- Checagem por geração: 128 prompts sintéticos (`gen_set: synth128`, `wsp.ifsynth` com sementes 50000..50127,
  disjuntas das do FT). Não usa nenhum prompt do IFEval. EP pareado esperado ~4 pp (contra ~8–9 com 48 prompts).
- O brief passa a informar o ruído medido (23% de discordância base contra base; diferenças < ~8 pp em 128 prompts
  são ruído) e diz que não há prêmio por terminar cedo (até ~4 h). `ifsynth.py` e `data/ft_if/` entram na lista
  proibida. O resto fica igual: 40 sondas, 6 checagens, mesmas sondas, mesmo espaço de edição.
- Como a checagem não toca o IFEval, **a confirmação usa o dev200 inteiro** (tag `dev200b2`, braços base e
  `m2aif_b2`, base reamostrado com as mesmas sementes). O recorte dev152 também é reportado, para comparar com o b1.
- **Critério para "passar":** ganho pareado no dev200 > 2 EP. Só nesse caso eu pergunto ao Luiz se abro o teste
  trancado do IFEval.

**Previsões:**
- b2 acha um candidato com ganho ≥ 8 pp na própria checagem: 35%. Passa na confirmação: 20%.
- Se o b2 falhar com a checagem forte, a leitura é: o defeito é difuso demais para ≤ 4 edições rank-1, não o ruído.
- FT: a melhor época ganha de +5 a +15 pp no dev152 (60%); ganho ≤ 2 pp (25%).

## 2026-11-09 (11h15) — t200 completo (5 braços principais) e baseline FT do IFEval

**t200** = GSM8K test[200:400], teste trancado aberto com autorização, com raciocínio, teto 12288. Comparações
pareadas (ganho ± EP; problemas que só o primeiro acerta / só o segundo):

| braço | acerto | previsão | cortes | tokens (média) | tempo |
|---|---|---|---|---|---|
| base | 0,155 | ≈ 0,13 | 0,845 | 11142 | 4h00 |
| astra | 0,535 | ≈ 0,42 | 0,425 | 8877 | 3h10 |
| bf4096 (proibir "Wait", decodificação) | 0,610 | ≈ 0,55 | 0,340 | 7012 | 2h57 |
| **a1 (edição do agente)** | **0,615** | ≈ 0,60 | 0,170 | 5049 | 1h47 |
| **a1 + bf4096** | **0,695** | ≈ 0,65 | 0,095 | 4081 | 1h42 |

| comparação | t200 | combinado t12k + t200 (232) |
|---|---|---|
| a1 − base | +46,0 ± 3,7 (95/3) | +45,7 ± 3,5 |
| a1 − astra | **+8,0 ± 3,4** (32/16) | **+9,1 ± 3,1** |
| a1 − bf4096 | +0,5 ± 3,4 (24/23) | −0,4 ± 3,1 |
| a1+bf − a1 | +8,0 ± 3,0 (26/10) | +7,8 ± 2,8 |
| a1+bf − bf4096 | +8,5 ± 3,6 (35/18) | +7,3 ± 3,3 |
| astra − base | +38,0 ± 3,5 | — |

**Leitura:**
- Com n = 200, todas as afirmações do paper se mantêm, e as diferenças agora são de 2 a 3 EP, não mais sugestões.
- A edição do agente vence o ASTRA (o steering da literatura) por 8–9 pp e empata com a intervenção na decodificação
  (bf4096). Pelo critério pré-registrado ("se bf4096 vencer por > 2 EP o texto muda"), **o texto não muda**.
- **Diferença em custo:** a a1 empata em acerto com metade dos cortes e ~30% menos tokens. Proibir o "Wait" impede
  o recomeço, mas não faz o modelo concluir.
- **Os dois mecanismos se somam:** a1 + bf4096 fica 8 pp acima de cada um, com o menor custo de todos.
- Todas as previsões acertaram a direção. Subestimei o astra e o bf4096.
- A cauda (lbx_w10_t1, ft_sft_ep2) não rodou. Fica para depois do b2, se a máquina continuar disponível.

**Baseline FT do IFEval** (`queue_if_ft.sh`):
- **Dados:** 600 prompts `ifsynth` × 2 amostras. 58% das amostras passam, e ficam 414 prompts (uma resposta cada).
  Cortes de 4,8%.
- **Treino:** LoRA-SFT com a receita do E4, 3 épocas.
- **Escolha da época** no IFEval dev[0:48]:

  | braço | prompt strict | inst strict | cortes | tokens |
  |---|---|---|---|---|
  | base | 0,458 | 0,648 | 0,250 | 673 |
  | ep1 | 0,500 | 0,592 | 0,271 | 717 |
  | **ep2** | **0,583** | 0,704 | 0,146 | 478 |
  | ep3 | 0,562 | 0,690 | 0,146 | 474 |

  ep2 escolhida (+12,5 ± 6,4 nos 48).
- **Confirmação dev152:** ep2 tem 0,480 contra 0,553 do base, **−7,2 ± 4,3** (16 resgates / 27 perdas). Os cortes caem
  um pouco: 0,164 aqui, contra 0,25 → 0,146 na escolha.
- **A previsão falhou.** Eu previa +5 a +15 pp (60%). Aconteceu o cenário "≤ 2 pp" (25%), e com sinal negativo.

**O que isso muda na leitura do IFEval:**
- O mesmo padrão aparece em três seletores diferentes: a busca do agente (b1), o system prompt (P) e a escolha de
  época do FT. Todos parecem ganhar 8–12 pp em 48 prompts e perdem 3–7 pp no dev152.
- A **escolha em 48 prompts é dominada pelo ruído** (maldição do vencedor), qualquer que seja o método. O base mediu
  0,458 no ftsel e 0,500 no gen_ref da checagem, nos mesmos 48 prompts.
- O FT com dados autodestilados também não melhora o IFEval. Então **o defeito não é corrigível de forma barata
  nesta escala**, nem por gradiente. Isso conta a favor do método, não contra: o agente não "falhou" onde o FT acerta.
- O b2 (checagem com 128 prompts) começou às 11h15 e testa se, com sinal melhor, a edição acha algo.

## 2026-11-09 (13h45) — b2: a checagem maior não muda nada; confirmação negativa

**Run b2** (gen_set synth128; 36 de 40 sondas e 6 de 6 checagens; ~1h50 de relógio, porque as checagens levaram
5–8 min, não 15–20). **Auditoria:** nenhuma leitura proibida nas 117 chamadas de ferramenta.

| checagem (128 sintéticos; base 0,578, cortes 0,078) | prompt strict | cortes |
|---|---|---|
| 112126_253: neurônios de parada tardios ×3 | 0,563 | 0,023 |
| **120302_651: d_a ×5 em L0, L1, L2, L4 (linear_attn.out_proj)** | **0,625** | 0,063 |
| 122209_766: d_a, outra ponderação | 0,578 | 0,078 |
| 123756_003: variante | 0,594 | 0,078 |
| 115957_748: d_a removida (sinal oposto) | 0,594 | 0,047 |
| 125218_060: direção fim-vs-quebra-de-linha em L19/L23 | 0,555 | 0,078 |

- **Achado mecanístico (do agente, só com os pesos):** uma direção residual, d_a, é lida com o mesmo sinal por
  99,7% das linhas dos portões de decaimento (`in_proj_a`) das 288 cabeças DeltaNet. O valor singular principal
  concentra 0,215 da energia, contra 0,001 numa direção aleatória.
  - É um canal global de controle do esquecimento. Editar d_a muda o modelo: neutral até +0,026, ou seja, o texto
    genérico fica *mais* provável.
  - Os dois sinais dão resultados parecidos no IFEval sintético, então o efeito não tem direção. O próprio agente
    estimou o ganho fora da amostra em 0 a +2 pp.
  - Vale investigar à parte (memória/esquecimento das DeltaNet), fora do segundo defeito.
- **Confirmação pré-registrada** (dev200, base reamostrado, mesmas sementes):
  - base 0,560 (inst 0,691; cortes 0,12); b2 = 120302_651 0,500 (inst 0,632; cortes 0,17).
  - **−6,0 ± 3,7** (22 resgates / 34 perdas). No recorte dev152: −7,2 ± 4,4.
  - Não passa no critério. O teste trancado do IFEval continua fechado.
- **Previsões:** "ganho ≥ 8 pp na própria checagem" não aconteceu (+4,7). "Passar na confirmação" (20%) também
  não aconteceu.
- **Ruído do base de novo:** a reamostra do base no dev[48:200], contra a amostra do dev152, deu +0,7 ± 4,0 (19/18
  discordâncias, 24%).

**Leitura conjunta do segundo defeito (5 braços escolhidos por 4 seletores):**

| braço | dev152 vs base |
|---|---|
| b1 (agente, checagem de 48) | −3,3 ± 4,1 |
| a1 (edição dos loops, sem seleção no IFEval) | −3,9 ± 4,3 |
| P (system prompt) | −6,6 ± 4,2 |
| FT ep2 (LoRA-SFT, época escolhida em 48) | −7,2 ± 4,3 |
| b2 (agente, checagem de 128) | −7,2 ± 4,4 (dev200: −6,0 ± 3,7) |

- Todos negativos. A checagem maior **não** resolveu, então a explicação "só ruído" fica fraca. A leitura que sobra:
  - neste modelo e nesta tarefa, perturbações pequenas e baratas (pesos, prompt ou gradiente com dados
    autodestilados) tiram o modelo de um ponto que já é quase o melhor da vizinhança;
  - a seleção em ~50–130 prompts não acha as raras que ajudam.
- O padrão de custo também se inverte entre os conjuntos. No sintético, o b2 cortava *menos* (0,063 contra 0,078);
  no IFEval, cortou *mais* (0,17 contra 0,12). A checagem sintética não reproduz bem a distribuição dos prompts
  reais do IFEval.
- **Para o paper:** o segundo defeito é um **resultado negativo bem controlado**. O método (e o FT) funcionam quando o
  defeito é concentrado e o sinal é grande (loops: +46 pp). Não funcionam quando o defeito é difuso e o efeito
  plausível fica abaixo do ruído de amostragem. Isso delimita o escopo da técnica.

## 2026-11-10 (09h40) — t200 sem raciocínio: PRÉ-REGISTRO (resposta à crítica "é só desligar o raciocínio")

**Motivo.** Reexecuções públicas do Experimento 8 do Sparks (exp8-loop-controls; Rethinking WSP; ver
`docs/literatura-sparks.md`) mostram que o modelo original **sem raciocínio** já faz 56–57% no GSM8K inteiro com teto
longo. Nosso único número sem raciocínio (0,485) é de test[0:200], com teto de 1.024 tokens. Ele não é pareado com o
t200 (test[200:400]).

**Conjunto e braços.** GSM8K test[200:400], os mesmos 200 problemas do t200, com as mesmas sementes por problema.
- Raciocínio desligado, amostragem NOTHINK do model card (temp 0,7, top_p 0,8, presence 1,5), teto de 8.192.
- Braços: base, a1 (`m2a_a1_062959_102`), astra.
- Tag `t200nt`, script `scripts/queue_t200_nothink.sh`.
- Em seguida, na mesma janela, retoma o braço pré-registrado `lbx_w10_t1` do t200, que é interrompível e retoma por
  item.

**Previsões, escritas antes:**
1. base sem raciocínio: entre 0,50 e 0,60. Central: 0,55.
2. a1 e astra sem raciocínio ≈ base (|Δ| < 5 pp): sem raciocínio não há loop longo para remover.
3. a1 com raciocínio (0,615) − base sem raciocínio: positivo, mas pequeno (+3 a +10 pp). a1 + bf4096 (0,695) − base
   sem raciocínio: ≥ +10 pp.

**Regra de texto, fixada antes:**
- Se o base sem raciocínio ficar ≥ 0,615, o paper passa a dizer explicitamente que a edição sozinha **não supera**
  desligar o raciocínio no GSM8K. A afirmação principal passa a ser a combinação edição + limite, se ela continuar
  acima.
- Se ficar < 0,615, o paper relata a diferença pareada, com erro-padrão, como mais uma alternativa da tabela principal.

A comparação entre modos é entre braços diferentes nos mesmos problemas: pareada por problema, mas não por semente.

## 2026-11-10 (10h50) — Runs f1 e f2: Claude Fable 5.2 como editor. PRÉ-REGISTRO

**Motivo.** O Fable 5.2 está disponível por tempo limitado na assinatura. Ele é um degrau acima do Opus 5.5 na escada
do editor (`docs/linhas-futuras.md`, Linha 1) e serve de réplica do protocolo v2 sem atlas. Também responde a uma
pergunta aberta pela literatura (`docs/literatura-sparks.md`): editores diferentes convergem para os mesmos neurônios
(a1, a2 e o tdotlab em L22n573) ou acham direções diferentes (ajaygrover: Opus quase ortogonal ao Astra)?

**Protocolo: idêntico ao do a2**, no harness original (`/mnt/dados/dev/deep_traces`, onde a1 e a2 rodaram).
- Mesmo `BRIEF_v2.md`, sem mudança nenhuma, e o mesmo prompt do a2 (versão final, com o limite de memória).
- Mesmo avaliador (`m2a_scorer.py`): 40 sondas e 6 checagens por geração.
- Sem atlas. No máximo 4 edições de posto 1, |α| ≤ 10.
- Duas runs, `f1` e `f2`, ao mesmo tempo, dividindo o avaliador, como a2 e r3.

**Desvios, todos fixados antes:**
1. O modelo do agente é o Claude Fable 5.2. a0–a2 e r1–r3 usaram o Opus 5.5.
2. Regra nova no prompt: sem acesso à web, e sem ler `/mnt/dados/dev/agentic-weight-editing`. As respostas agora são
   públicas (tdotlab e outros citam L22 col 573) e o repositório do paper não existia quando o a2 rodou.
3. A auditoria das chamadas de ferramenta inclui buscas na web e os dois repositórios.

**Confirmação.**
- A escolha nº 1 de cada `final.json` vai para o dev48 (GSM8K train, teto 12.288, mesmas sementes), pareada contra o
  base e contra a a1.
- O melhor dos dois no dev48 roda no t200 como braço novo. **O t200 já foi aberto**: esse braço é posterior aos
  resultados dos outros e será relatado assim. A escolha do braço usa só o dev48.

**Previsões, escritas antes:**
1. Pelo menos uma das duas runs chega a dev48 ≥ 0,60: 70%.
2. As duas chegam a dev48 ≥ 0,50: 50%.
3. Alguma run supera a a1 (0,667) no dev48 por ≥ 5 pp: 20%. Motivo: a a1 já remove a maior parte dos loops de
   "Wait", e o resto do loop não é uma palavra só (tdotlab, rodada 2).
4. Pelo menos uma run edita L22n573 ou um neurônio-freio de "Wait" em L17–L23: 65%.
5. Custo: o Fable usa menos sondas que o Opus para chegar à primeira checagem positiva. Palpite fraco, 55%.

**Leitura combinada, fixada antes:**
- Se as duas ficarem < 0,40: o resultado da a1/a2 depende mais do agente (ou da sorte) do que o paper sugere, e a
  seção de réplicas precisa dizer isso.
- Se convergirem para os mesmos neurônios: vira evidência de que a solução é uma propriedade do modelo-alvo, não do
  editor.

## 2026-11-10 (10h55) — Revisão independente (Fable 5.2) e fila de correções. PRÉ-REGISTRO

Uma revisão independente recalculou os números principais (todos conferem) e apontou quatro excessos de afirmação.
Conferi dois deles nos arquivos (`final.json` do a1; sinais de α em a1 e a2):
1. "a1 supera o Astra" só vale no teto de 12k: no t200, o Astra ainda sobe ~8 pp a cada 2k tokens ao bater no teto.
2. A a1 do paper (`062959_102`) é a escolha nº 2 do agente. A nº 1 (`074537_659`) nunca foi a teste trancado.
3. a1 e a2 editam L17n1915 e L22n573 com sinais opostos (α = −6 contra α = +10, com a mesma direção u).
4. O custo do bf4096 é inflado por respostas forçadas que continuam em loop depois do `</think>`.

**Fila 1 (GPU, hoje), braços fixados antes:**
- `t200nt` (já pré-registrado às 09h40), sem o corte das 13h50.
- **t200, braços novos `m2a_a1` (a nº 1 do agente a1) e `m2a_a2` (a nº 1 do agente a2).** Mesmo protocolo do t200.
  São braços posteriores à abertura do t200. Nenhuma escolha depende deles.
- Previsões:
  - `m2a_a1` (nº 1): 0,55–0,65, sem diferença significativa da `062959_102` (|Δ| < 2 EP). No dev48 foram 0,646 e
    0,667.
  - `m2a_a2`: 0,52–0,64. No dev48 foi 0,646.
  - As duas ficam acima do Astra (0,535) em ponto; não espero > 2 EP em cada uma.
- **Regra de texto:** o paper passa a relatar a nº 1 do agente como "a escolha do agente" e a `062959_102` como a
  variante escolhida pelo supervisor no dev48, com os dois números no t200.

**Em CPU, hoje:** sinal das ativações de L17n1915, L18n881, L22n573 e L16n998 em textos em loop, e a mudança líquida
no logit de "Wait" sob a1 e a2. Previsão: os neurônios disparam em média negativos nas posições antes de "Wait"
(hipótese do agente a1). Se for verdade, a a2 (inversão) deveria *aumentar* "Wait" por esses neurônios, e o ganho
dela viria de outra parte da edição (L16 ou a atenção da L22). Não sei qual das duas leituras vale: 50/50.

**Depois, nesta ordem:** runs f1/f2 do Fable (pré-registro das 10h50); limite de raciocínio com teto para a resposta,
varrido no dev48; braços `lbx_w10_t1` e `ft_sft_ep2` do t200; Astra contra a1 com teto longo.

## 2026-11-10 (11h15) — Sinal das ativações: o freio de "Wait" da a1 se confirma; a a2 age por outro caminho

`scripts/act_sign_check.py` → `data/act_sign_check.json`. 12 textos em loop do base (dev48, cortados), primeiros 1.500
tokens do raciocínio, 17.988 posições, 52 delas imediatamente antes de um "Wait". Rodou em GPU fp16: os kernels da
DeltaNet não rodam em CPU.

| neurônio | ativação média antes de "Wait" | fração negativa antes de "Wait" | uᵀw | escrita em u, base | depois da a1 | depois da a2 |
|---|---|---|---|---|---|---|
| L22n573 | −3,73 | 100% | +0,145 | −0,54 | −3,78 | +4,91 |
| L17n1915 | −0,23 | 83% | +0,133 | −0,03 | −0,22 | +0,28 |
| L18n881 | −0,02 | 44% | +0,136 | −0,003 | −0,02 | — |
| L16n998 | −2,60 | 100% | −0,142 | +0,37 | — | −3,33 |

Δ log p("Wait") contra o base, com o texto fixo (teacher-forced):

| variante | antes de "Wait" | todas as posições | Δ log p(`</think>`), todas |
|---|---|---|---|
| a1 (062959_102) | −16,5 | −0,89 | +1,45 |
| a1, só os 3 neurônios | −16,6 | −1,02 | −0,05 |
| a1, só o MLP da L23 | +0,07 | +0,12 | +1,50 |
| a1 nº 1 (074537_659) | −24,2 | −1,44 | +1,99 |
| a2 | −25,9 | −3,28 | +0,78 |
| a2, só os neurônios | −26,0 | −3,33 | +0,19 |
| a2, só a atenção da L22 | +0,05 | +0,06 | +0,76 |

**Leitura.**
- **A história do freio vale para a a1.** L22n573 dispara negativo em 100% das posições antes de "Wait" e escreve
  contra "Wait"; a a1 amplia essa escrita em 7×. As duas partes da a1 são separáveis: os neurônios tiram "Wait", o
  MLP da L23 sobe `</think>`. L18n881 quase não participa.
- **A a2 também derruba "Wait" (−26 nats), mas não pelo mesmo caminho direto.** Com as ativações do base, a inversão
  de L22n573 empurraria "Wait" para cima (+4,9), e a de L16n998, que no base *promove* "Wait" (+0,37), empurra para
  baixo (−3,3). A soma direta não explica −26 nats. O efeito precisa passar por camadas posteriores: as edições em
  L16 e L17 mudam o que L22n573 recebe. **Falta medir as ativações dentro do modelo editado.**
- Previsão (50/50 sobre qual leitura valia): a hipótese do agente a1 se confirma; a a2 fica em aberto.
- **Para o paper:** "os dois agentes acharam os mesmos neurônios de 'Wait' e a mesma alavanca de `</think>`; a a1
  amplia o freio, a a2 inverte neurônios a montante e chega ao mesmo efeito por um caminho indireto ainda não
  mapeado". A Figura 2 vale para a a1.

### 10/11, 15h30 — Resultado da fila 1 (t200 sem raciocínio; nº 1 do agente a1; a2)

A fila terminou às 14h56. Mesmos 200 problemas trancados (test[200:400]); p é o teste do sinal exato sobre os pares
discordantes.

**Sem raciocínio, teto 8.192:**

| braço | acerto | cortadas | tokens (média) | contra o original sem raciocínio |
|---|---|---|---|---|
| original | 58,5% | 5,0% | 1.263 | — |
| a1 (062959_102) | 56,0% | 2,0% | 925 | +24 / −29 |
| Astra | 57,5% | 5,5% | 1.384 | +22 / −24 |

**Com raciocínio, teto 12.288 (braços novos):**

| braço | acerto | cortadas | tokens (média) | contra o original |
|---|---|---|---|---|
| a1 nº 1 do agente (074537_659) | 62,0% | 5,5% | 2.269 | +97 / −4 |
| a2 nº 1 (015739_897) | 67,5% | 15,5% | 4.387 | +107 / −3 |

**Comparações pareadas contra o original sem raciocínio (58,5%):**

| braço (com raciocínio) | acerto | pares | p |
|---|---|---|---|
| a1 do paper | 61,5% | +28 / −22 | 0,48 |
| limite 4.096 | 61,0% | +29 / −24 | 0,58 |
| a1 + limite | 69,5% | +37 / −15 | 0,003 |
| a2 | 67,5% | +35 / −17 | 0,018 |
| Astra | 53,5% | +25 / −35 | 0,25 |

Outras: nº 1 do agente contra a1 do paper, +29 / −28 (p = 1,0). a2 contra a1 do paper, +29 / −17 (p = 0,10). a2
contra Astra, +42 / −14 (p < 0,001). a2 contra a1 + limite, +23 / −27 (p = 0,67).

**Previsões.**
- Original sem raciocínio em 0,50–0,60: **acertou** (0,585). a1 e Astra ≈ original sem raciocínio: **acertou**.
- nº 1 do agente em 0,55–0,65 e a menos de 2 EP da a1 do paper: **acertou** (0,62; diferença de 0,5 ponto).
- a2 em 0,52–0,64: **errou para baixo** (0,675). A a2 é a melhor edição isolada no t200.

**Regras de texto aplicadas.**
- O original sem raciocínio ficou abaixo de 0,615, mas a diferença para a a1 (3 pontos, p = 0,48) não é
  significativa. O paper diz: **a edição a1 sozinha empata com desligar o raciocínio**; só a1 + limite e a a2 ficam
  acima (11 e 9 pontos). Com o raciocínio desligado, nenhuma edição ajuda nem atrapalha.
- A nº 1 do agente a1 é indistinguível da a1 do paper e gasta menos tokens (2.269 contra ~2.900). O paper passa a
  reportá-la como "a escolha do agente"; a escolha do supervisor no dev48 não mudou o resultado.
- A a2 entra na tabela principal. Ressalvas: 15,5% cortadas no teto (ainda pode subir com teto maior, como o Astra),
  custo de tokens 1,5× o da a1, e p = 0,10 contra a a1. Sem correção para múltiplas comparações.

### 10/11, 15h50 — Runs f1 e f2 lançadas

- Avaliador (`wsp-m2a`) no harness original; f1 e f2 lançadas juntas às 15h48, modelo Claude Fable 5.2.
- Prompt: o do a2, com o id trocado e as regras do pré-registro (sem web, sem ler o repositório do paper).
- **Desvios não previstos no pré-registro:**
  1. Uma regra a mais no prompt: "não lance subagentes".
  2. `sealed/holdout_cases.json` tinha sido movido para este repositório na migração de 08/11. Copiei de volta para
     o harness original para o avaliador subir. O arquivo é o mesmo que o a2 usou; os agentes continuam proibidos de
     ler `sealed/`.
- O acompanhamento da a2 (ativações dentro do modelo editado) fica para depois das runs: a GPU está com o avaliador.

### 10/11, 16h40 — Paper reescrito depois da revisão (en e pt-BR)

Mudanças aplicadas em `docs/paper-draft.md` e `docs/paper-draft.pt-BR.md`; PDFs regerados (20 páginas).
- Enquadramento de eficiência: tabela de acerto por orçamento de tokens no t200; "vence o Astra" só no teto de 12k.
- Raciocínio desligado (58,5%) na tabela principal; a1 sozinha empata com ele.
- A a1 do paper é a nº 2 do agente (escolha nossa); a nº 1 e a a2 entram como braços tardios (†).
- Mesmos neurônios, sinais opostos; freio da a1 conferido nas ativações; caminho da a2 não mapeado.
- Limite de raciocínio: sem alegação de custo (cortado em 6.144, faz 60,5% com 4.910 tokens).
- O t12k virou "segundo conjunto de prática". Saiu a soma dos "232 problemas trancados".
- Entraram: edição por diferença de médias (31%), acerto entre as respostas terminadas, lista de 7 previsões que
  falharam, valores de p sem correção e o que sobra com Holm, ressalvas na tabela de custo, correção do MATH no
  texto, réplicas que não convergem (MatinKhajavi, ajaygrover), seção de uso de IA.
- **Pendente:** figuras 4–6 e 10 com os braços novos; README, `experiments.md` e `research-notes`.
- **Para o autor conferir:** a redação da seção "Uso de ferramentas de IA".

## 2026-11-10 (16h55) — Fila 2: limite de raciocínio com teto para a resposta. PRÉ-REGISTRO

**Motivo.** No `bf4096`, depois do `</think>` forçado a resposta pode usar o resto do teto (até 8.192 tokens), e 34%
das respostas seguem em loop até o fim. Isso infla o custo do limite. A versão nova, `bf<N>a<M>`, dá no máximo M tokens
à resposta depois do corte (`src/wsp/llama_gen.py`, `src/wsp/evalrun.py`).

**Fila (`scripts/queue_fix2.sh`), roda depois da confirmação das escolhas do Fable:**
1. dev48, modelo original: `bf1024a1024`, `bf2048a1024`, `bf4096a1024`.
2. Regra de escolha, fixada antes: maior acerto no dev48; em empate, menos tokens.
3. O vencedor no t200, sozinho e com a a1 (`m2a_a1_062959_102+<vencedor>`). Braços tardios.
4. Cauda pré-registrada do t200: `lbx_w10_t1` e `ft_sft_ep2`.

**Previsões.**
- No dev48, `bf4096a1024` faz 0,44–0,52 (o `bf4096` fez 0,479) com bem menos tokens (< 5.500 de média, contra 8.073).
- `bf1024a1024` fica abaixo de `bf4096a1024` no dev48: 65%. Motivo: com 1.024 tokens de raciocínio muitos problemas
  ainda não chegaram à resposta.
- No t200, o vencedor sozinho faz 0,56–0,64 e fica a menos de 2 EP da a1 (0,615) e do raciocínio desligado (0,585).
- a1 + vencedor faz 0,65–0,72 e continua acima do raciocínio desligado (p < 0,05).
- `lbx_w10_t1` no t200: 0,40–0,52. `ft_sft_ep2`: 0,22–0,34.

**Regra de texto.** Se o vencedor sozinho ficar ≥ 2 EP acima da a1 no t200, o paper diz que o limite bem implementado
supera a edição isolada. Se a1 + vencedor não ficar acima do raciocínio desligado, a frase "só a combinação fica
claramente acima" sai.

### 10/11, 18h40 — Runs f1 e f2 (Fable 5.2) terminadas; confirmação lançada

| run | nº 1 | edição | checagem por geração (16) | sondas / checagens usadas |
|---|---|---|---|---|
| f1 | `165020_381` | L23, todas as colunas, direção de "Wait": `mlp.down_proj` α = 1 e `self_attn.o_proj` α = 2 (2 edições) | 9/16, cortes 0,375, 8 resgates / 0 perdas | 22 / 4 |
| f2 | `162904_881` | saídas de atenção de L20–L23, todas as colunas, α = 4, direção "Wait" − 0,04·`</think>` (4 edições) | 10/16, cortes 0,25, 9 resgates / 0 perdas | 21 / 4 |

Base nas mesmas 16: 1/16, cortes 1,0.

**O que os agentes relatam.**
- Os dois acharam os neurônios de "wait" (L17n1915, L18n881, L22n573; a f1 também L20n405) e **não** os usaram na
  escolha final. A f1 testou invertê-los (α = 6): 0/16, com "WaitWaitWait…", o mesmo fracasso inicial da a1 e da a2.
  Ela não tentou reforçá-los. A f2 concluiu que eles codificam o verbo e não os editou.
- Os dois foram para edições em todas as colunas nas últimas camadas (edição direta dos logits de "Wait"), uma
  família diferente da a1/a2 e mais próxima do Astra na forma.
- Nos dois, a sonda de melhor nota não foi o melhor resultado por geração.
- Os dois pararam com metade do orçamento, por tempo.

**Previsões do pré-registro (10h50) já decididas.**
- 4 (alguma run edita L22n573 ou um neurônio-freio em L17–L23): **errou** (eu dava 65%).
- 5 (menos sondas que o Opus até a primeira checagem positiva): a conferir nos registros.
- 1–3 dependem do dev48.

**Desvios.**
- As duas runs pararam ~50 min–1 h no limite de uso de 5 horas da assinatura e foram retomadas por mensagem ("continue
  de onde parou"). O avaliador ficou ativo.
- A fila compartilhada deixou as checagens por geração lentas (uma levou mais de 1 h), e isso encurtou as duas runs.

**Auditoria das chamadas de ferramenta (f1: 145 Bash, 7 Read; f2: 42 Bash, 5 Read).** Nenhuma busca na web, nenhuma
leitura do repositório do paper, de `docs/`, `scripts/wsp/`, `probes.py`, do atlas, de outras runs ou de
`~/.claude/`. Única ocorrência: as duas rodaram `ls` na pasta `sealed/` da própria run no início (vazia naquele
momento; nenhum arquivo lido).

**Confirmação (`scripts/queue_fable.sh`, unit `wsp-fable`, 18h36):** nº 1 de cada no dev48; a melhor (maior acerto,
empate → menos tokens) vai ao t200; em seguida roda a fila 2 (pré-registro das 16h55).

### 10/11, 21h55 — f1 e f2 no dev48

| braço | acerto | cortes | tokens (média) | contra o original | contra a a1 do paper | contra o Astra |
|---|---|---|---|---|---|---|
| original | 0,146 | 0,854 | 11.714 | | | |
| Astra | 0,458 | 0,479 | 9.404 | | | |
| a1 do paper | 0,667 | 0,167 | 4.729 | | | |
| a2 | 0,646 | 0,083 | 3.910 | | | |
| **f1** | 0,646 | 0,250 | 6.711 | +25 / −1 | +6 / −7 (p = 1,0) | +9 / −0 (p = 0,004) |
| **f2** | 0,604 | 0,271 | 6.618 | +22 / −0 | +4 / −7 (p = 0,55) | +10 / −3 (p = 0,09) |

f1 contra f2: +9 / −7.

**Previsões do pré-registro (10h50).**
- 1 (alguma run ≥ 0,60): **acertou** (as duas).
- 2 (as duas ≥ 0,50): **acertou**.
- 3 (alguma supera a a1 por ≥ 5 pp): **não aconteceu**, como previsto (eu dava 20%).
- Leitura combinada: nenhuma ficou < 0,40, e os editores **não** convergiram para os mesmos neurônios. O acerto é o
  mesmo da a1/a2 dentro do ruído de 48 problemas, mas com ~40–70% mais tokens e mais cortes no teto (0,25 contra
  0,08–0,17). Quatro runs, três famílias de edição, mesmo acerto: o ganho não depende de achar aqueles neurônios; o
  custo em tokens, sim.

**Fila.** f1 venceu (0,646 > 0,604) e está no t200 desde 19h55 (160/200 às 21h53). Depois entra a fila 2.

### 10/11, 22h20 — Extensão das runs f1 e f2 (f1x, f2x). PRÉ-REGISTRO

**Correção.** Escrevi às 18h40 que f1 e f2 "pararam com metade do orçamento". Em sondas isso engana: a1 usou 18 e a2
17 das 40; f1 usou 22 e f2 21. A diferença real está nas checagens por geração: a1 e a2 usaram as 6; f1 e f2 só 4.

**Causa.** O prompt (igual ao do a2) pede terminar em ~3,5 h de relógio. As duas runs ficaram ~1 h paradas no limite
de uso e contaram essa hora; a fila compartilhada do avaliador também atrasou as checagens. As runs pararam por
relógio, não por falta de ideias.

**Extensão (pedido do Luiz).** Retomar as duas no mesmo contexto, com o orçamento que resta (f1: 18 sondas, 2
checagens; f2: 19 sondas, 2 checagens) e ~1h15 de relógio. Mesmo brief, mesmas regras, mesmo avaliador.

**Como relatar.**
- A escolha nº 1 das 18h40 continua sendo o resultado pré-registrado de f1 e f2. O dev48 delas já foi visto por mim,
  não pelos agentes; a mensagem de retomada não menciona dev48, t200, custo em tokens nem as edições a1/a2.
- Se o nº 1 mudar, o novo vai ao dev48 como `m2a_f1x` / `m2a_f2x`, relatado como extensão posterior. Não vai ao t200
  a menos que supere a f1 no dev48 por ≥ 5 pp.
- A fila 2 fica para depois da extensão (a GPU de 6 GB não comporta avaliador e llama-server juntos).

**Previsões.**
- Alguma run muda o nº 1: 55%.
- Algum nº 1 novo chega a ≥ 0,70 no dev48: 15%.
- Alguma run volta aos neurônios de "wait" (L17n1915, L18n881, L22n573) na escolha final: 20%.
- Tokens no dev48 do nº 1 novo continuam > 5.000: 70% (o avaliador não cobra tokens).

### 10/11, 22h35 — f1 no t200 (braço posterior); extensão lançada; erro na fila 2

**f1 no t200 (200 problemas, teto 12.288):** acerto 0,660; cortes 0,29; loop no teto 0,245; média de 6.534 tokens;
101 resgates / 0 perdas contra o original.

| contra | ganhos / perdas | p |
|---|---|---|
| a1 do paper (0,615) | +28 / −19 | 0,24 |
| nº 1 do agente a1 (0,620) | +32 / −24 | 0,35 |
| a2 (0,675) | +21 / −24 | 0,77 |
| Astra (0,535) | +40 / −15 | 0,001 |
| bf4096 (0,610) | +28 / −18 | 0,18 |
| a1 + bf4096 (0,695) | +19 / −26 | 0,37 |
| original sem raciocínio (0,585) | +36 / −21 | 0,06 |

- Acerto por teto (2.048 / 4.096 / 6.144 / 8.192 / 12.288): 12,0 / 40,5 / 52,5 / 61,5 / 66,0. Com 29% cortadas no
  teto e a curva ainda subindo, a f1 depende do teto como o Astra, mas num nível mais alto.
- Acerto entre as respostas terminadas: 93% (132 de 142), contra 74% da a1 e 80% da a2. A f1 não perde nenhum
  problema que o original acertava. Ela corta menos loops, mas estraga menos respostas.
- Sem correção para múltiplas comparações; braço posterior à abertura do t200.

**Extensão.** Avaliador religado às 22h30; f1 e f2 retomadas às 22h33 com a mensagem pré-registrada (sem dev48,
t200, tokens ou a1/a2). Os `final.json` das 18h40 foram guardados como `final_1840.json`.

**Erro na fila 2.** `queue_fix2.sh` chamava a varredura do dev48 sem `--split train`; o `gen_eval.py` recusou (split
de teste trancado) e a fila seguiu com vencedor vazio. Nenhum problema do teste foi gerado para os braços novos: o
braço `m2a_a1_062959_102+` (a1 sem limite, duplicata) começou e foi parado e apagado antes de terminar. Corrigido
(`--split train` e parada se não houver vencedor). A fila 2 roda depois da extensão.

### 10/11, 23h30 — Extensão: f1 terminou; fila da madrugada lançada

- **f1x:** nº 1 novo `224719_221` = edição das 18h40 + "Yes" em `layers.23.self_attn.o_proj` (α = 1). Checagem: 11/16,
  cortes 0,19, média de 3.958 tokens, 10 resgates / 0 perdas (18h40: 9/16, cortes 0,375). Usou 33/40 sondas e 6/6
  checagens. Testou L22n573 somado à edição: piorou. Diferenças entre os três do ranking dentro do ruído de 16.
- **Desvio:** segundo limite de uso às ~23h00; as duas runs retomadas às 23h11 (pausa de ~10 min, avaliador ativo).
- **f2x:** ainda rodando (33 sondas, 5 checagens às 23h29).
- **Fila da madrugada (`scripts/queue_night.sh`, unit `wsp-night`):** espera o `final.json` novo da f2 até 00h50
  (depois disso segue sem ela); copia as runs; confirma no dev48 cada nº 1 que mudou (`m2a_f1_224719_221`, e o da f2
  se mudar); roda a fila 2 corrigida. A auditoria das chamadas da extensão fica para depois da fila; os braços são
  posteriores e relatados assim.

### 10/11, 23h45 — Extensão: f2 terminou; auditoria; fila seguiu

- **f2x:** nº 1 **não mudou** (`162904_881`, 10/16, cortes 0,25). Usou 36/40 sondas e 6/6 checagens. As duas
  checagens novas: `223733_914` (L19, 21, 22, 23, α = 5) 10/16 com cortes 0,375; `232419_040` (com edição de "But")
  9/16. A direção "But" atinge os casos de sonda 1–6, mas com dano no indicador neutro; descartada.
- **Auditoria das chamadas da extensão** (f1: 23 Bash; f2: 13 Bash; nenhuma outra ferramenta além da entrega do
  relatório): nenhuma busca na web, nenhuma leitura do repositório do paper, de `sealed/`, `docs/`, `scripts/wsp/`,
  do atlas, de outras runs ou de `~/.claude/`. As únicas ocorrências dos termos vigiados são a palavra "atlas" no
  texto do `final.json` de cada run ("No atlas was used").
- **Previsões da extensão (22h20):** "alguma run muda o nº 1" (55%): **acertou** (f1). "Alguma volta aos neurônios de
  'wait' na escolha final" (20%): **não aconteceu**, como previsto. As duas sobre o dev48 dependem da confirmação.
- **Fila:** às 23h39 a `wsp-night` seguiu: avaliador parado, `m2a_f1_224719_221` no dev48, depois a fila 2.

## 2026-11-11 (09h10) — Resultados da madrugada: f1x no dev48 e fila 2

**f1x no dev48 (`m2a_f1_224719_221`, com a edição de "Yes"):** 0,562; cortes 0,31; média de 7.181 tokens.
- Contra a f1 das 18h40 (0,646): +2 / −6 (p = 0,29). Contra a a1: +5 / −10. Contra o Astra: +7 / −2.
- A extensão **não melhorou** a f1: os 11/16 da checagem eram ruído de 16 problemas. Pela regra (22h20), a f1x não
  vai ao t200. O resultado de f1 no paper continua sendo a escolha das 18h40.
- Previsões da extensão: "algum nº 1 novo ≥ 0,70 no dev48" (15%): **não aconteceu**, como previsto. "Tokens do nº 1
  novo > 5.000" (70%): **acertou**.

**Varredura do limite com teto de resposta (dev48, modelo original):**

| braço | acerto | cortes | tokens (média) |
|---|---|---|---|
| bf4096 (sem teto de resposta) | 0,479 | 0,458 | 8.073 |
| bf1024a1024 | 0,458 | 0,479 | 1.692 |
| bf2048a1024 | 0,458 | 0,417 | 2.649 |
| **bf4096a1024** | 0,500 | 0,375 | 4.679 |

Vencedor pela regra: `bf4096a1024`.

**t200 (200 problemas, teto 12.288; braços posteriores):**

| braço | acerto | cortes | tokens (média) | acerto entre terminadas |
|---|---|---|---|---|
| bf4096a1024 | 0,600 | 0,35 | 4.596 | 0,92 |
| a1 + bf4096a1024 | 0,695 | 0,15 | 3.172 | 0,81 |
| lbx_w10_t1 | 0,600 | 0,355 | 7.758 | 0,93 |

- bf4096a1024 contra bf4096: +14 / −16. O teto de resposta não muda o acerto e corta os tokens de 7.012 para 4.596.
- bf4096a1024 contra a a1: +20 / −23 (p = 0,76). Contra o original sem raciocínio: +30 / −27.
- a1 + bf4096a1024 contra a1 + bf4096: +17 / −17 (mesmo acerto, 3.172 contra 4.081 tokens). Contra a a1: +26 / −10
  (p = 0,011). Contra bf4096a1024: +33 / −14 (p = 0,008). Contra o original sem raciocínio: +37 / −15 (p = 0,003).
  Contra a a2: +27 / −23. Contra a f1: +24 / −17.
- lbx_w10_t1 contra bf4096: +23 / −25; contra a a1: +25 / −28; contra o Astra: +28 / −15 (p = 0,07).
- Acerto por teto (2.048 / 4.096 / 6.144 / 8.192 / 12.288): bf4096a1024 2,5 / 5,0 / 60,0 / 60,0 / 60,0;
  a1 + bf4096a1024 25,5 / 54,5 / 69,5 / 69,5 / 69,5; lbx_w10_t1 10,0 / 26,5 / 41,5 / 50,5 / 60,0.
- Sem correção para múltiplas comparações.

**Previsões da fila 2 (16h55).**
- bf4096a1024 no dev48 em 0,44–0,52 com < 5.500 tokens: **acertou** (0,500; 4.679).
- bf1024a1024 abaixo de bf4096a1024 no dev48: **acertou**.
- Vencedor sozinho no t200 em 0,56–0,64 e a menos de 2 EP da a1 e do raciocínio desligado: **acertou** (0,600).
- a1 + vencedor em 0,65–0,72 e acima do raciocínio desligado (p < 0,05): **acertou** (0,695; p = 0,003).
- lbx_w10_t1 em 0,40–0,52: **errou para baixo** (0,600). A penalidade de logit com janela empata com o limite de
  raciocínio e com a a1 no acerto, com mais tokens.
- ft_sft_ep2: rodando (104/200 às 09h03).

**Regras de texto.** O vencedor não ficou ≥ 2 EP acima da a1: nada muda. a1 + vencedor ficou acima do raciocínio
desligado: a frase "só a combinação fica claramente acima" fica (junto com a a2, p = 0,018). Entra no paper: a
penalidade de logit (60,0%) é mais uma alternativa sem edição que empata com a a1.

### 11/11, 11h30 — Fim da fila 2; paper atualizado

- **ft_sft_ep2 no t200:** 0,330; cortes 0,655; média de 9.354 tokens. Contra o original +44 / −9; a1 contra ele
  +69 / −12. Previsão 0,22–0,34: **acertou**. Fim da fila 2 às 10h39.
- **Decisão (Luiz):** sem orçamento para repetir as runs do Fable em condições limpas. Elas entram no paper como
  réplica em condições diferentes, sem comparação com a1/a2. A escada de modelos editores (incluindo menores) fica
  para um estudo seguinte; o paper fecha com o que há.
- **Paper (en e pt-BR):** seção nova "Outro modelo editor acha a mesma correção?"; tabela por orçamento com 5 braços
  tardios novos (LoRA, proibir a palavra, limite com teto, a1 + limite com teto, f1); tabela principal; proibir a
  palavra empata no teste trancado; limite com teto; combinação com 3.172 tokens; limitações; previsões 8 e 9 na
  lista das que falharam; tabela de status.
- **Figuras:** Figura 5 com a a2 e os braços novos no teste trancado; Figura 6 com painel do teste trancado (três
  painéis empilhados); Figura 3 com 26 pontos (r = −0,98, antes −0,99 com 20). O t12k passa a se chamar "segunda
  prática" também nas figuras.
- **PDFs:** en 22 pp, pt-BR 23 pp, impressão 25 pp (uma caixa 3,8 pt acima da margem) e 28 pp.
- README e `docs/experiments.md` atualizados.

### 11/11, 11h45 — Caminho da a2: ativações dentro do modelo editado. PRÉ-REGISTRO

`scripts/act_a2_path.py` → `data/act_a2_path.json`. Mesmos 12 textos em loop do base, 1.500 tokens, mesmas 52 posições
antes de "Wait" da medição das 11h15 de ontem. Mede:
1. Δ log p("Wait") antes de "Wait" com cada neurônio da a2 editado sozinho e com todos menos um.
2. A ativação de cada neurônio da a2 dentro do modelo editado (a2 inteira; só L16; só L16 + L17), contra o base.
3. A escrita direta de cada neurônio na direção de "Wait" dentro do modelo editado.

**Previsões.**
- L16n998 sozinho dá ≥ 60% dos −26 nats: 55%.
- Dentro do modelo editado, a ativação de L22n573 antes de "Wait" troca de sinal ou cai para < 30% do valor do base
  (−3,73): 50%.
- Tirar só a inversão de L22n573 deixa a supressão de "Wait" mais forte (a inversão atrapalha): 40%.

### 11/11, 12h20 — Caminho da a2: resultado

`data/act_a2_path.json`. 12 textos, 52 posições antes de "Wait", texto fixo.

| variante | Δ log p("Wait") antes de "Wait" |
|---|---|
| a2 inteira | −25,9 |
| só L16n998 | −26,2 |
| todos menos L16n998 | +1,6 |
| só L16n3422 | −0,01 |
| só L17n1915 | +0,28 |
| só L22n573 | +1,56 |
| todos menos L22n573 | −26,3 |

Ativação média antes de "Wait", base → dentro da a2: L22n573 −3,73 → −0,04 (ainda negativa em 96%); L17n1915
−0,23 → +0,01; L16n998 −2,60 (igual; nada a montante foi editado). Só com as edições da L16 o resultado é o mesmo.

**Leitura.** A a2 é, na prática, a edição de um neurônio: inverter L16n998 dá os 26 nats. Com ele invertido, o freio
L22n573 quase para de disparar, e a inversão dele pela a2 não tem sobre o que agir. a1 e a2 mexem no mesmo circuito
em pontos diferentes: a1 reforça o freio (L22), a2 tira o empurrão (L16). Limite: são log-probabilidades com texto
fixo; a edição de um neurônio só não foi testada por geração.

**Previsões.** L16n998 sozinho ≥ 60% do efeito (55%): **acertou** (100%). Ativação de L22n573 cai para < 30% (50%):
**acertou** (1%). Tirar a inversão de L22n573 deixa a supressão mais forte (40%): **acertou por pouco** (−26,3
contra −26,0; diferença de 0,3 nat, irrelevante).

**Paper:** parágrafo do mecanismo reescrito ("Neurônios em comum, alavancas diferentes"), TL;DR, trabalhos
concorrentes, limitações e status.

### 11/11, 15h15 — Caminho da a2 por geração: pré-registro

A medição das 12h20 usou texto fixo. Agora o teste é com o modelo gerando. Dois braços no dev48 (GSM8K train[0:48],
teto 12k, mesmas condições da a2), pareados com a a2 (31/48 = 64,6%; cortadas 8,3%; 3.910 tokens):
- `m2a_a2_red`: L16n998 (α = +10) + a edição de `</think>` na L22 `linear_attn.out_proj` (α = −4). São 2 das 4
  edições da a2, com os mesmos u e α.
- `m2a_a2_l16n998`: só L16n998.

`scripts/make_a2_variants.py` → `data/m2a/variants/`; GGUF por `scripts/poc7_build.py`; fila `scripts/queue_a2_red.sh`.
Sem t200: não crio mais um braço posterior no teste trancado.

**Previsões.**
- `m2a_a2_red` fica a até 4 problemas da a2 (27–35 de 48): 65%.
- `m2a_a2_l16n998` fica pelo menos 4 problemas abaixo de `m2a_a2_red`: 55%.
- `m2a_a2_l16n998` corta mais que `m2a_a2_red` (pelo menos 4 problemas cortados a mais): 55%.

**Limite.** Com 48 problemas, o erro padrão é de ~7 pontos; só diferenças grandes aparecem.

### 11/11, 17h00 — Caminho da a2 por geração: resultado

`data/gen_dev48.json` (fila `wsp-a2red`, 15h15 → ~16h50). dev48, teto 12k, pareado.

| braço | acertos | cortadas | tokens |
|---|---|---|---|
| a2 (4 edições) | 31/48 (64,6%) | 4 | 3.910 |
| a2 reduzida (L16n998 + `</think>`) | 33/48 (68,8%) | 6 | 4.797 |
| só L16n998 | 23/48 (47,9%) | 25 | 8.617 |

Pareados (sinal exato): reduzida × a2 +6/−4 (p = 0,75); só L16n998 × reduzida +3/−13 (p = 0,02); só L16n998 × a2
+3/−11 (p = 0,06); reduzida × a1 +4/−2; só L16n998 × base +19/−3.

**Leitura.** A medição com texto fixo previu bem o comportamento. A a2 é, na prática, uma correção de 2 edições: tirar
L16n3422, L17n1915 e L22n573 não muda nada mensurável. Mas baixar "Wait" sozinho não basta: sem a edição de
`</think>`, metade das respostas estoura o teto, como nas ablações da a1 (supressão sozinha 44–50%, os dois juntos 69%).
A reduzida escreve um pouco mais (2 cortes a mais respondem por boa parte da diferença).

**Previsões.** Reduzida a até 4 problemas da a2 (65%): **acertou** (+2). Só L16n998 ≥ 4 abaixo da reduzida (55%):
**acertou** (−10). Só L16n998 com ≥ 4 cortes a mais que a reduzida (55%): **acertou** (+19).

**Paper:** parágrafo do mecanismo e limitação atualizados (en e pt-BR). Sem t200 para as variantes.

### 11/11, 17h40 — Revisão independente das mudanças desde c51b187

Um agente novo, sem o meu contexto e só com leitura, conferiu contra os dados brutos o que mudou no paper e no README
desde a revisão de 10/11. Todos os números conferem. As correções foram de afirmação e redação, e as apliquei em en e
pt-BR depois de conferir as citações do CADERNO:
- **Braços tardios:** não é verdade que "não são pré-registrados". A proibição da palavra e o LoRA estavam no
  pré-registro original de 08/11 e só rodaram tarde. Os outros foram pré-registrados com previsões depois da
  abertura. A lista ganhou "a1 + limite com teto" (são 8), e a tabela de status foi corrigida.
- **TL;DR e raciocínio desligado:** a a2 contra o raciocínio desligado dá p = 0,02 sem correção e não sobrevive à de
  Holm. Agora fica "acima com menos firmeza"; só a a1 + limite fica "claramente acima".
- **Seção do Fable:** as comparações com a1 e a2 agora estão marcadas "só para orientação"; a tabela de orçamento
  ganhou "outras condições"; a regra "sem subagentes" entrou na lista; a extensão da f1 ganhou p = 0,29; a previsão 5
  (sondas) é dita não avaliada.
- **Mecanismo:** "outras duas edições" virou "outros três neurônios"; "um pouco mais longas" virou "mais longas"; a
  ativação do freio foi medida com os dois neurônios da L16, não só com o L16n998.
- **Previsão 8 do Fable:** citada como foi pré-registrada.
- **Figura 3:** agora diz que os 26 pontos são de antes das ablações da a2 (r = −0,98 com 26 ou 28).
- **README:** "convergiram para o mesmo mecanismo" e "vence o logit bias em todo lugar" foram corrigidos.

**Previsão 5 do Fable** (menos sondas que o Opus até o primeiro teste positivo, 55%): **não avaliada**, porque as
interrupções tornam ambíguo o que é "primeiro".

### 11/11, 18h30 — Fable reduzido a uma menção breve (decisão do Luiz)

As execuções f1/f2 foram uma checagem lateral, não o centro do estudo. No paper, a seção virou um parágrafo
("Uma checagem rápida com outro modelo editor"): diferenças de condição, o que as execuções acharam, 65% e 60% no
dev48 e 66,0% da f1 no t200, e a ressalva de que não dizem qual editor é melhor. Saíram das tabelas, do TL;DR, do
README e da Figura 3 (agora 25 pontos, r = −0,98). Os detalhes continuam neste CADERNO (entradas de 10 e 11/11) e em
`docs/experiments.md`. A escada de modelos editores fica para a continuação.

### 11/11, 19h00 — Segundo teste novo (t400): pré-registro

GSM8K test[400:600], 200 problemas que nenhum braço tocou (conferido no ledger: só [0:32], [0:200] e [200:400] foram
usados). Tag `t400` (raciocínio ligado, teto 12.288) e `t400nt` (desligado, teto 8.192), mesmas condições do t200.
Todos os braços abaixo foram escolhidos antes de abrir. Nada é escolhido ou reescolhido no t400. Ordem de prioridade
(`scripts/queue_t400.sh`), para que um corte preserve as comparações principais:
1. modelo original, raciocínio desligado (`t400nt base`);
2. a2 (`m2a_a2`);
3. a1 + limite com teto (`m2a_a1_062959_102+bf4096a1024`);
4. a2 reduzida (`m2a_a2_red`);
5. limite com teto sozinho (`bf4096a1024`);
6. a1 (`m2a_a1_062959_102`).

**Análise primária** (sinal exato pareado, Holm sobre as 5): a2 × raciocínio desligado; a1 + limite × raciocínio
desligado; a1 + limite × limite sozinho; a2 reduzida × a2; a2 × a1. **Secundária:** t200 + t400 juntos (400
problemas) para as mesmas comparações.

**Previsões.**
- Acertos no t400: raciocínio desligado 0,53–0,64; a2 0,60–0,72; a1 + limite 0,63–0,75; a2 reduzida 0,58–0,72;
  limite sozinho 0,54–0,66; a1 0,55–0,67.
- a1 + limite acima do raciocínio desligado com p < 0,05 no t400 sozinho: 70%.
- a2 acima do raciocínio desligado com p < 0,05 no t400 sozinho: 45%; com t200 + t400 juntos: 65%.
- a2 reduzida a até 5 pontos da a2 no t400: 65%.
- a2 reduzida com média de tokens ≥ a da a2 (como no dev48): 60%.

**Uso:** o t400 também é a validação da a2 (e da reduzida) para o Hugging Face.

### 12/11, 00h45 — MATH sem raciocínio, a2 e a2 reduzida no MATH-148: pré-registro

Janela de GPU liberada até 09h00. A fila `scripts/queue_math_nt.sh` espera o t400 acabar e roda, no MATH-500[0:148]
(os mesmos 148 problemas do math48 + math100b):
1. modelo original sem raciocínio, teto 8.192 (`math148nt base`, `math_eval.py --nothink`, flag nova; amostragem
   non-thinking, como no t200nt);
2. a2 (`math148 m2a_a2`, raciocínio ligado, teto 12.288);
3. a2 reduzida (`math148 m2a_a2_red`).

Prazo duro às 08h50: um braço que não termine até lá é interrompido e fica de fora. Nenhum braço é escolhido no
MATH; a2 e a2 reduzida foram fixadas antes (a reduzida, pelo teste por geração no dev48). Os 148 problemas já foram
usados por outros braços, então isto não é teste trancado; serve para fechar a limitação "sem raciocínio no MATH" e
para checar a a2 fora do GSM8K antes do Hugging Face.

**Comparações** (sinal exato pareado, sem correção; exploratórias): a1 + limite (0,527) × sem raciocínio; a1 (0,466)
× sem raciocínio; a2 × sem raciocínio; a2 × a1; a2 reduzida × a2.

**Previsões.**
- Acertos: sem raciocínio 0,38–0,52 (o exp8-loop-controls viu 49,9% no MATH-500 inteiro com teto 32k e corretor
  melhor; o nosso corretor é leve); a2 0,38–0,54; a2 reduzida 0,38–0,54.
- a1 + limite acima de sem raciocínio com p < 0,05: 30%. a1 acima com p < 0,05: 15%.
- Sem raciocínio acima de a1 (diferença positiva, qualquer p): 45%.
- a2 a até 5 pontos da a1: 55%. a2 reduzida a até 5 pontos da a2: 60%.
- Sem raciocínio usa < 1/3 dos tokens da a1: 85%.
