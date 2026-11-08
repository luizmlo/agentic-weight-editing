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
