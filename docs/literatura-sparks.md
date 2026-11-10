# Literatura em torno do *Sparks of Weight-Space Programming* (levantamento de 10/11/2026)

O *Sparks* (Takehi, Teoh, Alex Zhang, Khattab; MIT CSAIL) saiu como post de blog em 02/11/2026. Não há versão no arXiv;
a citação é `@misc` (chave `takehi2026weightspace`). Este documento registra quem já partiu dele, onde o nosso trabalho
se encaixa e o que é urgente. Rótulos: **[V]** = página primária aberta e conferida; **[A]** = só pelo resumo de uma
linha na lista comunitária `SajjadPSavoji/awesome-weight-space-programming`; **[S]** = trecho de busca ou cobertura
secundária.

## 1. O tamanho da onda

- **arXiv [V]:** 11 preprints com "weight-space programming" entre 03 e 09/11, e pelo menos mais 6 relevantes que não
  usam o termo. São cerca de 2 por dia.
- **Lista comunitária [A]:** 92 entradas desde 02/11, das quais 72 são repositórios no GitHub e 14 são papers.
- **Imprensa e discussão [S]:** VentureBeat, The Register, LessWrong, Hacker News e um assunto em alta no X.
- **O que não existe ainda:** nenhum trabalho revisado por pares, nenhum workshop dedicado e nenhuma contagem de
  citações no Semantic Scholar (a API limitou o acesso).
- **Leitura:** a atenção é real e muito rápida, mas quase toda sem revisão. Vários trabalhos foram escritos com ajuda de
  agentes, e um paper do arXiv declara ter sido gerado por IA.

## 2. O que o próprio *Sparks* mede no experimento 8 [V]

A fonte é `exp8_math/README.md` e `eval/*/merged.json` no repositório de registros.

**Protocolo:** raciocínio **ligado**, até 32.768 tokens novos, vLLM, temperatura 0,6, presence penalty 1,5. Uma resposta
sem `</think>` vale zero.

| teste | original | original: cortadas | editado | editado: cortadas |
|---|---|---|---|---|
| MATH-500 | 7,9% | 93% | 48,1% | 33% |
| GSM8K | 13,8% | 87,5% | 67,1% | 7,2% |
| IFEval | 21,0% | 74,9% | 62,3% | 2,2% |

**Consequência para nós:** o "IFEval 3×" do *Sparks* é o defeito dos loops. Não contradiz o nosso resultado negativo,
que foi medido com o raciocínio desligado. Duas reexecuções independentes com o raciocínio desligado confirmam que a
edição do Astra não ajuda ali:
- exp8-loop-controls: de 52,8% para 50,5%;
- Rethinking WSP: −2,2 contra a melhor decodificação sem edição.

**Detalhe técnico:** o blog fala em "remover" a direção. A conta nos registros é W ← W − 6·u(uᵀW), que inverte a
componente e a amplia para −5×. O nosso texto já descreve assim.

## 3. Trabalhos que se sobrepõem ao nosso

| trabalho | o que faz | relação conosco |
|---|---|---|
| **Hu et al., arXiv:2611.09382** (06/11) [V] | 16 trajetórias de edição com o Astra no Qwen3.5-0.8B e no 9B, em 7 benchmarks. MATH-500 +29,5 e IFEval +36,7 em held-out, com raciocínio ligado. 6 de 16 trajetórias mantêm os pesos originais. Sem GSM8K, sem LoRA, sem limite de raciocínio. | Principal concorrente no arXiv. Cobrimos o que falta nele: alternativas, teste trancado, custo e Claude como editor. |
| **tdotlab/weight-space-programming** (03–04/11) [V] | Claude como editor, Qwen3.5-0.8B e 2B, GTX 1070 com llama.cpp. O freio de "Wait" é **L22 col 573**. MATH-500 (100 problemas) de 5 para 30. Edição e orçamento se somam. Uma segunda rodada, com freios de "Final", deu nula. | Sobreposição mais próxima, e **convergência independente com a nossa a1 (L22n573)**. Reforça o resultado; precisa ser citado. |
| **MatinKhajavi/Weight-Space-Programming** (03–05/11) [V] | Replica os 8 experimentos com Claude Opus 5.5. No experimento 8, as edições são de outro tipo (zerar cabeças L15H7, L23H0, L23H5; L22-573 ×2). | Acaba com "primeiro com Claude". |
| **ajaygrover/…-stacking** (07/11) [V] | O Opus acha uma direção quase ortogonal à do Astra (cos ~0,1). Somadas, chegam a 52,2% no MATH-500. | Tensão com a nossa convergência entre dois agentes; pode ser específica do Claude. |
| **1099271/exp8-loop-controls** (03–09/11, pré-registrado) [V] | Sem raciocínio e com 32k tokens, o original faz 49,9% no MATH-500, o mesmo que a edição com raciocínio. No GSM8K, a edição com raciocínio (67,1) fica 10 pontos acima do original sem raciocínio (56,9). | Complementa o nosso enquadramento com o limite de raciocínio e o nosso IFEval negativo. |
| **ZhangYiqun018, "Rethinking WSP"** (05/11) [V] | Contra a melhor decodificação sem edição, o Astra soma +6,4 no MATH-500, +11,5 no GSM8K e −2,2 no IFEval. Uma diferença de médias sem LLM recupera 79% do ganho no MATH-500. | **Possível contradição:** a nossa diferença de médias (PoC-7b) e a nossa busca automática foram fracas. Precisamos conciliar. |
| **nathanrs/weight-space-lp** (06/11) [V] | Um programa linear, dada a direção do Astra, iguala a edição com 8× menos pesos. Sem LLM chega a ~93%. | Mesma tensão: com o defeito conhecido, a contribuição do LLM pode ser pequena. |
| **clavis-systems** (07/11) [V] | GSM8K, 100 problemas, teto 16k. Com raciocínio, de 7 para 50; sem raciocínio, de 47 para 43. | Ajuda a explicar por que o nosso Astra faz 53,5% (teto 12k, llama.cpp) e não 67,1%. |
| **WKJBryan/wsp-lab** (05/11) [V] | Reparo às cegas, sem dica do defeito, com editor GLM-5.3: resultado **nulo**. | A dica no prompt importa. O nosso briefing descreve o defeito, e isso deve ficar explícito no texto. |
| **IWIBench, arXiv:2611.08647** [V] | Benchmark de editores sobre o Gemma 3 4B, com 10 objetivos. | O mais perto de um benchmark padrão hoje. |
| **WSPBench** (wspbench.com) [V] | Em redes pequenas, pesos + sondas consertam 30/32 defeitos, só sondas 14/32 e um solver por script 5/32. | Evidência **a favor** de "raciocínio > busca por script", mesmo que em redes pequenas. |

Também relevantes:
- **Críticas metodológicas:** Liu & Yang (arXiv:2611.07847), Honnibal (blog), Butkeraites (2611.09559).
- **Resultados negativos ou frágeis:** IyadhKhalfallah, AuremHQ e outros [A].
- **Antes do *Sparks*, com Claude como editor:** `csinva/weight-space-programming-autoresearch`, de maio de 2026.

## 4. Onde nos encaixamos

**O que não podemos mais afirmar:**
- primeiro com Claude;
- primeira replicação independente da edição do Astra;
- primeiro numa GPU de consumo;
- primeiro a mostrar que alternativas de decodificação importam.

**O que, pelo levantamento, ninguém combinou ainda:**
1. **Teste trancado de 200 problemas com o limite de raciocínio como alternativa explícita:** edição 61,5%, limite
   61,0%, edição + limite 69,5%. A soma já apareceu no tdotlab, em subconjuntos do MATH; aqui ela corrobora, não é
   novidade.
2. **Fine-tuning LoRA como alternativa, com contabilidade de GPU:** 13× menos.
3. **Busca por script com as mesmas ferramentas e o mesmo orçamento, sem raciocínio.** Precisa ser conciliada com a
   diferença de médias e o programa linear acima.
4. **Agentes burlando indicadores de interpretabilidade (Goodhart).** Não vimos isso em nenhum outro trabalho.
5. **Dois agentes independentes convergindo para a mesma edição,** agora com um terceiro grupo (tdotlab) achando o
   mesmo neurônio.
6. **Um resultado negativo bem controlado num defeito espalhado:** IFEval com o raciocínio desligado, em que agente,
   prompt e fine-tuning falham. Os outros só mostram ganhos no regime com raciocínio e cortes.

**Posição defensável:** um paper de **rigor e de fronteira de escopo**, não de capacidade. Ele combina com o consenso
que está se formando: "a edição remove bloqueios, não adiciona habilidades" (o próprio *Sparks*, 2611.09376,
2611.11894, a issue #8).

**A crítica pública mais forte a responder:** "é só desligar o raciocínio".
- O que temos hoje: 48,5% sem raciocínio em outros 200 problemas, com teto de 1.024.
- O que falta: rodar sem raciocínio nos mesmos problemas do t200, com teto longo. Custa ~1 GPU-h. Se der ~56%, a edição
  com limite (69,5%) continua bem acima, e a edição sozinha (61,5%) fica alguns pontos acima.

## 5. Urgência

- Cerca de 2 preprints relevantes por dia. Vários repositórios (tdotlab, ajaygrover, exp8-loop-controls) podem virar
  paper no arXiv a qualquer momento.
- Os autores do *Sparks* anunciaram um benchmark e um harness.
- **Recomendação:** ir ao arXiv em **dias, não semanas**, com o enquadramento acima.
- Antes disso, o mínimo:
  1. a linha de base sem raciocínio no t200;
  2. o parágrafo de trabalhos concorrentes (já no rascunho);
  3. deixar explícito que o briefing descreve o defeito ao agente.
- Nenhum prazo de workshop do ICLR 2027 foi verificado. Os de workshops costumam cair em janeiro ou fevereiro.

## Fontes principais

- **Sparks:** <https://rikiyat.github.io/blog/weight-space-programming/>. Registros:
  <https://github.com/RikiyaT/lm_understand-logs> (`exp8_math/`, issues #4, #5, #8).
- **arXiv:** 2611.09382, 2611.08647, 2611.07847, 2611.08070, 2611.05341, 2611.12066, 2611.11604, 2611.11537,
  2611.09559, 2611.09376, 2611.09271, 2611.07319, 2611.04268, 2611.11894, 2611.11005, 2611.08383.
- **GitHub:** tdotlab/weight-space-programming, MatinKhajavi/Weight-Space-Programming,
  ajaygrover/weight-space-programming-stacking, 1099271/exp8-loop-controls,
  ZhangYiqun018/rethinking-weight-space-programming, nathanrs/weight-space-lp,
  clavis-systems/astra-gsm8k-transformers-16k, HomoHabilis/weight-space-programming,
  di37/weight-space-programming-skill, WKJBryan/wsp-lab, alexcarterhq/wspbench,
  csinva/weight-space-programming-autoresearch, SajjadPSavoji/awesome-weight-space-programming (atualizada diariamente).
