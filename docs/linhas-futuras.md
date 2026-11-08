# Linhas de pesquisa futuras (para amadurecer)

Registrado em 08/11/2026, a partir de uma conversa entre Luiz e Claude depois do pré-paper. Ainda não são planos de
experimento: cada linha tem uma pergunta, um esboço de desenho, as dependências e o que precisaria de hardware e
orçamento. A ideia é revisitar estas linhas na proposta de criação do laboratório e na estimativa de orçamento para
2027/1.

O paper 1 continua sendo a prioridade: teste trancado com n ≥ 200 e um segundo defeito no mesmo modelo
([PLANO.md](PLANO.md)).

## Como as linhas se encadeiam

```
Linha 1: qual é o editor mínimo?  ──►  Linha 2: um editor menor que o alvo funciona?  ──►  Linha 3: um modelo edita a si mesmo?
   (eixo do editor)                     (eixo do alvo)                                     (editor = alvo, e ciclos)
```

Cada degrau responde a uma pergunta que decide se o próximo vale a pena. Se o piso da Linha 1 ficar acima do nível
de um modelo de 4–9B, a Linha 3 perde o sentido na escala que conseguimos rodar.

O achado que motiva as três linhas vem do Experimento 6: com as mesmas ferramentas e o mesmo orçamento, um script
sem raciocínio chegou a 0,271 no dev48, e o agente chegou a 0,667. O que importa é o raciocínio do agente. A pergunta
seguinte é quanto raciocínio é preciso.

## Linha 1: descer a escada de inteligência do editor

**Pergunta.** Qual é o modelo mais fraco que ainda consegue fazer programação no espaço de pesos com o nosso
protocolo? Onde fica o piso dessa capacidade?

**Desenho.**
- Manter fixos: alvo (Qwen3.5-0.8B), defeito (loops), avaliador v2 (40 sondas + 6 checagens por geração), briefing e
  orçamento.
- Variar só o modelo do agente:
  - Claude Opus (já temos a0, a1 e a2);
  - Claude Sonnet;
  - Claude Haiku;
  - modelos open-weight de tamanhos decrescentes;
  - o script sem agente do E3, que é o piso da escada.
- **Réplicas são obrigatórias.** A variância entre execuções é enorme: entre agentes Opus, o dev48 vai de 0,042 (r1)
  a 0,667 (a1). Mínimo de 3 execuções por degrau.
- **Medidas:**
  - o acerto no dev48 da escolha de cada execução;
  - a fração de execuções que acham uma edição útil (dev48 > 0,4, por exemplo);
  - o custo em tokens e GPU-h;
  - uma análise qualitativa dos registros: em que ponto os modelos mais fracos se perdem (ler os pesos pelo
    vocabulário, interpretar uma checagem que falhou, resistir à métrica rápida).

**Viabilidade.**
- **Parte Claude:** viável já. Cabe na assinatura do Claude Code e usa a infraestrutura existente. São ~6
  execuções e ~8–10 GPU-h na GTX 1660 SUPER.
- **Parte open-weight:** não roda localmente, porque o editor precisa de contexto longo e de um modelo grande.
  - Por API, o volume de uma execução nossa (~5,4 M tokens de entrada e ~50 mil de saída, `docs/cost.md`) custaria
    poucos dólares em modelos abertos. É uma estimativa grosseira: conferir os preços na hora de orçar.
  - **Confusor:** modelos abertos rodam em outro ambiente de agente, não no Claude Code. É preciso fixar um ambiente
    comum (ou ao menos rodar um modelo Claude nele também) para que a comparação seja justa.

**Riscos.** Um degrau pode falhar por não saber usar as ferramentas, não por não saber raciocinar sobre pesos. Para
separar as duas coisas, registrar a taxa de erro de chamada de ferramenta de cada execução.

## Linha 2: um modelo menor editando um modelo maior

**Pergunta.** Um editor de capacidade média consegue melhorar um alvo maior do que ele? Em que condições um "aluno"
consegue corrigir o "professor"?

**Desenho.**
- Uma matriz editor × alvo, com o mesmo protocolo e tamanhos crescentes de alvo.
- Para isso, é preciso um defeito que exista em todos os tamanhos de alvo. Os loops diminuem nos modelos maiores, e
  os defeitos de saída estruturada do paper 1 são candidatos.
- **Pergunta auxiliar:** as ferramentas que funcionaram no 0.8B escalam? Ler neurônios pelo vocabulário fica mais
  difícil com mais camadas e neurônios mais polissemânticos.

**Viabilidade.**
- **Hardware é o gargalo.** Na GTX 1660 SUPER (6 GB, sem bf16), alvos acima de ~2B em precisão total não cabem com
  12 mil tokens de contexto. Editar pesos quantizados é outro problema, bem mais difícil.
- **Editor menor que o alvo:** precisa ser um modelo aberto rodando localmente, junto com o alvo, ou por API.
- **Ordem de grandeza:** uma GPU de 24 GB permitiria alvos até ~8B em fp16 com menos slots paralelos; uma de 48 GB,
  com folga. Orçar na proposta.

**Dependências.** O piso da Linha 1. Se o Haiku já não consegue editar, um modelo aberto de 9B dificilmente vai
conseguir.

## Linha 3: autoedição e ciclos recursivos

**Pergunta.** Um modelo pequeno (4–9B) consegue corrigir a si mesmo com estas técnicas? Repetir o ciclo (o modelo
editado edita a próxima versão) converge, melhora ou degenera?

**Desenho (esboço).**
- O modelo roda como agente, com o mesmo avaliador, e o alvo é uma cópia congelada dos seus próprios pesos.
- Ciclo k: a versão k edita uma cópia de si → avaliação externa → a versão k+1 só é aceita se passar no teste real.
- Medir a trajetória:
  - acerto no defeito-alvo;
  - efeitos colaterais (painel fixo);
  - a capacidade de edição em si (a versão k+1 edita melhor ou pior que a k?).

**A âncora externa é inegociável.** O Experimento 7 mostrou que toda métrica rápida foi burlada assim que um
otimizador forte a pressionou. O que manteve os agentes honestos foi um teste real, externo, mesmo pequeno. Um modelo
que edita a si mesmo e avalia a si mesmo é o pior caso possível desse problema. Por isso, nesta linha:
- a régua (testes reais, painel de efeitos colaterais, teste trancado) fica sempre fora do modelo;
- o modelo não pode ler nem alterar o código do avaliador.

Isso também é um ângulo de segurança publicável: "a autoedição só é estável com uma âncora externa". Já temos uma
evidência inicial disso.

**Viabilidade.**
- O modelo precisa rodar ao mesmo tempo como agente, com contexto longo, e como alvo avaliado. Não cabe em 6 GB;
  provavelmente exige ≥ 48 GB de VRAM, ou duas GPUs.
- Modelos de 4–9B ainda têm dificuldade com ciclos de agente de ~2 horas. A Linha 1 diz se faz sentido tentar.

## Outras linhas já anotadas no plano

- **F4: escada de capacidade com redes de brinquedo** (MinAtar, transformer de caractere): onde a edição agêntica
  funciona em redes pequenas o bastante para termos a verdade de referência do mecanismo.
- **Coding (MBPP)** como terceira tarefa-alvo.
- **Edição + decodificação** como receita combinada: a1 + budget forcing foi o melhor resultado em todos os conjuntos
  do paper 1.

## Perguntas abertas para a proposta do laboratório (2027/1)

1. **GPU:** qual é a menor configuração que destrava as Linhas 2 e 3? 24 GB serve para a Linha 2 com alvos de até
   ~8B; 48 GB, ou 2 × 24 GB, para a Linha 3. Comparar compra, nuvem acadêmica e créditos de pesquisa.
2. **API:** quanto reservar para editores open-weight na Linha 1? Na estimativa grosseira de hoje, são poucos dólares
   por execução × ~3 réplicas × ~4 modelos. Pedir créditos acadêmicos aos provedores.
3. **Pessoas:** cada execução de agente precisa de auditoria dos registros (conformidade com as regras) e de
   anotação qualitativa. Esse trabalho cabe bem a alunos de iniciação científica.
4. **Reprodutibilidade:** publicar o avaliador, as regras e os registros como um ambiente de teste aberto
   ("benchmark de editores"), para que outros grupos meçam seus modelos na mesma escada.
5. **Segurança e ética da Linha 3:** que salvaguardas formais exigir (âncora externa, registro de todas as versões,
   critério de parada) antes de rodar ciclos recursivos?
