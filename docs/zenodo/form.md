# Ficha do formulário do Zenodo (registro do paper)

DOI reservado: **10.5281/zenodo.24208535**. Campos na ordem em que aparecem na página de upload. O que estiver em bloco de
código é para colar como está.

## Comunidade
Nenhuma (deixe em branco). Dá para submeter a uma comunidade depois de publicado.

## Arquivos
Envie só os dois PDFs de tela, gerados depois de o DOI entrar no texto:
- `docs/pre-paper.pdf` (inglês)
- `docs/pre-paper.pt-BR.pdf` (português)

As cópias `.print.pdf` (margem para anotações) ficam de fora. Depois de publicar, os arquivos não podem mais ser trocados;
correções entram como **New version**.

## Basic information
- **DOI:** "Yes"/já reservado: `10.5281/zenodo.24208535` (não clique em reservar de novo).
- **Resource type:** `Publication` → `Preprint`
- **Title:**
```
Agentic Weight-Space Programming as an Alternative to Fine-Tuning in Small Language Models
```
- **Publication date:** a data em que você clicar em publicar, no formato `AAAA-MM-DD` (hoje: `2026-11-13`). Se publicar
  outro dia, troque também a data no topo do paper e gere os PDFs de novo.
- **Authors/Creators:** Person · Family name `Melo` · Given names `Luiz Felipe` · ORCID `0009-0007-5134-9914` ·
  Affiliation `Universidade do Sul de Santa Catarina` · Role: deixe em branco (criador).
- **Description:** cole os dois parágrafos abaixo, um depois do outro. Ponha em negrito só as palavras "English." e
  "Português." e em itálico o título *Sparks of Weight-Space Programming*.
```
English. Small reasoning language models often fail by looping: with thinking on and a 12,288-token budget, Qwen3.5-0.8B runs out of space before answering on about 85% of grade-school math problems. We ask whether frontier-model agents can repair this defect by editing the weights directly, without gradients, and how the result compares with simpler fixes. Claude agents, given the weights, a small testing budget and at most four rank-1 edits, found fixes in about two hours per run on a single consumer GPU (GTX 1660 SUPER, 6 GB). The edits raise accuracy on locked GSM8K test problems from 15.5% to 61.5–68% (two locked sets of 200, the second with every arm and prediction registered before opening), and six runs of the same protocol all reached 58–67% on practice problems. Independent agents found the same “Wait”-brake neurons and the same </think> lever. The edit that GPT-6 Astra wrote in Sparks of Weight-Space Programming (Takehi et al., MIT CSAIL, 2026) reaches the same accuracy only with a 20,480-token budget, so the agents’ advantage over it is efficiency: 1.7–1.9× fewer tokens. The agents’ edits also beat a LoRA fine-tuning baseline that used 13× more GPU time. Simple alternatives are strong: over the 400 locked problems, switching thinking off is only 6.5–9 points behind the edits at a third of their tokens, and ties the edits alone on MATH-500. A scripted search with the same tools but no agent reasoning failed, every cheap proxy score was eventually gamed, and more interpretability context hurt. On a diffuse defect (instruction following, IFEval), neither the agents, nor prompting, nor fine-tuning helped. Every prediction was written in a dated lab notebook before the result. The PDF is provided in English and in Brazilian Portuguese.
```
```
Português. Modelos de linguagem pequenos com raciocínio costumam falhar entrando em loop: com o raciocínio ligado e orçamento de 12.288 tokens, o Qwen3.5-0.8B fica sem espaço antes de responder em cerca de 85% dos problemas de matemática do ensino fundamental. Perguntamos se agentes de um modelo de fronteira conseguem consertar esse defeito editando os pesos diretamente, sem gradientes, e como o resultado se compara com correções mais simples. Agentes Claude, com os pesos, um pequeno orçamento de testes e no máximo quatro edições de posto 1, acharam correções em cerca de duas horas por execução numa única GPU de consumo (GTX 1660 SUPER, 6 GB). As edições levam o acerto em problemas trancados do GSM8K de 15,5% para 61,5–68% (dois conjuntos trancados de 200, o segundo com todos os braços e previsões registrados antes da abertura), e seis execuções do mesmo protocolo chegaram todas a 58–67% nos problemas de prática. Agentes independentes acharam os mesmos neurônios-freio de “Wait” e a mesma alavanca de </think>. A edição que o GPT-6 Astra escreveu em Sparks of Weight-Space Programming (Takehi et al., MIT CSAIL, 2026) só chega ao mesmo acerto com orçamento de 20.480 tokens; a vantagem dos agentes sobre ela é, portanto, de eficiência: 1,7–1,9× menos tokens. As edições dos agentes também superam um fine-tuning LoRA que usou 13× mais tempo de GPU. As alternativas simples são fortes: nos 400 problemas trancados, desligar o raciocínio fica só 6,5–9 pontos atrás das edições com um terço dos tokens, e empata com as edições sozinhas no MATH-500. Uma busca por script com as mesmas ferramentas, mas sem o raciocínio do agente, falhou; todo indicador barato acabou burlado; e mais contexto de interpretabilidade atrapalhou. Num defeito espalhado (seguir instruções, IFEval), nem os agentes, nem prompts, nem fine-tuning ajudaram. Toda previsão foi escrita num caderno de laboratório datado antes do resultado. O PDF está disponível em inglês e em português do Brasil.
```
- **Licenses:** `Creative Commons Attribution 4.0 International` (já vem selecionada).
- **Copyright:**
```
© 2026 Luiz Felipe Melo. Licensed under CC BY 4.0.
```

## Recommended information
- **Contributors:** nenhum. (O uso do Claude vai na nota abaixo, não como contribuidor.)
- **Keywords:** uma por vez:
```
weight editing, weight-space programming, rank-1 edits, LLM agents, mechanistic interpretability, reasoning loops, small language models, Qwen3.5, budget forcing, pre-registration
```
- **Subjects:** opcional. Se quiser, busque `Computer and information sciences`.
- **Languages:** `eng` (English) e `por` (Portuguese).
- **Dates:** em branco.
- **Version:**
```
v1 (pre-paper)
```
- **Publisher:** `Zenodo` (padrão).
- **Additional descriptions** (botão "Add description", tipo `Notes`):
```
Use of AI tools: the editing agents studied are Claude Opus 5.5, and Claude Fable 5.2 in two exploratory runs. Claude (Opus 5.5 and Fable 5.2, through Claude Code) was also used as a research assistant: it wrote most of the code, ran the experiments, kept the lab notebook, drafted the text and reviewed the claims against the raw data. The author designed the study, supervised and checked the work, and is responsible for the content. Languages: English (pre-paper.pdf) and Brazilian Portuguese (pre-paper.pt-BR.pdf).
```

## Funding
Em branco (o trabalho não teve financiamento).

## Related works
Adicione dois:
| relação | identificador | tipo |
|---|---|---|
| `Is supplemented by` | `https://github.com/luizmlo/agentic-weight-editing` | `Software` |
| `References` | `https://rikiyat.github.io/blog/weight-space-programming/` | `Publication / Other` |

Quando o registro do código existir, acrescente `Is supplemented by` com o DOI dele (gera uma versão só de metadados,
sem mudar o DOI do paper).

## References
Opcional, mas ajuda na indexação. Cole uma por linha:
```
ajaygrover (2026). weight-space-programming-stacking. GitHub repository, https://github.com/ajaygrover/weight-space-programming-stacking. Accessed Nov. 10, 2026.
Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024. arXiv:2406.11717.
Chen, X., Xu, J., Liang, T., He, Z., et al. (2025). Do NOT Think That Much for 2+3=? On the Overthinking of Long Reasoning Models. ICML 2025, PMLR 267. arXiv:2412.21187.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.
exp8-loop-controls (2026). Pre-registered controls for Experiment 8 of Sparks of Weight-Space Programming. GitHub repository, https://github.com/1099271/exp8-loop-controls. Accessed Nov. 10, 2026.
Gao, L., Schulman, J., Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
Gerganov, G., and contributors (2023–). llama.cpp. Software. https://github.com/ggml-org/llama.cpp.
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., Steinhardt, J. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2103.03874.
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685.
Hu, J., Tang, C., Li, W., Zhu, H., et al. (2026). Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations. arXiv:2611.09382.
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., et al. (2024). Let's Verify Step by Step. ICLR 2024. arXiv:2305.20050.
MatinKhajavi (2026). Weight-Space-Programming: a replication with Claude Opus 5.5. GitHub repository, https://github.com/MatinKhajavi/Weight-Space-Programming. Accessed Nov. 10, 2026.
Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022. arXiv:2202.05262.
Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., Bau, D. (2023). Mass-Editing Memory in a Transformer. ICLR 2023. arXiv:2210.07229.
Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., et al. (2025). s1: Simple test-time scaling. EMNLP 2025. arXiv:2501.19393.
nathanrs (2026). weight-space-lp. GitHub repository, https://github.com/nathanrs/weight-space-lp. Accessed Nov. 10, 2026.
Qwen Team (2026). Qwen3.5: Towards Native Multimodal Agents. Blog post, https://qwen.ai/blog?id=qwen3.5. Model: https://huggingface.co/Qwen/Qwen3.5-0.8B.
Rott Shaham, T., Schwettmann, S., Wang, F., Rajaram, A., Hernandez, E., Andreas, J., Torralba, A. (2024). A Multimodal Automated Interpretability Agent. ICML 2024. arXiv:2404.14394.
Skalse, J., Howe, N. H. R., Krasheninnikov, D., Krueger, D. (2022). Defining and Characterizing Reward Gaming. NeurIPS 2022. arXiv:2209.13085.
Takehi, R., Teoh, J., Zhang, A., Khattab, O. (2026). Sparks of Weight-Space Programming: LLMs Can Natively Read & Write Neural Network Weights. Blog post, MIT CSAIL, Nov. 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/. Logs: https://github.com/RikiyaT/lm_understand-logs.
tdotlab (2026). weight-space-programming: a home replication with a Claude editor. GitHub repository, https://github.com/tdotlab/weight-space-programming. Accessed Nov. 10, 2026.
Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., MacDiarmid, M. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248.
von Csefalvay, C., Doma, P., Unni Krishnan, A., Földi, T. (2026). IWIBench: Evaluating Gradient-Free Model Driven Intra-weight Injection. arXiv:2611.08647.
Wang, C., Feng, Y., Chen, D., Chu, Z., Krishna, R., Zhou, T. (2025). Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency. Findings of EMNLP 2025. arXiv:2506.08343.
WKJBryan (2026). wsp-lab. GitHub repository, https://github.com/WKJBryan/wsp-lab. Accessed Nov. 10, 2026.
Yang, S., Kautz, J., Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
Zhang, Y. (2026). Rethinking Weight-Space Programming. Blog post and repository, https://github.com/ZhangYiqun018/rethinking-weight-space-programming. Accessed Nov. 10, 2026.
Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., Hou, L. (2023). Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911.
Zou, A., Phan, L., Chen, S., Campbell, J., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
```

## Software, Publishing information, Thesis, Conference
Tudo em branco: é um preprint, não software, artigo de revista, tese ou anais.

## Visibility
`Public`, sem embargo.

## Antes de clicar em Publish
- O paper diz que o código está em `github.com/luizmlo/agentic-weight-editing`. Com o repositório privado, esse link dá
  404. Publique o paper quando o repositório for aberto, ou no mesmo dia.
- Use **Save draft** e **Preview** para conferir. O DOI só é registrado ao publicar.
