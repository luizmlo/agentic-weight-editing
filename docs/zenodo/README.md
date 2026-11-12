# Publicação no Zenodo — passo a passo

O trabalho vai em dois registros ligados entre si:

| registro | conteúdo | licença | metadados |
|---|---|---|---|
| **paper** | `pre-paper.pdf` (inglês) + `pre-paper.pt-BR.pdf` | CC BY 4.0 | `docs/zenodo/paper.json` |
| **código e dados** | o repositório (código, CADERNO, edições, resumos dos resultados) | Apache-2.0 (código) + CC BY 4.0 (texto e dados) | `.zenodo.json` na raiz |

O paper é o DOI que todos citam. O código tem DOI próprio para que cada versão dos números fique ligada ao código
exato que a gerou.

## 1. Conta
Entre em zenodo.org com o ORCID, que já liga o seu ID à conta. Depois ligue também o GitHub, em
Settings → Linked accounts.

## 2. Registro do paper (dá para fazer com o repositório ainda privado)
1. Abra **New upload** e escolha **Reserve DOI**. O Zenodo mostra o DOI antes de publicar.
2. Coloque esse DOI nos lugares que ainda têm `XXXXXXX`:
   - em `CITATION.cff`;
   - em `.zenodo.json`;
   - em `docs/hf/Qwen3.5-0.8B-LoopFix.md` (o `<zenodo-doi>`);
   - no rodapé do paper, se quiser.
3. Gere os PDFs de novo (`scripts/build_prepaper_pdf.py`, as quatro variantes) e envie os dois de tela.
4. Preencha o formulário com os valores de `paper.json`:
   - **tipo:** Publication / Preprint;
   - **autor:** com afiliação e ORCID;
   - **descrição:** o resumo em inglês e em português;
   - **palavras-chave;**
   - **licença:** CC BY 4.0;
   - **idiomas:** English e Portuguese;
   - **notas:** o uso de IA;
   - **relações:** o repositório e o *Sparks*.
5. Publique. Em até um dia o DataCite avisa o ORCID e o trabalho aparece no seu perfil para aprovar.

## 3. Registro do código
Publicar este registro torna o código público. Por isso ele espera a sua decisão de abrir o repositório.
- **Caminho normal:** com o repositório público, ative-o em Zenodo → GitHub e crie uma *release* (`v1.0`). O
  Zenodo lê `.zenodo.json` e cria o registro com DOI. Cada release nova vira uma versão.
- **Caminho alternativo:** um upload manual de um zip do repositório, como *Software*, com os mesmos metadados.
- **Depois:** volte ao registro do paper e troque `10.5281/zenodo.XXXXXXX` pelo DOI do código, na relação
  *is supplemented by*. Isso cria uma nova versão só de metadados, e o DOI do paper não muda.

## 4. Versões
Revisões futuras do paper entram com **New version** no mesmo registro. O DOI "de conceito" aponta sempre para a
versão mais recente, e cada versão também tem um DOI próprio.
