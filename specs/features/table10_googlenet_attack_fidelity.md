# Spec - Ajustar fidelidade dos ataques da Table 10 GoogLeNet

## Status

Proposto.

## Objetivo

Corrigir o experimento `table_10_googlenet` para reproduzir com maior
fidelidade as linhas 5, 6 e 7 da Table 10 do artigo base e do repositorio
original `OwenSec/DeepDetector`.

O ajuste cobre dois pontos:

- DeepFool/GoogLeNet deve operar no espaco Caffe `CHW/BGR/[0,255]`, sem
  clipar adversariais para `[0,1]`.
- FGSM/GoogLeNet deve calcular gradiente pelo grafo original com softmax
  (`prob`), e nao pelo prototxt sem softmax usado por DeepFool.

## Contexto

Os resultados atuais de `results/table_10/imagenet/googlenet/metrics.json`
mostram colapso estrutural na linha 7:

```text
DeepFool/GoogLeNet atual: #F=0, TP=4, FN=1203, FP=103, F1=0.61
DeepFool/GoogLeNet artigo: #F=405, TP=725, FN=42, FP=53, F1=93.85
```

O diagnostico indica que a imagem limpa carregada para GoogLeNet e convertida
para o espaco Caffe por `model.preprocess`, resultando em tensor
`CHW/BGR/[0,255]`. A row 7, porem, esta configurada com:

```yaml
clip_min: 0.0
clip_max: 1.0
```

Com isso, DeepFool gera adversariais em uma escala e depois os limita a outra,
destruindo a imagem no espaco usado por Caffe.

O repositorio original tambem separa os grafos por ataque:

- FGSM usa o modelo GoogLeNet original e faz backprop a partir do blob `prob`.
- DeepFool usa `deploy_original.prototxt` para predicao e
  `deploy_removeSoftmax.prototxt` para gerar perturbacoes.

Referencias:

- `https://github.com/OwenSec/DeepDetector`
- `https://raw.githubusercontent.com/OwenSec/DeepDetector/master/Test/FGSM/Test_FGSM_ImageNet.py`
- `https://raw.githubusercontent.com/OwenSec/DeepDetector/master/Test/DeepFool/GoogLeNet/Test_Deepfool_ImageNet_GoogLeNet.m`

## Business rules

1. GoogLeNet Table 10 usa imagens pre-processadas em espaco Caffe
   `CHW/BGR/[0,255]` para FGSM, DeepFool, predicao e filtro.
2. DeepFool/GoogLeNet deve clipar adversariais em `[0,255]`.
3. DeepFool/GoogLeNet deve usar `deploy_original.prototxt` para predicao e
   deteccao.
4. DeepFool/GoogLeNet deve usar `deploy_removeSoftmax.prototxt` somente para
   os gradientes do ataque.
5. FGSM/GoogLeNet deve usar o grafo original com softmax e gradiente pelo blob
   `prob`, conforme o script original de FGSM ImageNet.
6. O prototxt sem softmax nao pode afetar as rows 5 e 6.
7. `epsilon` das rows FGSM continua representando `1/255` e `2/255`, mas a
   geracao deve aplicar passos de `1.0` e `2.0` no espaco `[0,255]`.
8. Clean errors continuam descartados antes de contabilizar falhas de ataque,
   TP, FN, FP e RTP.
9. Falha de ataque continua sendo `adv_pred == clean_pred` para amostras
   clean-correct.
10. O output oficial do experimento continua restrito a `metrics.csv` e
    `metrics.json`.

## Functional requirements

1. Atualizar `table_10_googlenet` para que a row 7 tenha:

   ```yaml
   attack:
     name: deepfool
     max_iter: 50
     overshoot: 0.02
     clip_min: 0.0
     clip_max: 255.0
     num_classes: 10
   ```

2. Separar a configuracao de modelo usada por FGSM e DeepFool.

   Implementacao aceitavel:

   - manter `model.deploy_proto` apontando para
     `artifacts/models/imagenet/googlenet/deploy_original.prototxt`;
   - mover ou aplicar `attack_deploy_proto` apenas quando a row executada for
     `attack.name: deepfool`;
   - garantir que rows `fgsm` instanciem o wrapper sem
     `attack_deploy_prototxt`, ou chamem explicitamente o gradiente da rede de
     predicao original.

3. FGSM rows 5 e 6 devem chamar o helper Caffe-scale ja existente, usando:

   - `class_id=clean_pred`;
   - `epsilon_255=1.0` para row 5;
   - `epsilon_255=2.0` para row 6;
   - `clip_min=0.0`;
   - `clip_max=255.0`;
   - gradiente vindo da rede original com blob `prob`.

4. DeepFool row 7 deve chamar o dispatcher comum de ataques com:

   - imagem limpa em espaco Caffe;
   - `labels=[true_label]`;
   - `clip_min=0.0`;
   - `clip_max=255.0`;
   - gradiente vindo da rede sem softmax.

5. O wrapper Caffe deve expor uma forma clara de selecionar a rede de
   gradiente por ataque.

   A API exata fica a criterio da implementacao, mas os testes devem conseguir
   distinguir:

   - FGSM usou a rede original;
   - DeepFool usou a rede `attack_deploy_proto`.

6. A configuracao ou o avaliador deve falhar explicitamente se DeepFool for
   executado em imagem Caffe-scale (`max(clean_image) > 1.0`) com
   `clip_max <= 1.0`.

7. Adicionar um modo de diagnostico leve para execucoes reduzidas, sem alterar
   o output oficial.

   Quando configurado para uma amostra pequena, o logger deve emitir por
   amostra:

   ```text
   sample_index
   true_label
   clean_pred
   adv_pred
   attack_failed
   clean_min
   clean_max
   adv_min
   adv_max
   linf_delta
   entropy_clean
   entropy_adv
   filtered_adv_pred
   detected
   corrected
   ```

   Esse diagnostico nao deve criar arquivo novo por padrao.

## Non-functional requirements

- Nao criar scripts novos em `scripts/`.
- Nao criar runner exclusivo para GoogLeNet, FGSM ou DeepFool.
- Nao duplicar logica de metricas, filtros ou escrita de resultados.
- Nao alterar a matematica do filtro proposto.
- Nao baixar datasets, pesos ou prototxts automaticamente.
- Manter os testes pequenos e deterministas; testes com Caffe real devem ser
  evitados quando o mesmo contrato puder ser verificado com dummies/mocks.
- Preservar alteracoes existentes do usuario no working tree.

## Acceptance criteria

1. `configs/experiments.yaml` configura `table_10_googlenet` row 7 com
   `clip_min: 0.0` e `clip_max: 255.0`.
2. Um teste de config garante que rows 5 e 6 de FGSM nao usam
   `deploy_removeSoftmax.prototxt` para gradiente.
3. Um teste de config ou builder garante que row 7 de DeepFool usa
   `deploy_removeSoftmax.prototxt` para gradiente.
4. Um teste unitario garante que `_generate_table_10_adversarial` passa
   `epsilon_255=1.0`, `clip_min=0.0` e `clip_max=255.0` para a row 5.
5. Um teste unitario garante que `_generate_table_10_adversarial` passa
   `epsilon_255=2.0`, `clip_min=0.0` e `clip_max=255.0` para a row 6.
6. Um teste unitario garante que a row 7 chama `generate_attack("deepfool", ...)`
   com `clip_max=255.0`.
7. Um teste de regressao garante que DeepFool em imagem com `max > 1.0` e
   `clip_max <= 1.0` falha com mensagem clara, em vez de gerar metricas
   silenciosamente ruins.
8. Um teste do wrapper Caffe com dummies garante que `gradient` usa a rede
   original quando o ataque e FGSM.
9. Um teste do wrapper Caffe com dummies garante que `gradient` usa a rede
   `attack_deploy_proto` quando o ataque e DeepFool.
10. `python scripts/run_experiment.py --experiment table_10_googlenet` continua
    sendo o unico ponto de entrada oficial.
11. O experimento continua escrevendo apenas:

    ```text
    results/table_10/imagenet/googlenet/metrics.csv
    results/table_10/imagenet/googlenet/metrics.json
    ```

12. Em uma execucao reduzida de diagnostico, os logs incluem `clean_min/max`,
    `adv_min/max` e `linf_delta` para confirmar que DeepFool deixou de clipar
    imagens Caffe-scale para `[0,1]`.

## Error cases

- Se DeepFool receber imagem Caffe-scale com `clip_max <= 1.0`, falhar com
  `ValueError`.
- Se FGSM tentar usar `deploy_removeSoftmax.prototxt` para gradiente, falhar em
  teste de regressao.
- Se DeepFool estiver configurado sem `attack_deploy_proto`, falhar com
  `ValueError` no grupo GoogLeNet da Table 10.
- Se `clip_min >= clip_max`, manter a falha existente do ataque.
- Se a imagem adversarial tiver shape diferente da imagem limpa, falhar com
  `ValueError`.

## Out of scope

- Recalibrar thresholds do filtro.
- Alterar formulas de TP, FN, FP, RTP, Recall, Precision ou F1.
- Reprocessar ou versionar resultados finais dentro desta spec.
- Criar comparacao estatistica automatica com os valores do artigo.
- Alterar CaffeNet ou Inception v3.
- Baixar ou substituir pesos/prototxts do modelo.
