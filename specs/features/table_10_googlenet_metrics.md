# Spec - Table 10 GoogLeNet com metricas reais no fluxo oficial

## Status

Proposto.

## Objetivo

Garantir que `python scripts/run_experiment.py --experiment table_10_googlenet`
produza metricas reais para a linha 7 (DeepFool/GoogLeNet) usando o runner
oficial da Table 10, sem criar runners ou scripts paralelos.

## Contexto

- O ataque DeepFool ja foi implementado e registrado no registry.
- A configuracao da linha 7 ja possui parametros explicitos.
- O runner da Table 10 hoje apenas materializa linhas quando a row ja contem
  `metrics`, deixando qualquer `planned` como linha vazia.
- O conceito de `implemented` precisa significar "gera metricas reais no CLI
  oficial", e nao apenas "o ataque existe".

## Regras de negocio

1. Apenas rows com `status: implemented` devem gerar metricas reais.
2. Rows `planned` ou `blocked` devem continuar vazias (campos `null`/vazios).
3. `num_failures` conta falhas do ataque quando `adv_pred == clean_pred`.
4. A avaliacao do detector usa o filtro oficial ja existente no projeto.
5. O calculo de metricas deve ser zero-safe (sem divisao por zero).
6. Clean errors (clean_pred != true_label) devem ser descartados do calculo de
   TP/FN/FP/RTP e nao contam como falhas do ataque.

## Requisitos funcionais

1. Implementar um avaliador generico de rows da Table 10 para
   ImageNet/GoogLeNet que:
   - carrega amostras ImageNet a partir de config;
   - instancia o wrapper real do GoogLeNet;
   - valida que o wrapper oferece `predict_preprocessed_batch` ou `predict_batch`
     e `gradient(image, class_id)`;
   - gera `x_adv` com `generate_attack(attack.name, ...)` usando os parametros da row;
   - calcula `clean_pred`, `adv_pred` e `num_failures`;
   - aplica o detector/filtro existente em `x_clean` e `x_adv`;
   - agrega `tp`, `fn`, `fp`, `rtp`, `rtp_percent`, `recall`, `precision`, `f1`.

2. O avaliador deve usar helpers existentes de metricas, preferencialmente:
   - `deepdetector.evaluation.detector_metrics.compute_detector_counts`
   - `deepdetector.evaluation.detector_metrics.compute_precision_recall`

3. O filtro oficial deve ser o mesmo usado nas reproducoes (por padrao,
   `proposed_detection_filter` via `filters.factory`). Se a configuracao
   fornecer um filtro explicito, ele deve ser usado.

4. `run_table_10_group` deve despachar rows `implemented` para o avaliador
   sem criar runner especifico de DeepFool.

5. `table_10_googlenet` deve incluir configuracao de dataset e avaliacao
   suficiente para carregar amostras, por exemplo:

   - `dataset.index_csv` (CSV com `filename,label_index`)
   - `dataset.images_dir`
   - `dataset.image_size` (default 224)
   - `evaluation.n_samples` (limite de amostras)
   - `evaluation.seed` (para shuffle deterministico)

6. A linha 7 deve ser marcada como `implemented` somente quando o
   avaliador estiver funcional.

7. O output deve continuar restrito a `metrics.csv` e `metrics.json`.

## Requisitos nao funcionais

- Nenhum script novo em `scripts/`.
- Nenhuma dependencia nova.
- Sem escrita de outputs adicionais (`manifest.json`, `.md`).
- Respeitar limites de amostras para evitar runtime excessivo.
- Reaproveitar o maximo possivel de loaders, filtros e helpers existentes.

## Criterios de aceitacao

1. `python scripts/run_experiment.py --experiment table_10_googlenet` gera
   `metrics.csv` e `metrics.json` com valores reais na linha 7 quando
   `status: implemented`.
2. Linhas 5 e 6 (status `planned`) continuam com campos metricos vazios.
3. `num_failures` corresponde ao numero de pares em que
   `adv_pred == clean_pred` (apenas para clean-correct).
4. `rtp` e `rtp_percent` sao derivados de `TTP` e `tp` conforme helpers
   existentes, com divisao segura.
5. O runner da Table 10 nao cria arquivos adicionais alem de CSV/JSON.
6. Smoke test com `evaluation.n_samples: 2` executa o avaliador e retorna
   um dict de metricas com chaves completas.

## Casos de erro

- Dataset sem `index_csv` ou `images_dir` deve falhar com `ValueError`.
- Modelo sem `gradient` ou sem scores/previsao deve falhar com
  `NotImplementedError`.
- Ataque nao registrado deve falhar com `ValueError`.
- Dataset vazio deve falhar com `ValueError`.
- Shapes inconsistentes entre imagens devem falhar com `ValueError`.

## Fora de escopo

- Implementar metricas para FGSM (rows 5 e 6).
- Implementar CaffeNet ou Inception v3.
- Criar runners especificos para DeepFool.
- Alterar a matematica do filtro ou do ataque.
- Baixar datasets ou pesos de modelo automaticamente.
