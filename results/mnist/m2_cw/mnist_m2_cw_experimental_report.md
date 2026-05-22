# Relatorio Experimental - M2 + MNIST + CW

## 1. Objetivo

Reproduzir as linhas 9-13 e 19 da Tabela 10 do artigo DeepDetector.

## 2. Escopo

Apenas CW L2 kappa=0.0,0.5,1.0,2.0,4.0 e CW Linf.

## 3. Dataset

MNIST, ultimos 1000 digitos: indices 9000-9999 do conjunto de teste.

## 4. Modelo M2

Foi criada uma implementacao separada em `src/deepdetector/models/mnist_m2.py`.
Ela usa uma CNN MNIST TF1/Keras compativel com CleverHans:

- Conv2D(32) + ReLU
- Conv2D(32) + ReLU
- MaxPooling2D
- Conv2D(64) + ReLU
- Conv2D(64) + ReLU
- MaxPooling2D
- Flatten
- Dense(200) + ReLU
- Dropout(0.5)
- Dense(200) + ReLU
- Dense(10) (logits)

A arquitetura exata da referencia [36] nao esta no repositorio. Portanto, este
M2 segue a base encontrada nas implementacoes de treino, como aproximacao
operacional para a reproducao, sem substituir o modelo M1 existente.

## 5. Ataques

CW L2 com kappa=0.0,0.5,1.0,2.0,4.0 foi implementado em
`src/deepdetector/attacks/cw_l2.py` e orquestrado por
`scripts/mnist_m2_cw/generate_mnist_cw_l2.py`.

CW Linf foi implementado localmente em `src/deepdetector/attacks/cw_linf.py`
com TensorFlow 1.x. A stack atual nao fornece Carlini-Wagner Linf via
CleverHans, entao essa etapa nao usa uma API pronta do CleverHans.

## 6. Detector

O script `scripts/mnist_m2_cw/evaluate_mnist_m2_cw_detector.py` usa a regra:

`C(x) != C(T(x)) => adversarial detectado`

Por padrao ele aplica o filtro final definido em
`deepdetector.evaluation.article_reproduction.proposed_detection_filter`.
Tambem ha flags para `scalar` e `adaptive`.

## 7. Resultados

Ainda nao ha resultados numericos neste relatorio parcial. Execute o pipeline
para preencher:

- `results/mnist/m2_cw/clean_baseline/summary.csv`
- `results/mnist/m2_cw/cw_l2/summary.csv`
- `results/mnist/m2_cw/detector/cw_l2_detector_results.csv`
- `results/mnist/m2_cw/article_comparison/table_10_m2_cw_comparison.csv`

## 8. Discussao

Diferencas provaveis em relacao ao artigo:

- arquitetura M2 aproximada;
- parametros CW e custo computacional;
- numero de iteracoes;
- versoes de TensorFlow, Keras e CleverHans;
- detalhes da implementacao do filtro final;
- CW Linf usa implementacao local TF1, o que pode divergir da implementacao
  original usada no artigo.

## 9. Conclusao Parcial

O fluxo M2 + MNIST + CW esta estruturado para execucao e comparacao com a Tabela
10. A proximidade numerica com o artigo so deve ser avaliada depois da execucao
do pipeline completo.
