# Plano Experimental - M2 + MNIST + CW

## Objetivo

Reproduzir as linhas 9-13 e 19 da Tabela 10 do artigo DeepDetector
(M2 + MNIST + CW).

## Escopo

Apenas:

- CW L2 kappa=0.0
- CW L2 kappa=0.5
- CW L2 kappa=1.0
- CW L2 kappa=2.0
- CW L2 kappa=4.0
- CW Linf

## Dataset

MNIST, ultimos 1000 digitos: indices 9000-9999.

## Artigo-base

Tabela 10: linhas CW L2 kappa={0.0,0.5,1.0,2.0,4.0} e CW Linf.

## Amostras

Por padrao os scripts usam `test_start=9000` e `test_end=10000`, isto e, indices
9000-9999 do conjunto de teste MNIST com 10000 amostras.

## Modelo

A arquitetura M2 exata da referencia [36] nao foi localizada; esta
implementacao usa uma arquitetura MNIST compativel com CW como aproximacao
experimental.

Foi encontrada uma arquitetura base usada em implementacoes de treino, e ela
foi alinhada em `src/deepdetector/models/mnist_m2.py`:

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

Essa implementacao e compativel com TensorFlow 1.x, Keras legado e CleverHans.

## Metricas

- #F
- TP
- FN
- FP
- RTP
- RTP%
- Recall
- Precision
- F1

## Ataques

- CW L2 com kappa = 0.0, 0.5, 1.0, 2.0, 4.0
- CW Linf

CW L2 e gerado por `src/deepdetector/attacks/cw_l2.py`, usando
`CarliniWagnerL2` quando disponivel na instalacao CleverHans compativel.

CW Linf foi preparado como interface em `src/deepdetector/attacks/cw_linf.py`.
Na stack declarada no `environment.yml` (`cleverhans==3.1.0`), nao ha uma API
CW Linf estavel equivalente ao CW L2 para este projeto TF1/Keras. Por isso o
script `scripts/mnist_m2_cw/generate_mnist_cw_linf.py` registra `status="not_executed"` e
um motivo claro em vez de inventar resultados.

## Saidas esperadas

- adversariais salvos
- metricas do ataque
- metricas do detector
- comparacao com a Tabela 10

Os artefatos ficam isolados em `results/mnist/m2_cw/` para nao sobrescrever o
fluxo M1 + FGSM.

## Limitacoes

- A arquitetura M2 segue a base encontrada, mas ainda pode haver divergencias
  com a referencia [36].
- CW L2 e computacionalmente caro; a execucao completa com 1000 amostras e cinco
  kappas pode demorar bastante.
- CW Linf nao foi implementado nesta etapa por falta de API compativel na stack
  CleverHans legada do projeto.
- As metricas so devem ser comparadas com a Tabela 10 depois de treinar/restaurar
  M2, gerar adversariais CW L2 e rodar o detector.
