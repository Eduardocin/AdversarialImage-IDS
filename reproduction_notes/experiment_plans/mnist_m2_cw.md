# Plano Experimental - M2 + MNIST + CW

## Objetivo

Executar o fluxo M2 + MNIST + CW e comparar as metricas obtidas com os valores
de referencia configurados nos scripts.

## Escopo

- CW L2 kappa=0.0
- CW L2 kappa=0.5
- CW L2 kappa=1.0
- CW L2 kappa=2.0
- CW L2 kappa=4.0
- CW Linf

## Dataset

MNIST, ultimos 1000 digitos: indices 9000-9999.

## Amostras

Por padrao os scripts usam `test_start=9000` e `test_end=10000`, isto e, indices
9000-9999 do conjunto de teste MNIST com 10000 amostras.

## Modelo

A arquitetura M2 exata da referencia [36] nao foi localizada no projeto. Esta
implementacao usa uma CNN MNIST compativel com TensorFlow 1.x, Keras legado e
ataques CW:

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

CW L2 e gerado por `src/deepdetector/attacks/cw_l2.py`.

CW Linf e gerado por `src/deepdetector/attacks/cw_linf.py` com uma
implementacao TensorFlow 1.x local que otimiza uma margem untargeted e reduz um
limiar Linf (`tau`) ao longo das iteracoes.

## Saidas esperadas

- adversariais salvos
- metricas do ataque
- metricas do detector
- comparacao com valores de referencia

Os artefatos ficam isolados em `results/mnist/m2_cw/` para nao sobrescrever o
fluxo M1 + FGSM.

## Limitacoes

- A arquitetura M2 pode divergir da referencia [36].
- CW L2 e computacionalmente caro; a execucao completa com 1000 amostras e cinco
  kappas pode demorar bastante.
- CW Linf usa uma implementacao local porque a stack CleverHans 3.1.0 usada aqui
  nao fornece uma API CW Linf equivalente ao CW L2.
- As metricas so devem ser comparadas depois de treinar/restaurar M2, gerar os
  adversariais CW e rodar o detector.
