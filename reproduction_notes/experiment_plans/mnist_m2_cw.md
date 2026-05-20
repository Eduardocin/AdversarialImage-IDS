# Plano Experimental - M2 + MNIST + CW

## Objetivo

Executar o fluxo M2 + MNIST + CW e comparar as métricas obtidas com os valores de referência quando eles estiverem configurados para o cenario CW/M2.

## Escopo

- Treinamento/restauração do baseline MNIST M2
- CW L2 com kappa = 0.0, 0.5, 1.0, 2.0, 4.0
- CW Linf com implementacao TensorFlow 1.x local
- Avaliação do detector por mudança de predição após filtro final
- Isolamento dos artefatos em `results/mnist/m2_cw/`

## Dataset

MNIST, com recortes diferentes entre os scripts originais e o fluxo atual:

- scripts originais CW/MNIST: 1000 digitos a partir de `start=5500`
- fluxo atual M2/CW: ultimos 1000 digitos, indices 9000-9999

## Amostras

Por padrão os scripts atuais usam  `samples=1000` e `start_index=9000`, isto é,
indices 9000-9999 do conjunto de teste MNIST com 10000 amostras.

Nos scripts originais preservados em `src/original/DeepDetector/Test/CW/`,
`generate_data()` tem default `start=9000` ou `start=9500`, mas a chamada
executada nos arquivos MNIST usa explicitamente `start=5500` com `samples=1000`.

## Modelo

A implementação M2 atual está em `src/deepdetector/models/mnist_m2.py` e usa uma
CNN Keras/TensorFlow 1.x com saída em logits:

- Conv2D(32, 3x3, valid) + ReLU
- Conv2D(32, 3x3, valid) + ReLU
- MaxPooling2D(2x2)
- Conv2D(64, 3x3, valid) + ReLU
- Conv2D(64, 3x3, valid) + ReLU
- MaxPooling2D(2x2)
- Flatten
- Dense(200) + ReLU
- Dropout(0.5)
- Dense(200) + ReLU
- Dense(10) logits

O treinamento padrão usa:

- epochs = 10
- batch_size = 128
- learning_rate = 0.001
- label_smoothing = 0.1
- seed_tf = 1234
- seed_numpy = 20170830
- checkpoint em `results/mnist/m2_cw/clean_baseline/checkpoints`

## Métricas

- #F / F
- TP
- FN
- FP
- RTP
- RTP%
- Recall
- Precision
- F1
- n_clean_wrong
- n_attack_failed

## Ataques

- CW L2 com kappa = 0.0, 0.5, 1.0, 2.0, 4.0
- CW Linf

CW L2 e gerado por `src/deepdetector/attacks/cw_l2.py` e executado por `scripts/mnist/m2_cw/generate_attack_l2.py`.

Parâmetros padrão do CW L2 atual:

- batch_size = 32
- max_iterations = 1000
- learning_rate = 0.01
- binary_search_steps = 5
- clip_min = 0.0
- clip_max = 1.0

Nos scripts originais, `src/original/DeepDetector/Test/CW/Test_CWL2_MNIST.py` usa `CarliniL2` com:

- batch_size = 1
- max_iterations = 2000
- confidence = 0
- binary_search_steps = 5
- initial_const = 1.0
- learning_rate = 1e-1
- targeted = False

CW Linf e gerado por `src/deepdetector/attacks/cw_linf.py` e executado por `scripts/mnist/m2_cw/generate_attack_linf.py`.

Parâmetros padrão do CW Linf atual:

- batch_size = 32
- max_iterations = 1000
- learning_rate = 0.01
- confidence = 0.0
- initial_tau = 1.0
- const = 1.0
- tau_decay = 0.9
- tau_check_interval = 50
- clip_min = 0.0
- clip_max = 1.0

Nos scripts originais, `src/original/DeepDetector/Test/CW/Test_CWLi_MNIST.py` usa `CarliniLi` com `max_iterations=1000` e `targeted=False`.

## Cenários experimentais

### Fluxo operacional M2 + CW

Este fluxo treina/restaura M2, gera adversariais CW L2 e CW Linf, e avalia o detector usando os adversariais salvos.

```bash
python scripts/mnist/m2_cw/train.py --load-model
python scripts/mnist/m2_cw/generate_attack_l2.py --load-model --kappas 0.0,0.5,1.0,2.0,4.0 --samples 1000 --start-index 9000
python scripts/mnist/m2_cw/generate_attack_linf.py --load-model --samples 1000 --start-index 9000
python scripts/mnist/m2_cw/evaluate_detector.py --attack all --kappas 0.0,0.5,1.0,2.0,4.0 --samples 1000 --start-index 9000
```

O detector usa `--filter final` por padrão, isto é, `proposed_detection_filter` de `src/deepdetector/evaluation/article_reproduction.py`.
Também é possível executar com `--filter adaptive` ou `--filter scalar`.

### Execução parcial

Para executar somente CW L2:

```bash
python scripts/mnist/m2_cw/generate_attack_l2.py --load-model --kappas 0.0,0.5,1.0,2.0,4.0 --samples 1000 --start-index 9000
python scripts/mnist/m2_cw/evaluate_detector.py --attack cw_l2 --kappas 0.0,0.5,1.0,2.0,4.0 --samples 1000 --start-index 9000
```

Para executar somente CW Linf:

```bash
python scripts/mnist/m2_cw/generate_attack_linf.py --load-model --samples 1000 --start-index 9000
python scripts/mnist/m2_cw/evaluate_detector.py --attack cw_linf --samples 1000 --start-index 9000
```

### Scripts originais

Os scripts originais de referência ficam em:

- `src/original/DeepDetector/Test/CW/Test_CWL2_MNIST.py`
- `src/original/DeepDetector/Test/CW/Test_CWLi_MNIST.py`

Eles dependem do pacote externo `nn_robust_attacks`, usam modelo e dados do setup original de Carlini/Wagner e operam na escala de imagem `[-0.5, 0.5]`.
O fluxo atual usa os helpers locais do projeto e imagens MNIST em `[0.0, 1.0]`.

## Saídas esperadas

- checkpoint do baseline M2 limpo
- adversariais CW L2 salvos por kappa
- adversariais CW Linf salvos
- metricas de sucesso dos ataques
- metricas do detector
- logs de progresso para ataques longos

As saídas ficam em:

- `results/mnist/m2_cw/`

## Limitações

- Não foi localizado um script `article_reproduction` dedicado a valores fixos de referência para M2 + CW; a comparação precisa ser feita contra resultados
  originais executados ou contra uma tabela de referencia ainda a consolidar.
- O orquestrador `scripts/mnist/m2_cw/run_experiments.py` está desalinhado com os nomes atuais dos scripts: ele referência caminhos `scripts/mnist_m2_cw/*`
  que não existem neste workspace. Até ajustar o orquestrador, use os comandos individuais listados neste plano.
- O mesmo orquestrador usa `sys.executable`, mas nao importa `sys`.
- CW L2 e computacionalmente caro; a execucao completa com 1000 amostras e cinco kappas pode demorar bastante.
- CW Linf usa uma implementacao local porque a stack CleverHans 3.1.0 usada aqui não fornece uma API CW Linf equivalente ao CW L2.
- Os scripts originais CW/MNIST usam `start=5500`, enquanto o fluxo atual usa `start_index=9000` por padrão; essa diferença deve ser considerada em qualquer
  comparação direta.
- As métricas só devem ser comparadas depois de treinar/restaurar M2, gerar os adversariais CW e rodar o detector com o mesmo recorte de amostras.
