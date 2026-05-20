# Plano Experimental - M1 + MNIST + FGSM

## Objetivo

Executar o fluxo M1 + MNIST + FGSM e comparar as métricas obtidas com os valores de referência configurados nos scripts.

## Escopo
- Treinamento/restauração do baseline MNIST M1
- FGSM operacional com epsilon = 0.2
- Reprodução da comparação final com epsilons = 0.1, 0.2, 0.3, 0.4
- Avaliação do detector por mudança de predição após filtros
- Comparação consolidada dos filtros registrados

## Dataset
MNIST, com recortes diferentes conforme a etapa experimental:

- treino/seleção de filtros: indices 0-4499 do conjunto de teste
- validação do filtro adaptativo: indices 4500-5499
- teste final da Tabela 10: indices 5500-9999

## Amostras

Por padrão o fluxo operacional usa `samples=4500`, isto e, indices 0-4499 do conjunto de teste MNIST com 10000 amostras.

Para a reprodução da Tabela 10, o script `scripts/article_reproduction/table_10.py` usa `load_mnist_test_slice(5500,
10000)`, isto e, os ultimos 4500 digitos do conjunto de teste.

## Modelo

A implementacao M1 usa a CNN MNIST do CleverHans/Keras exposta por `src/deepdetector/models/mnist_cnn.py` via `cnn_model()`, com execução em TensorFlow 1.x, Keras legado e checkpoints `tf.train.Saver`.

O treinamento padrão usa:

- epochs = 6
- batch_size = 128
- learning_rate = 0.001
- label_smoothing = 0.1
- checkpoint em `results/mnist/clean_baseline/checkpoints`

## Métricas

- #F / n_discarded
- TP
- FN
- FP
- TN
- TTP / RTP
- TTP% / RTP%
- Recall
- Precision
- F1

## Ataque

- FGSM operacional com epsilon = 0.2
- FGSM da Tabela 10 com epsilons = 0.1, 0.2, 0.3, 0.4
- `clip_min=0.0`
- `clip_max=1.0`

FGSM e gerado por `src/deepdetector/attacks/fgsm.py` e executado pelo script `scripts/mnist/m1_fgsm/generate_attack.py`.

Nos scripts originais preservados em `src/original/DeepDetector/`, a separação observada é:

- `Train/Train_FGSM_MNIST.py`: FGSM `eps=0.2` sobre `X_test[:4500]`
- `Validation/Validate_MNIST.py`: FGSM `eps=0.2` sobre `X_test[4500:5500]`
- `Test/FGSM/Test_FGSM_MNIST.py`: FGSM `eps=0.3` sobre `X_test[5500:]`
- `scripts/article_reproduction/table_10.py`: reproduz a comparacao final para
  `eps=0.1,0.2,0.3,0.4` sobre `X_test[5500:10000]`

## Cenários experimentais

### Fluxo operacional M1 + FGSM

Este fluxo gera adversariais persistidos e avalia todos os filtros do registry.
Ele e útil para desenvolvimento local e comparação entre filtros.

```bash
python scripts/mnist/m1_fgsm/train.py --epochs 6 --batch-size 128 --learning-rate 0.001
python scripts/mnist/m1_fgsm/generate_attack.py --epsilons 0.2 --samples 4500 --load-model
python scripts/mnist/m1_fgsm/run_comparison.py --epsilon 0.2 --samples 4500
```

Para gerar todos os epsilons da comparação final como artefatos salvos:

```bash
python scripts/mnist/m1_fgsm/generate_attack.py --epsilons 0.1,0.2,0.3,0.4 --samples 4500 --load-model
```

### Reprodução da Tabela 10

Este fluxo compara diretamente com os valores de referencia do artigo para
FGSM/M1 em MNIST. Ele gera os adversariais em memoria para cada epsilon e aplica
o filtro final `proposed_detection_filter`.

```bash
python scripts/article_reproduction/table_10.py
```

Valores de referencia configurados:

| epsilon | #F | TP | FN | FP | RTP | RTP% | Recall | Precision | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 4026 | 410 | 40 | 24 | 398 | 97.07 | 91.11 | 94.47 | 92.76 |
| 0.2 | 1910 | 2467 | 106 | 32 | 2430 | 98.50 | 95.88 | 98.72 | 97.28 |
| 0.3 | 455 | 3856 | 172 | 32 | 3768 | 97.71 | 95.73 | 99.18 | 97.42 |
| 0.4 | 132 | 4078 | 273 | 32 | 3820 | 93.67 | 93.73 | 99.22 | 96.40 |

## Filtros avaliados

O fluxo consolidado avalia todos os filtros registrados em
`src/deepdetector/filters/registry.py`.

O script dedicado `scripts/mnist/m1_fgsm/evaluate_detector.py` tambem mantem a
avaliacao dos filtros de quantizacao usados no fluxo original:

- scalar_interval_128
- scalar_interval_64
- scalar_interval_43
- nonuniform_quantization

## Ordem de execucao

Execute os scripts em ordem a partir da raiz do repositorio:

```bash
python scripts/mnist/m1_fgsm/train.py --epochs 6 --batch-size 128 --learning-rate 0.001
python scripts/mnist/m1_fgsm/generate_attack.py --epsilons 0.2 --samples 4500 --load-model
python scripts/mnist/m1_fgsm/run_comparison.py --epsilon 0.2 --samples 4500
```

Quando ja existir checkpoint treinado, o primeiro passo pode ser executado com
`--load-model` para restaurar o baseline.

Para a comparacao com os diferentes epsilons da Tabela 10, execute:

```bash
python scripts/article_reproduction/table_10.py
```

## Saídas esperadas

- checkpoint do baseline limpo
- adversariais FGSM salvos
- metricas do ataque
- metricas do detector
- comparacao consolidada dos filtros
- comparacao com valores de referencia

As principais saidas ficam em:

- `results/mnist/clean_baseline/`
- `results/mnist/fgsm/eps_0p2/`
- `results/mnist/final_mnist_results.csv`
- `results/mnist/final_mnist_report.md`
- `results/mnist/article_reproduction/table_10_mnist_fgsm_test.csv`
- `results/mnist/article_reproduction/table_10_mnist_fgsm_test.md`

## Limitações

- O ambiente e legado: TensorFlow 1.x, Keras standalone e CleverHans 3.1.0.
- A avaliação descarta pares em que a imagem limpa já é classificada incorretamente e pares em que o FGSM não muda a classe verdadeira.
- O contador da faixa `mid` na avaliacao por entropia tem um bug documentado em `reproduction_notes/mnist_reproduction_notes.md`; esse comportamento permanece sem correção nesta etapa.
- O script original de teste FGSM/MNIST fixa `eps=0.3`, mas a reprodução da tabela do artigo usa a varredura `0.1, 0.2, 0.3, 0.4`.
- O fluxo M1 + FGSM usa `results/mnist/` diretamente, enquanto o fluxo M2 + CW fica isolado em `results/mnist/m2_cw/`.
