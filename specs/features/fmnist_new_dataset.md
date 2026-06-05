# SPEC — Fashion-MNIST New Dataset Evaluation with M3 and M2

## Objective

Cumprir o requisito do projeto de avaliar o sistema reproduzido em **outro conjunto de dados**, executando duas combinações inspiradas na Table 10 sobre o **Fashion-MNIST**, um dataset grayscale `28x28` compatível em formato com o fluxo MNIST do projeto.

Os experimentos devem ser:

| No. | Attack/Model       | Dataset       |
| --: | ------------------ | ------------- |
|   1 | `FGSM (ε=0.2)/M3`  | Fashion-MNIST |
|   9 | `CW L2 (κ=0.0)/M2` | Fashion-MNIST |

Cada combinação deve ter uma execução pública separada via runner centralizado:

```bash
python scripts/run_experiment.py --experiment fashion_mnist_fgsm_m3
python scripts/run_experiment.py --experiment fashion_mnist_cw_l2_m2
```

Estes experimentos **não substituem** a reprodução oficial da Table 10. Eles formam uma extensão comparativa enxuta que reutiliza a mesma lógica de detecção, a mesma transformação adaptativa e a mesma forma de contagem, mudando a população avaliada para Fashion-MNIST.

A estrutura desta spec segue o mesmo padrão da extensão `imagenet_new_dataset`, que define novos experimentos como execuções públicas separadas e mantém o schema, a lógica de detecção e a forma de contagem da Table 10. 

---

## Scope

Este experimento deve:

* utilizar o Fashion-MNIST como novo dataset;
* validar que o dataset é `mnist_compatible`;
* usar imagens grayscale;
* usar imagens `28x28x1`;
* usar escala de valores compatível com os modelos M3 e M2;
* executar `FGSM (ε=0.2)/M3`;
* executar `CW L2 (κ=0.0)/M2`;
* aplicar a transformação adaptativa do DeepDetector;
* calcular métricas agregadas no schema oficial da Table 10;
* gerar somente os artefatos oficiais definidos nesta spec;
* usar exclusivamente o runner centralizado `scripts/run_experiment.py` para as execuções públicas de avaliação.

Este experimento não deve:

* usar ImageNet;
* usar `ambulance`, `scholar_bus`, `soccer_ball`, `cab`, `panda` ou `zebra`;
* executar GoogLeNet;
* executar CaffeNet;
* executar Inception v3;
* executar FGSM `ε=1/255`;
* executar DeepFool;
* executar CW L2/Inception v3;
* alterar a implementação oficial dos ataques;
* alterar a implementação oficial dos filtros;
* gerar diagnósticos públicos;
* gerar relatórios em Markdown como output do runner;
* criar scripts paralelos de experimento fora do runner central;
* misturar resultados de M3 e M2 no mesmo diretório de experimento.

---

## Background

Os modelos `M2` e `M3` pertencem ao fluxo MNIST-compatible do projeto. Por isso, eles esperam entradas compatíveis com MNIST:

```text
grayscale
28x28
1 canal
10 classes
escala de valores compatível com o treinamento do modelo
```

O Fashion-MNIST é compatível em **formato** com MNIST, pois também possui imagens grayscale `28x28` e 10 classes. Entretanto, ele não é semanticamente igual ao MNIST original: MNIST contém dígitos manuscritos, enquanto Fashion-MNIST contém itens de vestuário.

Portanto, para que a avaliação seja semanticamente válida, os modelos M3 e M2 devem ser treinados ou restaurados a partir de checkpoints compatíveis com Fashion-MNIST. Se forem usados checkpoints treinados em MNIST dígitos, o experimento deve ser explicitamente descrito como uma avaliação fora de domínio, não como uma avaliação plenamente compatível.

A spec anterior já indicava que o novo dataset deveria ser compatível com os modelos MNIST em formato, usando imagens grayscale `28x28` e escala esperada pelos modelos M2/M3.

O modelo `M1` atual não deve ser usado como classificador principal do experimento FGSM em Fashion-MNIST, pois sua arquitetura é considerada insuficiente para obter boa performance limpa nesse dataset. O experimento FGSM deve usar um novo modelo `M3`, definido em `src/deepdetector/models`, com arquitetura CNN simples, mais adequada ao Fashion-MNIST, mantendo entrada `28x28x1` e saída de 10 classes.

---

## Dataset Choice Justification

O Fashion-MNIST foi escolhido porque preserva as propriedades estruturais do fluxo MNIST:

| Propriedade                | Fashion-MNIST                   |
| -------------------------- | ------------------------------- |
| Resolução                  | `28x28`                         |
| Canais                     | 1 canal, grayscale              |
| Número de classes          | 10                              |
| Tipo de entrada            | imagem pequena, baixa resolução |
| Compatibilidade estrutural | compatível com M2/M3            |

A escolha permite avaliar se a estratégia de detecção baseada em redução adaptativa de ruído mantém comportamento consistente em um domínio visual diferente do MNIST original, mas sem exigir o pipeline ImageNet.

Diferente do MNIST original, que contém dígitos manuscritos, o Fashion-MNIST contém classes de vestuário. Isso torna o experimento útil para avaliar transferência da lógica de detecção para outro domínio visual de baixa resolução.

O arquivo local `data/fashion_mnist/fashion-mnist_test.csv` deve ser tratado como a fonte de dados desta extensão. A separação deve ser balanceada por classe:

| Uso | Total | Por classe |
| --- | ----: | ---------: |
| Treino dos checkpoints Fashion-MNIST M2/M3 | 9000 | 900 |
| Avaliação dos experimentos públicos | 1000 | 100 |

Os 1000 exemplos de avaliação não devem ser usados no treinamento dos checkpoints Fashion-MNIST.

---

## Fashion-MNIST Classes

O experimento deve usar as 10 classes oficiais do Fashion-MNIST:

| Índice | Classe        |
| -----: | ------------- |
|      0 | `t_shirt_top` |
|      1 | `trouser`     |
|      2 | `pullover`    |
|      3 | `dress`       |
|      4 | `coat`        |
|      5 | `sandal`      |
|      6 | `shirt`       |
|      7 | `sneaker`     |
|      8 | `bag`         |
|      9 | `ankle_boot`  |

---

## Experimental Design

Devem existir dois experimentos públicos separados.

### Experiment 1 — Fashion-MNIST FGSM/M3

```bash
python scripts/run_experiment.py --experiment fashion_mnist_fgsm_m3
```

Este experimento executa:

```text
FGSM (ε=0.2)/M3
```

sobre Fashion-MNIST.

### Experiment 2 — Fashion-MNIST CW L2/M2

```bash
python scripts/run_experiment.py --experiment fashion_mnist_cw_l2_m2
```

Este experimento executa:

```text
CW L2 (κ=0.0)/M2
```

sobre Fashion-MNIST.

---

## Detection Rule

A regra central de detecção deve permanecer igual à regra usada pelo DeepDetector:

```text
C(x_adv) != C(T(x_adv))
```

Onde:

* `C` é o classificador;
* `x_adv` é a imagem adversarial;
* `T` é a transformação adaptativa do DeepDetector;
* `T(x_adv)` é a imagem adversarial após redução de ruído.

Se a predição da imagem adversarial muda após a transformação `T`, o exemplo adversarial é considerado detectado.

A mesma ideia é usada na spec do novo dataset ImageNet, que preserva a regra `C(x_adv) != C(T(x_adv))` para a extensão experimental. 

---

## Counting Rules

Para cada imagem limpa corretamente classificada pelo modelo dentro da partição de avaliação:

1. Calcular a predição limpa:

```text
C(x)
```

2. Aplicar a transformação adaptativa na imagem limpa:

```text
T(x)
```

3. Calcular falso positivo:

```text
FP se C(x) != C(T(x))
TN interno se C(x) == C(T(x))
```

O contador `TN` pode ser usado internamente para validação, mas não deve ser exportado em `metrics.csv` porque não pertence ao schema oficial da Table 10.

4. Gerar a imagem adversarial:

```text
x_adv
```

5. Se o ataque não alterar a predição esperada, contar como:

```text
disturbed_failure
```

6. Caso o ataque tenha sucesso, calcular:

```text
C(x_adv)
C(T(x_adv))
```

7. Contar:

```text
TP se C(x_adv) != C(T(x_adv))
FN se C(x_adv) == C(T(x_adv))
```

As quotas Fashion-MNIST definem a quantidade de candidatos reservados por classe na partição de avaliação, não uma exigência de 100 acertos limpos por classe. Como a partição de avaliação possui exatamente 100 candidatos por classe, erros de classificação limpa devem ser descartados e registrados no `manifest.json`; eles não devem bloquear o experimento nem ser substituídos por amostras da partição de treino.

---

## Metrics

Cada experimento deve gerar uma linha agregada no schema oficial da Table 10.

Não devem existir métricas públicas por classe. As classes e quotas do Fashion-MNIST são usadas somente para seleção balanceada da população avaliada.

### Official Table 10 metrics

Para cada experimento:

```text
no
attack_model
dataset
num_failures
tp
fn
fp
rtp
rtp_percent
recall
precision
f1
```

### Formulas

```text
num_failures = quantidade de ataques que não alteram a predição limpa
tp = exemplos adversariais detectados
fn = exemplos adversariais bem-sucedidos não detectados
fp = imagens limpas cujo rótulo muda após a transformação T
rtp = exemplos adversariais detectados cuja predição filtrada volta ao rótulo verdadeiro
rtp_percent = rtp / tp
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = 2 * precision * recall / (precision + recall)
```

Se algum denominador for zero, o runner deve usar o comportamento zero-safe já adotado pelos helpers da Table 10. Linhas bloqueadas ou não executadas devem usar `null` no JSON e vazio no CSV.

---

## Configuration

A implementação deve adicionar dois experimentos ao `configs/experiments.yaml`.

### `fashion_mnist_fgsm_m3`

```yaml
fashion_mnist_fgsm_m3:
  kind: table_10_group
  description: Fashion-MNIST new dataset evaluation with FGSM epsilon 0.2 against M3
  output_dir: results/experiments/fashion_mnist/fgsm_m3

  dataset:
    name: fashion_mnist
    domain: mnist_compatible
    split: test
    csv_path: data/fashion_mnist/fashion-mnist_test.csv
    split_strategy:
      name: balanced_by_class
      train_samples: 9000
      evaluation_samples: 1000
      train_class_quota: 900
      evaluation_class_quota: 100
    image_shape: [28, 28, 1]
    value_range:
      min: 0.0
      max: 1.0
    require_clean_correct: true
    class_order:
      - t_shirt_top
      - trouser
      - pullover
      - dress
      - coat
      - sandal
      - shirt
      - sneaker
      - bag
      - ankle_boot
    class_indices:
      t_shirt_top: 0
      trouser: 1
      pullover: 2
      dress: 3
      coat: 4
      sandal: 5
      shirt: 6
      sneaker: 7
      bag: 8
      ankle_boot: 9
    class_quotas:
      t_shirt_top: 100
      trouser: 100
      pullover: 100
      dress: 100
      coat: 100
      sandal: 100
      shirt: 100
      sneaker: 100
      bag: 100
      ankle_boot: 100

  model:
    name: m3
    family: mnist
    checkpoint_dir: artifacts/models/fashion_mnist/m3/checkpoints
    dataset_name: fashion_mnist
    input_shape: [28, 28, 1]
    num_classes: 10

  checkpoint_training:
    source_csv: data/fashion_mnist/fashion-mnist_test.csv
    split_name: train
    selection:
      method: first_n_per_class
      per_class_start: 0
      per_class_end: 900
      class_quota: 900
      total_samples: 9000
    validation_sample:
      split_name: evaluation
      method: next_n_per_class
      per_class_start: 900
      per_class_end: 1000
      class_quota: 100
      total_samples: 1000
      excludes_training: true

  model_group: m3
  dataset_group: fashion_mnist
  dataset_label: Fashion-MNIST

  rows:
    - "no": 1
      attack_model: "FGSM (ε=0.2)/M3"
      status: implemented
      attack:
        name: fgsm
        epsilon: 0.2
        clip_min: 0.0
        clip_max: 1.0

  filter:
    name: proposed_detection_filter
    type: proposed_detection_filter

  output:
    manifest: true
```

### `fashion_mnist_cw_l2_m2`

```yaml
fashion_mnist_cw_l2_m2:
  kind: table_10_group
  description: Fashion-MNIST new dataset evaluation with CW L2 kappa 0.0 against M2
  output_dir: results/experiments/fashion_mnist/cw_l2_m2

  dataset:
    name: fashion_mnist
    domain: mnist_compatible
    split: test
    csv_path: data/fashion_mnist/fashion-mnist_test.csv
    split_strategy:
      name: balanced_by_class
      train_samples: 9000
      evaluation_samples: 1000
      train_class_quota: 900
      evaluation_class_quota: 100
    image_shape: [28, 28, 1]
    value_range:
      min: 0.0
      max: 1.0
    require_clean_correct: true
    class_order:
      - t_shirt_top
      - trouser
      - pullover
      - dress
      - coat
      - sandal
      - shirt
      - sneaker
      - bag
      - ankle_boot
    class_indices:
      t_shirt_top: 0
      trouser: 1
      pullover: 2
      dress: 3
      coat: 4
      sandal: 5
      shirt: 6
      sneaker: 7
      bag: 8
      ankle_boot: 9
    class_quotas:
      t_shirt_top: 100
      trouser: 100
      pullover: 100
      dress: 100
      coat: 100
      sandal: 100
      shirt: 100
      sneaker: 100
      bag: 100
      ankle_boot: 100

  model:
    name: m2
    family: mnist
    checkpoint_dir: artifacts/models/fashion_mnist/m2/checkpoints
    dataset_name: fashion_mnist
    input_shape: [28, 28, 1]
    num_classes: 10

  checkpoint_training:
    source_csv: data/fashion_mnist/fashion-mnist_test.csv
    split_name: train
    selection:
      method: first_n_per_class
      per_class_start: 0
      per_class_end: 900
      class_quota: 900
      total_samples: 9000
    validation_sample:
      split_name: evaluation
      method: next_n_per_class
      per_class_start: 900
      per_class_end: 1000
      class_quota: 100
      total_samples: 1000
      excludes_training: true

  model_group: m2
  dataset_group: fashion_mnist
  dataset_label: Fashion-MNIST

  rows:
    - "no": 9
      attack_model: "CW L2 (κ=0.0)/M2"
      status: implemented
      attack:
        name: cw_l2_nn_robust
        kappa: 0.0

  filter:
    name: proposed_detection_filter
    type: proposed_detection_filter

  output:
    manifest: true
```

---

## Dataset Loader Requirements

A implementação deve adicionar suporte ao Fashion-MNIST no loader de datasets MNIST-compatible.

O loader deve:

* carregar o Fashion-MNIST test split a partir de `data/fashion_mnist/fashion-mnist_test.csv`;
* falhar com erro claro se `dataset.csv_path` não existir;
* aplicar separação balanceada por classe a partir do CSV local;
* reservar 9000 exemplos para treinamento dos checkpoints Fashion-MNIST, com 900 exemplos por classe;
* selecionar a amostra de treino do checkpoint usando os primeiros 900 exemplos disponíveis de cada classe no CSV local;
* reservar 1000 exemplos para avaliação, com 100 exemplos por classe;
* selecionar a amostra de avaliação usando os próximos 100 exemplos disponíveis de cada classe no CSV local;
* garantir que os exemplos reservados para avaliação não sejam usados no treinamento;
* tratar `class_quotas` como quotas de candidatos de avaliação por classe;
* descartar da avaliação adversarial as imagens da partição de avaliação que não forem classificadas corretamente pelo checkpoint;
* registrar no manifesto os contadores `clean_errors`, `clean_correct`, `selected_clean_correct` e `selection_policy`;
* retornar imagens em shape `N x 28 x 28 x 1`;
* retornar labels inteiros de `0` a `9`;
* normalizar valores para `[0.0, 1.0]`;
* preservar a ordem de classes definida em `class_order`;
* permitir seleção por quota clean-correct;
* validar que `dataset.domain == "mnist_compatible"`.

O loader não deve:

* baixar o Fashion-MNIST automaticamente;
* converter Fashion-MNIST para RGB;
* redimensionar para ImageNet;
* aplicar preprocessing ImageNet;
* mapear classes Fashion-MNIST para classes MNIST dígitos.

---

## Model Requirements

Os modelos M3 e M2 usados neste experimento devem ser compatíveis com Fashion-MNIST.

O modelo M3 deve ser criado como um novo módulo em `src/deepdetector/models`, separado de `mnist_cnn.py` e `mnist_m2.py`. Ele deve ser a arquitetura padrão para o experimento `fashion_mnist_fgsm_m3`.

O M3 deve:

* aceitar tensores `28x28x1` em escala `[0.0, 1.0]`;
* produzir predições para exatamente 10 classes Fashion-MNIST;
* ser uma CNN simples e mais expressiva que o M1 atual para imagens Fashion-MNIST;
* usar blocos convolucionais com ativações não lineares, pooling e camadas densas finais;
* usar regularização moderada, evitando dropout excessivo que prejudique acurácia limpa no Fashion-MNIST;
* ser compatível com o fluxo TensorFlow 1.x/Keras legado usado pelos modelos MNIST do projeto;
* expor helpers de build, save, load e latest checkpoint consistentes com o padrão dos modelos existentes;
* usar checkpoints próprios em `artifacts/models/fashion_mnist/m3/checkpoints`;
* não substituir nem modificar o comportamento público do M1 usado nos experimentos MNIST oficiais.

Arquitetura mínima esperada para M3:

```text
Conv2D(32, 3x3) -> ReLU
Conv2D(32, 3x3) -> ReLU
MaxPool(2x2)
Dropout(0.25)
Conv2D(64, 3x3) -> ReLU
Conv2D(64, 3x3) -> ReLU
MaxPool(2x2)
Dropout(0.25)
Flatten
Dense(256) -> ReLU
Dropout(0.5)
Dense(10)
Softmax
```

A camada final deve expor `softmax` para compatibilidade com o `KerasModelWrapper` usado pelo FGSM/CleverHans; o ataque pode recuperar os logits pré-softmax a partir dessa camada. As taxas de dropout devem permanecer na configuração inicial `0.25/0.25/0.5`, que apresentou melhor comportamento agregado no detector Fashion-MNIST/FGSM, sem introduzir uma arquitetura pesada ou dependências novas.

O treino padrão inicial do checkpoint M3 deve usar 10 épocas, learning rate `0.001` e label smoothing `0.1`, salvo override explícito por CLI. Essa configuração corresponde ao baseline M3 inicial que apresentou melhor comportamento agregado no detector Fashion-MNIST/FGSM, sem alterar nomes ou shapes de variáveis do checkpoint.

Os checkpoints Fashion-MNIST de M3 e M2 devem ser treinados usando somente a partição balanceada de treino definida nesta spec:

```text
total_train = 9000
train_per_class = 900
per_class_start = 0
per_class_end = 900
```

A partição balanceada de avaliação deve ficar reservada para os experimentos públicos:

```text
total_evaluation = 1000
evaluation_per_class = 100
per_class_start = 900
per_class_end = 1000
```

Cada modelo deve declarar ou validar:

```text
dataset_name == fashion_mnist
input_shape == [28, 28, 1]
num_classes == 10
```

Se um checkpoint MNIST original for usado, a execução deve falhar ou registrar explicitamente que se trata de avaliação fora de domínio.

Recomendação de comportamento padrão:

```text
fail-fast se checkpoint_dir não existir
fail-fast se checkpoint não for compatível com Fashion-MNIST
fail-fast se número de classes do modelo for diferente de 10
```

### Checkpoint Training Helper

A implementação pode fornecer um script utilitário para materializar os checkpoints Fashion-MNIST antes da execução pública dos experimentos:

```bash
python scripts/train_fashion_mnist_checkpoint.py --model m3
python scripts/train_fashion_mnist_checkpoint.py --model m2
```

Esse script deve:

* ler `configs/experiments.yaml`;
* usar `fashion_mnist_fgsm_m3` como configuração padrão para `--model m3`;
* usar `fashion_mnist_cw_l2_m2` como configuração padrão para `--model m2`;
* exigir o bloco `checkpoint_training` no experimento selecionado;
* carregar `data/fashion_mnist/fashion-mnist_test.csv`;
* validar que `checkpoint_training.source_csv` corresponde a `dataset.csv_path`;
* usar somente os 9000 exemplos da partição balanceada de treino para treinar o checkpoint;
* validar que a amostra de treino configurada corresponde às posições `[0, 900)` dentro de cada classe;
* manter os 1000 exemplos da partição balanceada de avaliação apenas para validação de acurácia limpa;
* validar que a amostra de avaliação configurada corresponde às posições `[900, 1000)` dentro de cada classe;
* gravar o checkpoint no `model.checkpoint_dir` configurado para o modelo selecionado;
* permitir sobrescrever `--config`, `--experiment`, `--train-dir`, `--filename`, `--epochs`, `--batch-size`, `--learning-rate`, `--label-smoothing` e `--load-model`;
* usar 10 épocas como padrão para `--model m3` quando `--epochs` não for informado;
* usar learning rate `0.001` como padrão para `--model m3` quando `--learning-rate` não for informado;
* usar label smoothing `0.1` como padrão para `--model m3` quando `--label-smoothing` não for informado;
* imprimir um resumo JSON do checkpoint gerado/restaurado.

Esse script não deve:

* executar ataques adversariais;
* executar o filtro de detecção;
* escrever `metrics.csv`, `metrics.json` ou `manifest.json`;
* substituir `scripts/run_experiment.py` como entrada pública dos experimentos.

---

## Attack Requirements

### FGSM/M3

O ataque FGSM deve usar:

```yaml
name: fgsm
epsilon: 0.2
clip_min: 0.0
clip_max: 1.0
```

A implementação não deve alterar a implementação oficial do FGSM.

### CW L2/M2

O ataque CW L2 deve usar:

```yaml
name: cw_l2_nn_robust
kappa: 0.0
```

A implementação deve reutilizar o backend robusto já adotado no fluxo M2 da Table 10 e não deve alterar a implementação oficial do CW L2.

---

## Filter Requirements

O experimento deve usar o filtro oficial do projeto:

```yaml
filter:
  name: proposed_detection_filter
  type: proposed_detection_filter
```

A implementação não deve criar um filtro específico para Fashion-MNIST.

O filtro deve receber imagens no mesmo formato e escala definidos pelo dataset:

```text
28x28x1
[0.0, 1.0]
```

Se internamente o filtro precisar operar em escala `0-255`, a conversão deve ser feita de forma encapsulada e consistente, sem alterar a escala pública do experimento.

---

## Output Artifacts

Cada experimento deve gerar apenas os artefatos oficiais abaixo.

### FGSM/M3

```text
results/experiments/fashion_mnist/fgsm_m3/
  metrics.csv
  metrics.json
  manifest.json
```

### CW L2/M2

```text
results/experiments/fashion_mnist/cw_l2_m2/
  metrics.csv
  metrics.json
  manifest.json
```

Não devem ser gerados:

```text
diagnostics.md
debug.json
plots
notebooks
adversarial image dumps públicos
filtered image dumps públicos
```

Artefatos adicionais só podem ser gerados se forem temporários, locais e ignorados pelo Git.

---

## `metrics.csv` Schema

Cada `metrics.csv` deve seguir exatamente:

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
```

Cada arquivo deve conter somente uma linha de métricas, correspondente à única linha Table 10 executada pelo experimento público.

---

## `metrics.json` Schema

O arquivo `metrics.json` deve conter:

```json
{
  "table": 10,
  "dataset_group": "fashion_mnist",
  "model_group": "m3",
  "rows": [
    {
      "no": 1,
      "attack_model": "FGSM (ε=0.2)/M3",
      "dataset": "Fashion-MNIST",
      "num_failures": 0,
      "tp": 0,
      "fn": 0,
      "fp": 0,
      "rtp": 0,
      "rtp_percent": 0.0,
      "recall": 0.0,
      "precision": 0.0,
      "f1": 0.0
    }
  ]
}
```

Para o experimento `fashion_mnist_cw_l2_m2`, `model_group` deve ser `m2`, `no` deve ser `9` e `attack_model` deve ser `CW L2 (κ=0.0)/M2`.

Os campos de métricas devem usar os mesmos nomes e unidades do schema oficial da Table 10. Percentuais devem ser gravados como valores percentuais numéricos, não frações.

---

## `manifest.json` Schema

O `manifest.json` deve registrar:

```json
{
  "table": 10,
  "experiment_id": "fashion_mnist_fgsm_m3",
  "dataset_group": "fashion_mnist",
  "model_group": "m3",
  "dataset": {
    "name": "fashion_mnist",
    "domain": "mnist_compatible",
    "split": "test",
    "csv_path": "data/fashion_mnist/fashion-mnist_test.csv",
    "split_strategy": {
      "name": "balanced_by_class",
      "train_samples": 9000,
      "evaluation_samples": 1000,
      "train_class_quota": 900,
      "evaluation_class_quota": 100
    },
    "image_shape": [28, 28, 1],
    "value_range": {
      "min": 0.0,
      "max": 1.0
    },
    "class_order": [
      "t_shirt_top",
      "trouser",
      "pullover",
      "dress",
      "coat",
      "sandal",
      "shirt",
      "sneaker",
      "bag",
      "ankle_boot"
    ],
    "class_quotas": {
      "t_shirt_top": 100,
      "trouser": 100,
      "pullover": 100,
      "dress": 100,
      "coat": 100,
      "sandal": 100,
      "shirt": 100,
      "sneaker": 100,
      "bag": 100,
      "ankle_boot": 100
    },
    "clean_errors": {
      "t_shirt_top": 0,
      "trouser": 0,
      "pullover": 0,
      "dress": 0,
      "coat": 0,
      "sandal": 0,
      "shirt": 0,
      "sneaker": 0,
      "bag": 0,
      "ankle_boot": 0
    },
    "clean_correct": {
      "t_shirt_top": 100,
      "trouser": 100,
      "pullover": 100,
      "dress": 100,
      "coat": 100,
      "sandal": 100,
      "shirt": 100,
      "sneaker": 100,
      "bag": 100,
      "ankle_boot": 100
    },
    "selected_clean_correct": {
      "t_shirt_top": 100,
      "trouser": 100,
      "pullover": 100,
      "dress": 100,
      "coat": 100,
      "sandal": 100,
      "shirt": 100,
      "sneaker": 100,
      "bag": 100,
      "ankle_boot": 100
    },
    "selection_policy": "discard_clean_errors",
    "checkpoint_training": {
      "source_csv": "data/fashion_mnist/fashion-mnist_test.csv",
      "split_name": "train",
      "selection": {
        "method": "first_n_per_class",
        "per_class_start": 0,
        "per_class_end": 900,
        "class_quota": 900,
        "total_samples": 9000
      },
      "validation_sample": {
        "split_name": "evaluation",
        "method": "next_n_per_class",
        "per_class_start": 900,
        "per_class_end": 1000,
        "class_quota": 100,
        "total_samples": 1000,
        "excludes_training": true
      }
    }
  },
  "rows": [
    {
      "no": 1,
      "attack_model": "FGSM (ε=0.2)/M3",
      "attack": {
        "name": "fgsm",
        "epsilon": 0.2
      },
      "status": "completed"
    }
  ]
}
```

Para o experimento `fashion_mnist_cw_l2_m2`, `experiment_id` deve ser `fashion_mnist_cw_l2_m2`, `model_group` deve ser `m2`, `no` deve ser `9`, `attack_model` deve ser `CW L2 (κ=0.0)/M2` e o ataque registrado deve ser `cw_l2_nn_robust`.

Se o experimento falhar, o manifesto deve registrar:

```json
{
  "status": "blocked",
  "blocked_reason": "..."
}
```

---

## Error Cases

A implementação deve falhar com erro claro se:

* `dataset.domain` não for `mnist_compatible`;
* `dataset.name` não for `fashion_mnist`;
* `dataset.csv_path` não existir;
* o CSV não tiver exemplos suficientes para separar 900 exemplos de treino e 100 exemplos de avaliação por classe;
* `checkpoint_training` não estiver configurado para Fashion-MNIST;
* `checkpoint_training.source_csv` divergir de `dataset.csv_path`;
* `checkpoint_training.selection` divergir da amostra de treino `[0, 900)` por classe;
* `checkpoint_training.validation_sample` divergir da amostra de avaliação `[900, 1000)` por classe;
* houver sobreposição entre exemplos de treino e exemplos de avaliação;
* imagens não puderem ser convertidas para `28x28x1`;
* labels não estiverem no intervalo `0..9`;
* `class_order` não tiver exatamente 10 classes;
* `class_indices` não cobrir todas as classes;
* `checkpoint_dir` do modelo não existir;
* checkpoint do modelo não for compatível com Fashion-MNIST;
* modelo tiver número de classes diferente de 10;
* `fashion_mnist_fgsm_m3` tentar usar `model.name` diferente de `m3`;
* `fashion_mnist_fgsm_m3` tentar carregar checkpoint de M1 ou MNIST dígitos;
* o módulo M3 não estiver disponível em `src/deepdetector/models`;
* `kind` não for `table_10_group`;
* o ataque configurado não estiver implementado;
* o filtro configurado não existir;
* não houver nenhuma amostra clean-correct na partição de avaliação;
* o experimento tentar usar modelo ImageNet;
* o experimento tentar usar dataset ImageNet.

---

## Acceptance Criteria

A implementação será aceita se:

1. O comando abaixo executar somente o experimento FGSM/M3:

```bash
python scripts/run_experiment.py --experiment fashion_mnist_fgsm_m3
```

2. O comando abaixo executar somente o experimento CW L2/M2:

```bash
python scripts/run_experiment.py --experiment fashion_mnist_cw_l2_m2
```

3. Ambos os experimentos usarem Fashion-MNIST.

4. Ambos os experimentos validarem `domain: mnist_compatible`.

5. Ambos os experimentos usarem imagens `28x28x1`.

6. Ambos os experimentos carregarem imagens de `data/fashion_mnist/fashion-mnist_test.csv`.

7. A separação de dados for balanceada, com 9000 exemplos para treino e 1000 exemplos para avaliação.

8. Cada classe Fashion-MNIST contribuir com 900 exemplos para treino e 100 exemplos para avaliação.

9. Os 1000 exemplos de avaliação não forem usados para treinar M3 ou M2.

10. O experimento FGSM usar M3.

11. O experimento CW L2 usar M2.

12. O experimento FGSM usar `epsilon: 0.2`.

13. O experimento CW L2 usar `kappa: 0.0`.

14. Ambos os experimentos aplicarem o filtro oficial do DeepDetector.

15. Ambos os experimentos calcularem `num_failures`, `tp`, `fn`, `fp`, `rtp`, `rtp_percent`, `recall`, `precision` e `f1`.

16. Ambos os experimentos gerarem `metrics.csv`.

17. Ambos os experimentos gerarem `metrics.json`.

18. Ambos os experimentos gerarem `manifest.json`.

19. Nenhum experimento ImageNet for executado.

20. Nenhum script paralelo fora do runner central for necessário para executar os experimentos públicos.

21. Os outputs forem gravados apenas nos diretórios oficiais definidos nesta spec.

22. O script `scripts/train_fashion_mnist_checkpoint.py --model m3` usar a partição balanceada de treino Fashion-MNIST e gravar no checkpoint configurado para M3.

23. O script `scripts/train_fashion_mnist_checkpoint.py --model m2` usar a partição balanceada de treino Fashion-MNIST e gravar no checkpoint configurado para M2.

24. `configs/experiments.yaml` declarar explicitamente a amostra de treino do checkpoint em `checkpoint_training` para M3 e M2.

25. O script de checkpoint falhar se `checkpoint_training` divergir da separação balanceada usada pelo loader.

26. Os experimentos Fashion-MNIST descartarem erros limpos da partição de avaliação e registrarem esses descartes no `manifest.json`, sem exigir 100 clean-correct por classe.

27. O módulo M3 existir em `src/deepdetector/models` e expor helpers de construção e checkpoint compatíveis com o padrão dos modelos MNIST existentes.

28. O M3 aceitar entrada `28x28x1`, produzir 10 classes e usar checkpoints em `artifacts/models/fashion_mnist/m3/checkpoints`.

29. Nenhum experimento Fashion-MNIST público usar M1 como classificador.

30. O M3 usar a configuração inicial com dropout `0.25/0.25/0.5`, 10 épocas, learning rate `0.001` e label smoothing `0.1` por padrão para `--model m3`.

---

## Reporting Guidance

No relatório final, descrever estes experimentos como:

```text
Extensão experimental em Fashion-MNIST
```

e não como reprodução oficial da Table 10.

Texto sugerido:

```text
Além da reprodução principal, avaliamos o DeepDetector em um novo dataset compatível em formato com o fluxo MNIST. Utilizamos Fashion-MNIST por possuir imagens grayscale 28x28 e 10 classes, preservando a estrutura de entrada dos modelos M3 e M2. Foram executadas duas combinações: FGSM (ε=0.2)/M3 e CW L2 (κ=0.0)/M2. Os experimentos mantiveram a regra de detecção original, baseada na comparação entre a predição da imagem adversarial e a predição após a transformação adaptativa.
```

Se os modelos tiverem sido treinados em Fashion-MNIST:

```text
Os modelos M3 e M2 foram treinados/restaurados com checkpoints específicos para Fashion-MNIST, permitindo interpretar as métricas agregadas de forma semanticamente válida.
```

Se os modelos forem os checkpoints originais de MNIST:

```text
Como os checkpoints disponíveis foram treinados em MNIST dígitos, os resultados em Fashion-MNIST devem ser interpretados como avaliação fora de domínio, e não como desempenho final de classificação sobre as classes de vestuário.
```

---

## Non-goals

Esta spec não busca:

* reproduzir todas as linhas da Table 10;
* comparar Fashion-MNIST com ImageNet;
* treinar novos modelos como parte obrigatória do runner;
* salvar imagens adversariais publicamente;
* otimizar hiperparâmetros dos ataques;
* introduzir novos filtros;
* modificar a regra central de detecção;
* substituir os experimentos oficiais de reprodução.

---

## Implementation Notes

A implementação deve ser pequena e integrada ao padrão atual do projeto.

Recomenda-se criar ou adaptar:

```text
scripts/train_fashion_mnist_checkpoint.py
src/deepdetector/data/fashion_mnist.py
src/deepdetector/models/mnist_m3.py
src/deepdetector/evaluation/tables/table_10.py
```

O runner central deve reutilizar o kind já existente:

```yaml
kind: table_10_group
```

O fluxo `table_10_group` deve ser estendido para avaliar `dataset.name: fashion_mnist` com `model_group: m3` e `model_group: m2`, sem criar um novo kind público.

O código deve reutilizar, sempre que possível:

```text
ataques já existentes
filtro proposto já existente
writers de metrics.csv, metrics.json e manifest.json
validações de output_dir
padrão de execução do scripts/run_experiment.py
```
