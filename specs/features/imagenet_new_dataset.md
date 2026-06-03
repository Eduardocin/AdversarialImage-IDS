# SPEC — New Dataset Evaluation: ImageNet Ambulance, School Bus and Soccer Ball

## Objective

Cumprir o requisito do projeto de obter resultados do sistema proposto no artigo reproduzido em **outro conjunto de dados**, avaliando o DeepDetector em um novo subconjunto do ImageNet composto pelas classes:

```text
ambulance
school_bus
soccer_ball
````

O objetivo deste experimento é verificar se o comportamento observado nos experimentos reproduzidos do artigo também aparece em classes ImageNet diferentes das utilizadas originalmente no projeto.

A implementação deve seguir a arquitetura atual do projeto, utilizando exclusivamente o runner centralizado:

```bash
python scripts/run_experiment.py --experiment imagenet_new_classes
```

---

## Scope

Este experimento deve:

* utilizar ImageNet como dataset;
* utilizar apenas as classes `ambulance`, `school_bus` e `soccer_ball`;
* executar o pipeline de detecção adversarial reproduzido do artigo;
* gerar exemplos adversariais para as novas classes;
* aplicar a transformação adaptativa do DeepDetector;
* calcular métricas de detecção para o novo conjunto de dados;
* produzir resultados agregados por classe e resultados globais;
* gerar somente os artefatos oficiais definidos pelo projeto.

Este experimento não deve:

* reutilizar as classes ImageNet já usadas nos experimentos originais do projeto;
* executar MNIST;
* executar Table 6;
* executar Table 7;
* executar Table 8;
* executar Table 9;
* executar defense-aware CW-L2;
* alterar a implementação oficial dos filtros;
* gerar diagnósticos públicos;
* gerar relatórios em Markdown;
* criar experimentos auxiliares públicos.

---

## Background

O artigo reproduzido propõe detectar exemplos adversariais por meio de uma transformação adaptativa baseada em redução de ruído. A ideia central é comparar a predição da imagem adversarial antes e depois da transformação:

```text
C(x_adv) != C(T(x_adv))
```

Quando a transformação altera a predição do adversarial, o exemplo é considerado detectado.

Para cumprir o requisito do projeto, este experimento avalia o sistema em um novo subconjunto do ImageNet. ImageNet é um benchmark amplamente usado para avaliação de modelos de classificação visual em larga escala, e trabalhos posteriores também discutem sua importância como referência para estudar generalização de classificadores.

As classes escolhidas foram:

| Classe        | Tipo visual                                            | Justificativa                                                                                          |
| ------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `ambulance`   | Objeto/veículo com formato estruturado                 | Classe com padrões visuais fortes, como veículo, janelas, rodas, luzes e cores contrastantes           |
| `school_bus`  | Objeto/veículo com estrutura semelhante à ambulance    | Permite avaliar se o detector se comporta de forma consistente entre classes visualmente próximas      |
| `soccer_ball` | Objeto não veicular, compacto e com textura geométrica | Introduz uma classe visualmente distinta das duas anteriores, aumentando a diversidade do novo dataset |

A escolha dessas três classes é justificada por combinar:

* duas classes visualmente relacionadas: `ambulance` e `school_bus`;
* uma classe visualmente distinta: `soccer_ball`;
* objetos comuns e semanticamente bem definidos;
* classes ImageNet compatíveis com modelos pré-treinados de classificação;
* um subconjunto pequeno o suficiente para execução viável no ambiente do projeto.

---

## Public Execution Interface

A única interface pública permitida é:

```bash
python scripts/run_experiment.py --experiment imagenet_new_classes
```

Não devem existir comandos adicionais como:

```bash
python scripts/run_experiment.py --experiment ambulance
python scripts/run_experiment.py --experiment school_bus
python scripts/run_experiment.py --experiment soccer_ball
python scripts/run_experiment.py --experiment imagenet_ambulance
python scripts/run_experiment.py --experiment imagenet_school_bus
python scripts/run_experiment.py --experiment imagenet_soccer_ball
python scripts/run_experiment.py --experiment new_dataset
```

As três classes devem ser tratadas como componentes internos do mesmo experimento.

---

## Configuration

A configuração deve residir em:

```text
configs/experiments.yaml
```

Estrutura sugerida:

```yaml
imagenet_new_classes:
  description: Evaluation of DeepDetector on new ImageNet classes

  seed: 42

  dataset:
    name: imagenet
    split: test
    classes:
      - ambulance
      - school_bus
      - soccer_ball
    samples_per_class: all
    value_range:
      min: 0.0
      max: 1.0

  model:
    name: imagenet_classifier

  attack:
    type: fgsm
    epsilon: 0.00392156862745098  # 1/255

  detector:
    type: final_adaptive_detection_filter
    entropy_thresholds:
      low: 4.0
      medium: 5.0
    quantization:
      low_entropy_step: 128
      medium_entropy_step: 64
      high_entropy_step: 43
    spatial_filter:
      type: cross_mean
      radius: 3

  output:
    aggregate_by_class: true
    aggregate_global: true
```

Não devem existir configurações independentes:

```yaml
imagenet_ambulance:
imagenet_school_bus:
imagenet_soccer_ball:
```

---

# Dataset

## ImageNet New Classes

O dataset deve conter apenas imagens das seguintes classes:

```text
ambulance
school_bus
soccer_ball
```

A estrutura esperada dos dados locais é:

```text
data/
└── imagenet/
    └── new_test/
        ├── ambulance/
        ├── school_bus/
        └── soccer_ball/
```

Cada diretório deve conter apenas imagens pertencentes à respectiva classe.

---

## Dataset Validation

Antes da execução do experimento, o sistema deve validar:

* se as três classes existem localmente;
* se cada classe possui pelo menos uma imagem;
* se os arquivos possuem extensão de imagem suportada;
* se as classes configuradas correspondem aos diretórios esperados.

Se alguma classe estiver ausente, a execução deve falhar com erro claro.

Exemplo de erro permitido:

```text
Missing ImageNet class directory: data/imagenet/test/ambulance
```

---

## Class Selection Justification

A justificativa da escolha do novo dataset deve ser preservada para uso posterior no relatório.

Resumo da justificativa:

```text
As classes ambulance, school_bus e soccer_ball foram escolhidas por permitirem avaliar o sistema em um subconjunto ImageNet novo, visualmente diverso e não utilizado nos experimentos originais do projeto. Ambulance e school_bus representam objetos veiculares com estruturas visuais semelhantes, enquanto soccer_ball representa um objeto compacto com textura geométrica distinta. Essa combinação permite observar se o comportamento do detector se mantém tanto em classes visualmente próximas quanto em uma classe visualmente diferente.
```

Essa justificativa deve ser usada no relatório, mas não precisa ser salva como artefato do experimento.

---

# Attack Definition

## FGSM

O ataque utilizado deve ser FGSM.

Para ImageNet, utilizar:

```text
epsilon = 1/255
```

Valor decimal:

```text
0.00392156862745098
```

O ataque deve gerar uma imagem adversarial `x_adv` a partir da imagem original `x`.

Critério de sucesso do ataque:

```python
C(x_adv) != y_original
```

Se o ataque não alterar a classificação, a amostra deve ser contabilizada como falha de perturbação.

---

# DeepDetector Transformation

A transformação `T` deve corresponder ao filtro final adaptativo de detecção usado pelo projeto.

A intenção deste experimento é aplicar a transformação composta por:

1. cálculo de entropia;
2. quantização adaptativa;
3. filtro espacial para imagens de alta entropia;
4. escolha entre imagem quantizada e imagem filtrada quando aplicável.

---

## Step 1 — Clamp

Garantir que os valores estejam no intervalo esperado pelo modelo ImageNet atual do projeto:

```python
x = clamp(x, min_value=0.0, max_value=1.0)
```

---

## Step 2 — Entropy Calculation

Calcular a entropia da imagem.

Para imagens RGB:

```python
entropy = mean(
    entropy(red),
    entropy(green),
    entropy(blue)
)
```

---

## Step 3 — Adaptive Quantization Selection

Selecionar a transformação conforme a entropia.

| Condição        | Transformação                                         |
| --------------- | ----------------------------------------------------- |
| `H < 4.0`       | Scalar Quantization com step `128`                    |
| `4.0 ≤ H < 5.0` | Scalar Quantization com step `64`                     |
| `H ≥ 5.0`       | Scalar Quantization com step `43` + Cross Mean Filter |

---

## Step 4 — Low Entropy Transformation

Quando `H < 4.0`:

```python
input_final = scalar_quantization(input, step=128)
```

---

## Step 5 — Medium Entropy Transformation

Quando `4.0 ≤ H < 5.0`:

```python
input_final = scalar_quantization(input, step=64)
```

---

## Step 6 — High Entropy Transformation

Quando `H ≥ 5.0`, aplicar:

```python
input_after_q = scalar_quantization(input, step=43)
input_after_q_and_f = cross_mean_filter(input_after_q, radius=3)
input_final = choose_closer_filter(input, input_after_q, input_after_q_and_f)
```

A chamada do filtro espacial deve seguir a API atual do projeto:

```python
cross_mean_filter(image, radius=3)
```

Não utilizar diretamente a assinatura antiga do código original:

```python
cross_mean_filter(image, start=3, end=25, coefficient=13)
```

---

# Processing Flow

## Step 1 — Load Class Images

Para cada classe configurada:

```python
for class_name in ["ambulance", "school_bus", "soccer_ball"]:
    load_images(class_name)
```

---

## Step 2 — Clean Prediction

Executar inferência na imagem original.

Se a classificação original estiver incorreta:

```python
original_classified_wrong += 1
```

A amostra não participa da avaliação de detecção adversarial.

Caso contrário:

```python
total_valid += 1
```

---

## Step 3 — Generate Adversarial Example

Gerar amostra adversarial utilizando FGSM:

```python
x_adv = fgsm(model, x, y_original, epsilon=1/255)
```

---

## Step 4 — Attack Validation

Executar inferência na amostra adversarial:

```python
adv_pred = C(x_adv)
```

Se o ataque não alterar a classificação:

```python
disturbed_failure += 1
```

A amostra não participa do cálculo de TP/FN.

Se o ataque alterar a classificação:

```python
test_number += 1
```

---

## Step 5 — Apply DeepDetector Transformation

Aplicar a transformação adaptativa:

```python
x_adv_transformed = T(x_adv)
```

Executar nova inferência:

```python
adv_transformed_pred = C(x_adv_transformed)
```

---

## Step 6 — Detection

Se a predição da imagem adversarial mudar após a transformação:

```python
TP += 1
```

Condição:

```python
C(x_adv) != C(T(x_adv))
```

Caso contrário:

```python
FN += 1
```

Condição:

```python
C(x_adv) == C(T(x_adv))
```

---

## Step 7 — False Positive Evaluation

Aplicar a transformação também na imagem limpa:

```python
x_transformed = T(x)
```

Executar nova inferência:

```python
clean_transformed_pred = C(x_transformed)
```

Se a imagem limpa mudar de classe após a transformação:

```python
FP += 1
```

Condição:

```python
C(x) != C(T(x))
```

---

# Counting Rules

Os contadores devem ser mantidos por classe e também de forma global.

## Original Classified Wrong

Imagem limpa classificada incorretamente:

```python
original_classified_wrong += 1
```

---

## Disturbed Failure

Ataque adversarial não altera a classificação:

```python
disturbed_failure += 1
```

---

## Test Number

Número de exemplos adversariais válidos gerados:

```python
test_number += 1
```

---

## True Positive

Exemplo adversarial detectado:

```python
TP += 1
```

Condição:

```python
C(x_adv) != C(T(x_adv))
```

---

## False Negative

Exemplo adversarial não detectado:

```python
FN += 1
```

Condição:

```python
C(x_adv) == C(T(x_adv))
```

---

## False Positive

Imagem limpa alterada pela transformação:

```python
FP += 1
```

Condição:

```python
C(x) != C(T(x))
```

---

# Aggregation Strategy

O experimento deve produzir métricas:

1. por classe;
2. globais.

As métricas globais devem ser calculadas pela soma dos contadores das três classes.

```python
TP_total = TP_ambulance + TP_school_bus + TP_soccer_ball
FN_total = FN_ambulance + FN_school_bus + FN_soccer_ball
FP_total = FP_ambulance + FP_school_bus + FP_soccer_ball
```

---

## Forbidden Aggregation

Não utilizar média simples das métricas por classe.

Proibido:

```python
recall_global = mean([
    recall_ambulance,
    recall_school_bus,
    recall_soccer_ball
])
```

Proibido:

```python
precision_global = mean([
    precision_ambulance,
    precision_school_bus,
    precision_soccer_ball
])
```

Proibido:

```python
f1_global = mean([
    f1_ambulance,
    f1_school_bus,
    f1_soccer_ball
])
```

---

## Correct Aggregation

Após somar os contadores:

```python
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = (2 * recall * precision) / (recall + precision)
```

Se algum denominador for zero, registrar a métrica como:

```text
0.0
```

---

# Metrics

## Recall

```text
TP / (TP + FN)
```

---

## Precision

```text
TP / (TP + FP)
```

---

## F1 Score

```text
2 * Recall * Precision
----------------------
 Recall + Precision
```

---

## Attack Success Rate

```text
test_number / total_valid
```

---

## Original Accuracy on Selected Samples

```text
total_valid / total_loaded
```

---

# Output Artifacts

A execução deve gerar apenas:

```text
results/
└── experiments/
    └── imagenet_new_classes/
        ├── metrics.csv
        └── metrics.json
```

---

## CSV Format

```csv
class,total_loaded,total_valid,original_classified_wrong,disturbed_failure,test_number,TP,FN,FP,recall_percent,precision_percent,f1_percent,attack_success_rate_percent,original_accuracy_percent
```

Exemplo:

```csv
class,total_loaded,total_valid,original_classified_wrong,disturbed_failure,test_number,TP,FN,FP,recall_percent,precision_percent,f1_percent,attack_success_rate_percent,original_accuracy_percent
ambulance,50,47,3,5,42,36,6,2,85.71,94.74,90.00,89.36,94.00
school_bus,50,48,2,4,44,38,6,3,86.36,92.68,89.41,91.67,96.00
soccer_ball,50,45,5,6,39,30,9,4,76.92,88.24,82.19,86.67,90.00
global,150,140,10,15,125,104,21,9,83.20,92.04,87.40,89.29,93.33
```

Os números acima são apenas ilustrativos. (vamos seguir a mesma proporção dos testes 40/40/20)

---

## JSON Format

O arquivo `metrics.json` deve conter os mesmos valores semânticos do CSV.

```json
{
  "ambulance": {
    "total_loaded": 0,
    "total_valid": 0,
    "original_classified_wrong": 0,
    "disturbed_failure": 0,
    "test_number": 0,
    "TP": 0,
    "FN": 0,
    "FP": 0,
    "recall_percent": 0.0,
    "precision_percent": 0.0,
    "f1_percent": 0.0,
    "attack_success_rate_percent": 0.0,
    "original_accuracy_percent": 0.0
  },
  "school_bus": {
    "total_loaded": 0,
    "total_valid": 0,
    "original_classified_wrong": 0,
    "disturbed_failure": 0,
    "test_number": 0,
    "TP": 0,
    "FN": 0,
    "FP": 0,
    "recall_percent": 0.0,
    "precision_percent": 0.0,
    "f1_percent": 0.0,
    "attack_success_rate_percent": 0.0,
    "original_accuracy_percent": 0.0
  },
  "soccer_ball": {
    "total_loaded": 0,
    "total_valid": 0,
    "original_classified_wrong": 0,
    "disturbed_failure": 0,
    "test_number": 0,
    "TP": 0,
    "FN": 0,
    "FP": 0,
    "recall_percent": 0.0,
    "precision_percent": 0.0,
    "f1_percent": 0.0,
    "attack_success_rate_percent": 0.0,
    "original_accuracy_percent": 0.0
  },
  "global": {
    "total_loaded": 0,
    "total_valid": 0,
    "original_classified_wrong": 0,
    "disturbed_failure": 0,
    "test_number": 0,
    "TP": 0,
    "FN": 0,
    "FP": 0,
    "recall_percent": 0.0,
    "precision_percent": 0.0,
    "f1_percent": 0.0,
    "attack_success_rate_percent": 0.0,
    "original_accuracy_percent": 0.0
  }
}
```

---

# Reproducibility

A execução deve usar a seed definida em configuração:

```yaml
seed: 42
```

A seed deve ser aplicada sempre que o projeto já possuir suporte para controle determinístico.

Não adicionar metadados de reprodutibilidade ao `metrics.json`.

---

# Architecture Constraints

O experimento deve seguir o padrão arquitetural do projeto.

Toda execução deve partir de:

```text
scripts/run_experiment.py
```

A lógica deve residir em:

```text
src/deepdetector/
```

Nenhuma lógica experimental deve ser implementada diretamente em scripts.

---

## Suggested Internal Modules

A implementação pode reutilizar os módulos internos:

```text
src/deepdetector/attacks/fgsm.py
src/deepdetector/data/imagenet.py
src/deepdetector/filters/adaptive_noise_reduction.py
src/deepdetector/evaluation/detection.py
src/deepdetector/experiments/imagenet_new_classes.py
```

O script público deve apenas resolver a configuração e chamar o experimento interno.

---

# Implementation Requirements

## Dataset Requirement

O loader ImageNet deve aceitar uma lista explícita de classes:

```python
classes = ["ambulance", "school_bus", "soccer_ball"]
```

O experimento não deve carregar automaticamente todas as classes disponíveis em:

```text
data/imagenet/test/
```

---

## Detector Consistency Requirement

A mesma transformação `T` deve ser usada em:

1. avaliação de imagens limpas;
2. avaliação de imagens adversariais;
3. cálculo de FP, TP e FN.

Não podem existir versões divergentes da transformação dentro do experimento.

---

## Attack Requirement

O ataque FGSM deve usar o mesmo modelo utilizado na inferência limpa.

A amostra adversarial deve preservar o intervalo válido:

```text
[0.0, 1.0]
```

Após a perturbação, aplicar clamp:

```python
x_adv = clamp(x_adv, min_value=0.0, max_value=1.0)
```

---

## Aggregation Requirement

As métricas globais devem ser calculadas por soma dos contadores.

Não calcular métricas globais por média simples das métricas por classe.

---

## Report Requirement

O relatório final do projeto deve justificar a escolha das classes.

A justificativa mínima deve mencionar:

* uso de um subconjunto ImageNet novo;
* duas classes veiculares visualmente relacionadas;
* uma classe visualmente distinta;
* diversidade visual;
* viabilidade computacional.

---

# Forbidden Artifacts

Não criar:

```text
results/experiments/imagenet_new_classes/debug/
```

Não criar:

```text
results/experiments/imagenet_new_classes/report.md
```

Não criar:

```text
results/experiments/imagenet_new_classes/diagnostic.json
```

Não criar:

```text
results/experiments/imagenet_new_classes/ambulance/
```

Não criar:

```text
results/experiments/imagenet_new_classes/school_bus/
```

Não criar:

```text
results/experiments/imagenet_new_classes/soccer_ball/
```

Não criar:

```text
results/experiments/ambulance/
```

Não criar:

```text
results/experiments/school_bus/
```

Não criar:

```text
results/experiments/soccer_ball/
```

---

# Acceptance Criteria

* O experimento é executado por:

```bash
python scripts/run_experiment.py --experiment imagenet_new_classes
```

* Apenas um experimento público existe.
* O experimento usa exclusivamente as classes:

```text
ambulance
school_bus
soccer_ball
```

* As três classes são executadas internamente pelo mesmo experimento.
* O experimento não reutiliza as classes ImageNet originais do projeto.
* O loader valida a existência das três classes.
* Amostras limpas classificadas incorretamente são ignoradas na avaliação adversarial.
* O ataque usado é FGSM.
* O epsilon usado é `1/255`.
* A transformação `T` corresponde ao filtro final adaptativo de detecção do projeto.
* A mesma transformação `T` é usada para imagens limpas e adversariais.
* O experimento calcula resultados por classe.
* O experimento calcula resultado global.
* O resultado global é calculado por soma dos contadores.
* O resultado global não é calculado por média simples das métricas por classe.
* O CSV possui exatamente:

```csv
class,total_loaded,total_valid,original_classified_wrong,disturbed_failure,test_number,TP,FN,FP,recall_percent,precision_percent,f1_percent,attack_success_rate_percent,original_accuracy_percent
```

* O JSON possui os mesmos valores semânticos do CSV.
* O JSON não contém metadados extras.
* Nenhum relatório adicional é produzido.
* Nenhum diagnóstico é produzido.
* Nenhum artefato morto é criado.
* Toda a lógica permanece compatível com o runner centralizado do projeto.
