# SPEC — Top-K Prediction Detection for Ambiguous Images

## Objective

Implementar e avaliar uma melhoria no DeepDetector: usar **top-k predições** em vez de apenas a classe mais provável na regra de detecção.

A motivação é reduzir falsos positivos em imagens ambíguas, nas quais a transformação adaptativa pode alterar a classe top-1 sem necessariamente indicar comportamento adversarial.

A regra atual de detecção compara apenas a classe mais provável:

```text
C_top1(x) != C_top1(T(x))
````

A nova regra deve considerar as `k` classes mais prováveis:

```text
top_k(C(x)) ∩ top_k(C(T(x))) == ∅
```

Se houver interseção entre os conjuntos top-k antes e depois da transformação, a amostra não deve ser considerada detectada.

A implementação deve seguir a arquitetura atual do projeto, utilizando exclusivamente o runner centralizado:

```bash
python scripts/run_experiment.py --experiment topk_detection
```

---

## Scope

Este experimento deve:

* selecionar imagens ambíguas para avaliação;
* implementar uma regra de detecção baseada em top-k;
* comparar a regra original top-1 com a nova regra top-k;
* avaliar imagens limpas e adversariais;
* medir impacto em falsos positivos;
* medir impacto em recall, precision e F1;
* executar o experimento usando o runner centralizado;
* gerar apenas os artefatos oficiais definidos pelo projeto.

Este experimento não deve:

* substituir permanentemente a regra padrão do DeepDetector sem configuração explícita;
* alterar a implementação dos ataques;
* alterar os pesos dos modelos;
* alterar a transformação adaptativa `T`;
* executar defense-aware CW-L2;
* criar múltiplos experimentos públicos;
* gerar diagnósticos públicos;
* gerar relatórios em Markdown;
* criar artefatos intermediários permanentes fora dos caminhos definidos.

---

## Background

O DeepDetector detecta exemplos adversariais comparando a predição da imagem original ou adversarial antes e depois de uma transformação de redução de ruído.

Na regra top-1, uma amostra é considerada alterada quando:

```text
argmax(C(x)) != argmax(C(T(x)))
```

Esse critério é sensível a imagens ambíguas. Por exemplo, uma imagem pode ter predições próximas para classes visualmente semelhantes. Após a transformação, a classe top-1 pode mudar, mas a classe anterior ainda pode continuar entre as mais prováveis.

Nesse caso, marcar a imagem como falso positivo pode ser excessivamente rígido.

A melhoria proposta usa top-k para tornar a decisão mais robusta:

```text
top_k(C(x)) ∩ top_k(C(T(x))) != ∅
```

Se os conjuntos ainda compartilham pelo menos uma classe, a transformação não será considerada uma mudança semântica forte.

---

## Public Execution Interface

A única interface pública permitida é:

```bash
python scripts/run_experiment.py --experiment topk_detection
```

Não devem existir comandos adicionais como:

```bash
python scripts/run_experiment.py --experiment top1_detection
python scripts/run_experiment.py --experiment top3_detection
python scripts/run_experiment.py --experiment ambiguous_images
python scripts/run_experiment.py --experiment topk_ambiguous
python scripts/run_experiment.py --experiment reduce_false_positives
```

A seleção de imagens ambíguas, a avaliação top-1 e a avaliação top-k devem ser componentes internos do mesmo experimento.

---

## Configuration

A configuração deve residir em:

```text
configs/experiments.yaml
```

Estrutura sugerida:

```yaml
topk_detection:
  description: Top-k prediction based detection for ambiguous images

  seed: 42

  dataset:
    name: imagenet
    split: test
    classes:
      - ambulance
      - school_bus
      - soccer_ball
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

  ambiguity_selection:
    enabled: true
    method: top1_top2_margin
    max_margin: 0.15
    min_top1_confidence: 0.20
    max_top1_confidence: 0.80
    max_samples_per_class: 50

  topk_detection:
    baseline_k: 1
    candidate_k_values:
      - 2
      - 3
      - 5
    default_k: 3

  output:
    aggregate_by_class: true
    aggregate_global: true
    include_selection_summary: true
```

Não devem existir configurações independentes:

```yaml
top1_detection:
top2_detection:
top3_detection:
top5_detection:
ambiguous_images:
```

---

# Dataset

## Dataset Choice

Este experimento deve usar o subconjunto ImageNet já escolhido para avaliação em novo conjunto de dados:

```text
ambulance
school_bus
soccer_ball
```

A estrutura esperada dos dados locais é:

```text
data/
└── imagenet/
    └── test/
        ├── ambulance/
        ├── school_bus/
        └── soccer_ball/
```

---

## Dataset Validation

Antes da execução, o sistema deve validar:

* se as três classes existem localmente;
* se cada classe possui pelo menos uma imagem;
* se os arquivos possuem extensão de imagem suportada;
* se as classes configuradas correspondem aos diretórios esperados.

Se alguma classe estiver ausente, a execução deve falhar com erro claro.

Exemplo:

```text
Missing ImageNet class directory: data/imagenet/test/ambulance
```

---

# Ambiguous Image Selection

## Motivation

A melhoria top-k deve ser testada principalmente em imagens ambíguas, pois são os casos em que a regra top-1 tende a produzir falsos positivos.

Uma imagem será considerada ambígua quando o modelo não estiver extremamente confiante na classe top-1 ou quando as classes top-1 e top-2 tiverem probabilidades próximas.

---

## Selection Input

A seleção deve ser feita a partir das imagens limpas do dataset configurado.

Para cada imagem, executar inferência no modelo original:

```python
probs = softmax(model(x))
```

Obter:

```python
top1_class, top1_prob
top2_class, top2_prob
margin = top1_prob - top2_prob
```

---

## Ambiguity Criterion

Uma imagem deve ser considerada ambígua se satisfizer:

```python
margin <= max_margin
```

e:

```python
min_top1_confidence <= top1_prob <= max_top1_confidence
```

Valores sugeridos:

```yaml
max_margin: 0.15
min_top1_confidence: 0.20
max_top1_confidence: 0.80
```

---

## Selection by Class

A seleção deve ser feita separadamente por classe.

Para cada classe:

```python
selected_images[class_name] = ambiguous_images[:max_samples_per_class]
```

A ordenação deve priorizar as imagens mais ambíguas:

```python
sort by margin ascending
```

Ou seja, imagens com menor diferença entre top-1 e top-2 aparecem primeiro.

---

## Fallback Rule

Se uma classe não tiver imagens suficientes que satisfaçam o critério de ambiguidade, completar a seleção com as imagens de menor margem disponíveis.

Exemplo:

```python
if len(ambiguous_images) < max_samples_per_class:
    fill_with_lowest_margin_images()
```

A execução não deve falhar apenas porque uma classe tem poucas imagens ambíguas.

---

## Selection Counters

A seleção deve contabilizar por classe:

```text
total_loaded
ambiguous_candidates
selected_for_experiment
mean_top1_confidence
mean_top1_top2_margin
```

Esses contadores devem ser salvos em `selection.csv` e `selection.json`.

---

# Detection Rules

## Baseline Top-1 Rule

A regra original considera uma mudança quando a classe mais provável muda após a transformação:

```python
detected_top1 = top1(C(x)) != top1(C(T(x)))
```

Para imagens adversariais:

```python
detected_top1 = top1(C(x_adv)) != top1(C(T(x_adv)))
```

---

## Top-K Rule

A nova regra considera uma mudança apenas quando não há interseção entre os conjuntos top-k:

```python
before = set(top_k(C(x), k))
after = set(top_k(C(T(x)), k))

detected_topk = len(before.intersection(after)) == 0
```

Para imagens adversariais:

```python
before = set(top_k(C(x_adv), k))
after = set(top_k(C(T(x_adv)), k))

detected_topk = len(before.intersection(after)) == 0
```

---

## Expected Effect

A regra top-k deve reduzir falsos positivos em imagens limpas ambíguas.

A melhoria esperada é:

```text
FP_topk <= FP_top1
```

Entretanto, é aceitável que o recall em adversariais diminua, pois a regra top-k é mais tolerante.

Por isso, o experimento deve reportar separadamente:

* redução de falsos positivos;
* recall adversarial;
* precision;
* F1;
* trade-off entre FP e FN.

---

# Top-K Values

O experimento deve avaliar os seguintes valores:

```text
k = 1
k = 2
k = 3
k = 5
```

Onde:

* `k = 1` representa a regra original;
* `k = 2`, `k = 3` e `k = 5` representam a melhoria proposta.

O valor recomendado para análise principal deve ser:

```text
k = 3
```

---

# Processing Flow

## Step 1 — Load Dataset

Carregar imagens das classes configuradas:

```python
classes = ["ambulance", "school_bus", "soccer_ball"]
```

---

## Step 2 — Clean Prediction for Ambiguity Selection

Para cada imagem limpa:

```python
probs = predict_proba(model, x)
top_classes = top_k(probs, k=5)
top1_prob = probs[top_classes[0]]
top2_prob = probs[top_classes[1]]
margin = top1_prob - top2_prob
```

---

## Step 3 — Select Ambiguous Images

Selecionar imagens com base no critério:

```python
margin <= max_margin
```

e:

```python
min_top1_confidence <= top1_prob <= max_top1_confidence
```

Ordenar por menor margem e limitar:

```python
max_samples_per_class
```

---

## Step 4 — Clean Image Evaluation

Para cada imagem selecionada:

```python
x_transformed = T(x)
```

Calcular predições antes e depois da transformação:

```python
probs_before = C(x)
probs_after = C(T(x))
```

Avaliar FP para cada `k`:

```python
detected_clean_k = no_intersection(
    top_k(probs_before, k),
    top_k(probs_after, k)
)
```

Se `detected_clean_k == True`:

```python
FP_k += 1
```

Caso contrário:

```python
TN_k += 1
```

---

## Step 5 — Generate Adversarial Example

Gerar imagem adversarial usando FGSM:

```python
x_adv = fgsm(model, x, y_original, epsilon=1/255)
```

Aplicar clamp:

```python
x_adv = clamp(x_adv, min_value=0.0, max_value=1.0)
```

---

## Step 6 — Attack Validation

Executar inferência na imagem adversarial:

```python
adv_pred = top1(C(x_adv))
```

Se o ataque não alterar a classificação top-1 original:

```python
disturbed_failure += 1
```

A amostra não participa do cálculo de TP/FN adversarial.

Se alterar:

```python
test_number += 1
```

---

## Step 7 — Adversarial Detection

Aplicar a transformação:

```python
x_adv_transformed = T(x_adv)
```

Calcular predições antes e depois:

```python
probs_adv_before = C(x_adv)
probs_adv_after = C(T(x_adv))
```

Para cada `k`:

```python
detected_adv_k = no_intersection(
    top_k(probs_adv_before, k),
    top_k(probs_adv_after, k)
)
```

Se `detected_adv_k == True`:

```python
TP_k += 1
```

Caso contrário:

```python
FN_k += 1
```

---

# Counting Rules

Os contadores devem ser mantidos:

1. por classe;
2. por valor de `k`;
3. globalmente por valor de `k`.

---

## Total Loaded

Número de imagens carregadas antes da seleção:

```python
total_loaded += 1
```

---

## Selected Clean Images

Número de imagens selecionadas como ambíguas:

```python
selected += 1
```

---

## False Positive

Imagem limpa marcada incorretamente como adversarial:

```python
FP_k += 1
```

Condição:

```python
top_k(C(x), k) ∩ top_k(C(T(x)), k) == ∅
```

---

## True Negative

Imagem limpa corretamente não marcada como adversarial:

```python
TN_k += 1
```

Condição:

```python
top_k(C(x), k) ∩ top_k(C(T(x)), k) != ∅
```

---

## True Positive

Imagem adversarial corretamente detectada:

```python
TP_k += 1
```

Condição:

```python
top_k(C(x_adv), k) ∩ top_k(C(T(x_adv)), k) == ∅
```

---

## False Negative

Imagem adversarial não detectada:

```python
FN_k += 1
```

Condição:

```python
top_k(C(x_adv), k) ∩ top_k(C(T(x_adv)), k) != ∅
```

---

## Disturbed Failure

Ataque adversarial não altera a classificação top-1:

```python
disturbed_failure += 1
```

Essas amostras não entram nos cálculos de TP/FN.

---

# Aggregation Strategy

As métricas globais devem ser calculadas por soma dos contadores das classes.

Para cada `k`:

```python
TP_global_k = sum(TP_class_k)
FN_global_k = sum(FN_class_k)
FP_global_k = sum(FP_class_k)
TN_global_k = sum(TN_class_k)
```

---

## Forbidden Aggregation

Não utilizar média simples das métricas por classe.

Proibido:

```python
fpr_global = mean([
    fpr_ambulance,
    fpr_school_bus,
    fpr_soccer_ball
])
```

Proibido:

```python
recall_global = mean([
    recall_ambulance,
    recall_school_bus,
    recall_soccer_ball
])
```

---

## Correct Aggregation

Após somar os contadores:

```python
false_positive_rate = FP / (FP + TN)
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

## False Positive Rate

```text
FP / (FP + TN)
```

---

## False Positive Reduction

Redução de falsos positivos em relação ao baseline top-1:

```text
(FP_top1 - FP_topk) / FP_top1
```

Se `FP_top1 == 0`, registrar:

```text
0.0
```

---

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
test_number / selected
```

---

# Output Artifacts

A execução deve gerar apenas:

```text
results/
└── experiments/
    └── topk_detection/
        ├── selection.csv
        ├── selection.json
        ├── metrics.csv
        └── metrics.json
```

---

## Selection CSV Format

```csv
class,total_loaded,ambiguous_candidates,selected_for_experiment,mean_top1_confidence,mean_top1_top2_margin
```

Exemplo:

```csv
class,total_loaded,ambiguous_candidates,selected_for_experiment,mean_top1_confidence,mean_top1_top2_margin
ambulance,50,18,18,0.61,0.09
school_bus,50,14,14,0.66,0.11
soccer_ball,50,22,22,0.58,0.08
global,150,54,54,0.61,0.09
```

Os números acima são apenas ilustrativos.

---

## Selection JSON Format

```json
{
  "ambulance": {
    "total_loaded": 0,
    "ambiguous_candidates": 0,
    "selected_for_experiment": 0,
    "mean_top1_confidence": 0.0,
    "mean_top1_top2_margin": 0.0
  },
  "school_bus": {
    "total_loaded": 0,
    "ambiguous_candidates": 0,
    "selected_for_experiment": 0,
    "mean_top1_confidence": 0.0,
    "mean_top1_top2_margin": 0.0
  },
  "soccer_ball": {
    "total_loaded": 0,
    "ambiguous_candidates": 0,
    "selected_for_experiment": 0,
    "mean_top1_confidence": 0.0,
    "mean_top1_top2_margin": 0.0
  },
  "global": {
    "total_loaded": 0,
    "ambiguous_candidates": 0,
    "selected_for_experiment": 0,
    "mean_top1_confidence": 0.0,
    "mean_top1_top2_margin": 0.0
  }
}
```

---

## Metrics CSV Format

```csv
class,k,selected,test_number,disturbed_failure,TP,FN,FP,TN,recall_percent,precision_percent,f1_percent,false_positive_rate_percent,false_positive_reduction_percent,attack_success_rate_percent
```

Exemplo:

```csv
class,k,selected,test_number,disturbed_failure,TP,FN,FP,TN,recall_percent,precision_percent,f1_percent,false_positive_rate_percent,false_positive_reduction_percent,attack_success_rate_percent
ambulance,1,18,15,3,13,2,5,13,86.67,72.22,78.79,27.78,0.00,83.33
ambulance,3,18,15,3,11,4,2,16,73.33,84.62,78.57,11.11,60.00,83.33
global,1,54,45,9,38,7,14,40,84.44,73.08,78.35,25.93,0.00,83.33
global,3,54,45,9,32,13,5,49,71.11,86.49,78.05,9.26,64.29,83.33
```

Os números acima são apenas ilustrativos.

---

## Metrics JSON Format

O arquivo `metrics.json` deve conter os mesmos valores semânticos do CSV.

```json
{
  "ambulance": {
    "1": {
      "selected": 0,
      "test_number": 0,
      "disturbed_failure": 0,
      "TP": 0,
      "FN": 0,
      "FP": 0,
      "TN": 0,
      "recall_percent": 0.0,
      "precision_percent": 0.0,
      "f1_percent": 0.0,
      "false_positive_rate_percent": 0.0,
      "false_positive_reduction_percent": 0.0,
      "attack_success_rate_percent": 0.0
    },
    "2": {},
    "3": {},
    "5": {}
  },
  "school_bus": {
    "1": {},
    "2": {},
    "3": {},
    "5": {}
  },
  "soccer_ball": {
    "1": {},
    "2": {},
    "3": {},
    "5": {}
  },
  "global": {
    "1": {},
    "2": {},
    "3": {},
    "5": {}
  }
}
```

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

A implementação pode criar ou reutilizar os seguintes módulos internos:

```text
src/deepdetector/data/imagenet.py
src/deepdetector/attacks/fgsm.py
src/deepdetector/filters/adaptive_noise_reduction.py
src/deepdetector/evaluation/topk_detection.py
src/deepdetector/experiments/topk_detection.py
```

O script público deve apenas resolver a configuração e chamar o experimento interno.

---

# Implementation Requirements

## Top-K Utility Requirement

Implementar uma função reutilizável para obter top-k classes:

```python
top_k_indices(probs, k)
```

A função deve retornar os índices das `k` maiores probabilidades em ordem decrescente.

---

## Top-K Detection Requirement

Implementar uma função reutilizável:

```python
is_detected_topk(probs_before, probs_after, k)
```

Comportamento esperado:

```python
before = set(top_k_indices(probs_before, k))
after = set(top_k_indices(probs_after, k))

return len(before.intersection(after)) == 0
```

---

## Baseline Equivalence Requirement

Para `k = 1`, a nova função deve produzir o mesmo comportamento da regra top-1 original:

```python
is_detected_topk(probs_before, probs_after, k=1)
```

deve ser equivalente a:

```python
argmax(probs_before) != argmax(probs_after)
```

---

## Ambiguous Selection Requirement

A seleção de imagens ambíguas deve ser feita antes da geração adversarial.

Não selecionar imagens com base na resposta ao ataque.

A ambiguidade deve ser calculada apenas usando a predição limpa original.

---

## Detector Consistency Requirement

A mesma transformação `T` deve ser usada em:

1. avaliação top-1;
2. avaliação top-k;
3. imagens limpas;
4. imagens adversariais.

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

## Improvement Requirement

O experimento deve permitir comparar explicitamente:

```text
k = 1
k = 2
k = 3
k = 5
```

A redução de falso positivo deve ser sempre calculada em relação a `k = 1`.

---

# Forbidden Artifacts

Não criar:

```text
results/experiments/topk_detection/debug/
```

Não criar:

```text
results/experiments/topk_detection/report.md
```

Não criar:

```text
results/experiments/topk_detection/diagnostic.json
```

Não criar:

```text
results/experiments/topk_detection/ambiguous_images/
```

Não criar:

```text
results/experiments/topk_detection/adversarial_examples/
```

Não criar:

```text
results/experiments/topk_detection/top1/
```

Não criar:

```text
results/experiments/topk_detection/top2/
```

Não criar:

```text
results/experiments/topk_detection/top3/
```

Não criar:

```text
results/experiments/topk_detection/top5/
```

---

# Acceptance Criteria

* O experimento é executado por:

```bash
python scripts/run_experiment.py --experiment topk_detection
```

* Apenas um experimento público existe.
* O experimento seleciona imagens ambíguas antes de gerar adversariais.
* A ambiguidade é calculada com base na margem entre top-1 e top-2.
* A seleção usa apenas predições limpas.
* A seleção não depende do resultado do ataque.
* O experimento avalia `k = 1`, `k = 2`, `k = 3` e `k = 5`.
* `k = 1` reproduz a regra top-1 original.
* A regra top-k usa interseção entre conjuntos de classes.
* A imagem só é detectada com top-k se não houver interseção entre os conjuntos antes e depois da transformação.
* A mesma transformação `T` é usada para todos os valores de `k`.
* A mesma transformação `T` é usada em imagens limpas e adversariais.
* O ataque usado é FGSM.
* O epsilon usado é `1/255`.
* A saída inclui `selection.csv`.
* A saída inclui `selection.json`.
* A saída inclui `metrics.csv`.
* A saída inclui `metrics.json`.
* O CSV de seleção possui exatamente:

```csv
class,total_loaded,ambiguous_candidates,selected_for_experiment,mean_top1_confidence,mean_top1_top2_margin
```

* O CSV de métricas possui exatamente:

```csv
class,k,selected,test_number,disturbed_failure,TP,FN,FP,TN,recall_percent,precision_percent,f1_percent,false_positive_rate_percent,false_positive_reduction_percent,attack_success_rate_percent
```

* O JSON possui os mesmos valores semânticos do CSV.
* O resultado global é calculado por soma dos contadores.
* O resultado global não é calculado por média simples das métricas por classe.
* A redução de falso positivo é calculada em relação a `k = 1`.
* Nenhum relatório adicional é produzido.
* Nenhum diagnóstico é produzido.
* Nenhum artefato morto é criado.
* Toda a lógica permanece compatível com o runner centralizado do projeto.
