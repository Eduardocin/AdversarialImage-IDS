# SPEC — Table 6: Adaptive Quantization Strategy

## Objective

Reproduzir a **Table 6** do artigo *Detecting Adversarial Image Examples in Deep Networks with Adaptive Noise Reduction* utilizando a estratégia de **Adaptive Scalar Quantization** baseada em entropia.

A implementação deve seguir a arquitetura atual do projeto, utilizando exclusivamente o runner centralizado:

```bash
python scripts/run_experiment.py --experiment table_6
```

A Table 6 representa a etapa de validação da estratégia adaptativa de quantização, antes da introdução dos filtros espaciais utilizados posteriormente nas Tables 7, 8 e 9.

---

## Scope

Este experimento deve:

- executar a estratégia adaptativa de quantização;
- utilizar FGSM como único ataque;
- utilizar os conjuntos MNIST e ImageNet;
- produzir métricas agregadas para Training e Validation;
- gerar somente os artefatos oficiais definidos pelo projeto.

Este experimento não deve:

- aplicar filtros espaciais;
- executar DeepFool;
- executar Carlini & Wagner;
- executar a lógica da Table 7;
- executar a lógica da Table 8;
- executar a lógica da Table 9;
- executar a lógica da Table 10;
- gerar diagnósticos;
- gerar relatórios em Markdown;
- criar experimentos auxiliares públicos.

---

## Background

No artigo, os autores observam que diferentes imagens respondem melhor a diferentes níveis de quantização.

Após experimentos realizados sobre o conjunto de treinamento, foi definida a seguinte estratégia adaptativa:

| Entropia | Número de intervalos |
|---|---:|
| `H < 4.0` | 2 |
| `4.0 ≤ H < 5.0` | 4 |
| `H ≥ 5.0` | 6 |

A Table 6 avalia essa estratégia sobre os conjuntos de **Training** e **Validation**.

---

## Public Execution Interface

A única interface pública permitida é:

```bash
python scripts/run_experiment.py --experiment table_6
```

Não devem existir comandos adicionais como:

```bash
python scripts/run_experiment.py --experiment table_6_mnist
python scripts/run_experiment.py --experiment table_6_imagenet
```

MNIST e ImageNet devem ser tratados como componentes internos do experimento.

---

## Configuration

A configuração deve residir em:

```text
configs/experiments.yaml
```

Estrutura sugerida:

```yaml
table_6:
  description: Adaptive quantization strategy
  datasets:
    - mnist
    - imagenet

  attack:
    type: fgsm

  entropy_thresholds:
    low: 4.0
    medium: 5.0

  quantization:
    low_entropy_intervals: 2
    medium_entropy_intervals: 4
    high_entropy_intervals: 6
```

Não devem existir configurações independentes:

```yaml
table_6_mnist:
table_6_imagenet:
```

---

# Datasets

## MNIST

Utilizar os mesmos splits definidos pelo artigo.

| Split | Intervalo |
|---|---|
| Training | `0-4499` |
| Validation | `4500-5499` |

Ataque:

```text
FGSM ε = 0.2
```

---

## ImageNet

Utilizar os grupos locais correspondentes aos splits oficiais do projeto.
Classes presentes apenas em `data/imagenet/test`, como `zebra`, não devem ser
usadas no split `Validation`.

| Split | Classes |
|---|---|
| Training | Goldfish, Pineapple, Clock |
| Validation | Jellyfish |

Ataque:

```text
FGSM ε = 1/255
```

---

# Processing Flow

## Step 1 — Clean Prediction

Executar inferência na imagem original.

Se a classificação estiver incorreta:

```python
original_classified_wrong += 1
```

A amostra não participa da avaliação.

---

## Step 2 — Generate Adversarial Example

Gerar amostra adversarial utilizando FGSM.

Quando já existirem amostras adversariais compatíveis em
`artifacts/adversarial_examples/`, a execução deve reutilizá-las e não deve
chamar novamente a geração FGSM para o mesmo split.

Quando não existir cache compatível para um split, a execução deve gerar as
amostras adversariais via FGSM e persistir o resultado no caminho padrão para
que execuções futuras reutilizem o artefato.

Para ImageNet, o caminho padrão do cache por split é:

```text
artifacts/adversarial_examples/imagenet/fgsm/{split}/adversarial_examples.npy
```

---

## Step 3 — Attack Validation

Executar inferência na amostra adversarial.

Se o ataque não alterar a classificação:

```python
disturbed_failure += 1
```

A amostra não participa da avaliação.

---

## Step 4 — Entropy Calculation

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

## Step 5 — Adaptive Quantization Selection

Selecionar o número de intervalos.

| Condição | Intervalos |
|---|---:|
| `H < 4.0` | 2 |
| `4.0 ≤ H < 5.0` | 4 |
| `H ≥ 5.0` | 6 |

---

## Step 6 — Quantization

Aplicar quantização:

| Intervalos | Step |
|---:|---:|
| 2 intervalos | `128` |
| 4 intervalos | `64` |
| 6 intervalos | `43` |

---

## Step 7 — Detection

Aplicar a quantização:

- na imagem original;
- na imagem adversarial.

Executar nova inferência.

---

# Counting Rules

## False Positive

A imagem limpa muda de classe após a quantização.

```python
FP += 1
```

---

## True Positive

A imagem adversarial muda de classe após a quantização.

```python
TP += 1
```

---

## False Negative

A imagem adversarial permanece na mesma classe adversarial.

```python
FN += 1
```

---

# Aggregation Strategy

A Table 6 representa um único experimento.

MNIST e ImageNet devem ser executados internamente e agregados.

Os contadores finais são obtidos por soma:

```python
TP_total = TP_mnist + TP_imagenet
FN_total = FN_mnist + FN_imagenet
FP_total = FP_mnist + FP_imagenet
```

---

## Forbidden Aggregation

Não utilizar médias simples.

Proibido:

```python
f1 = mean([f1_mnist, f1_imagenet])
```

Proibido:

```python
recall = mean([recall_mnist, recall_imagenet])
```

Proibido:

```python
precision = mean([precision_mnist, precision_imagenet])
```

---

## Correct Aggregation

Após somar os contadores:

```python
recall = TP / (TP + FN)
precision = TP / (TP + FP)

f1 = (
    2 * recall * precision
) / (
    recall + precision
)
```

---

# Metrics

## Recall

```text
TP / (TP + FN)
```

## Precision

```text
TP / (TP + FP)
```

## F1 Score

```text
2 * Recall * Precision
----------------------
 Recall + Precision
```

---

# Output Artifacts

A execução deve gerar apenas:

```text
results/
└── experiments/
    └── table_6/
        ├── metrics.csv
        └── metrics.json
```

---

## CSV Format

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
```

Exemplo:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
train,3482,370,146,90.39,95.98,93.10
validation,939,81,48,92.06,95.14,93.58
```

Os números acima são apenas ilustrativos.

---

## JSON Format

```json
{
  "train": {
    "tp": 0,
    "fn": 0,
    "fp": 0,
    "recall_percent": 0,
    "precision_percent": 0,
    "f1_percent": 0
  },
  "validation": {
    "tp": 0,
    "fn": 0,
    "fp": 0,
    "recall_percent": 0,
    "precision_percent": 0,
    "f1_percent": 0
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

# Forbidden Artifacts

Não criar:

```text
results/experiments/table_6/mnist/
```

Não criar:

```text
results/experiments/table_6/imagenet/
```

Não criar:

```text
results/experiments/table_6/debug/
```

Não criar:

```text
results/experiments/table_6/report.md
```

Não criar:

```text
results/experiments/table_6/diagnostic.json
```

---

# Acceptance Criteria

- O experimento é executado por:

```bash
python scripts/run_experiment.py --experiment table_6
```

- Apenas um experimento público existe.
- MNIST e ImageNet são executados internamente.
- Amostras adversariais FGSM compatíveis em `artifacts/adversarial_examples/` são reutilizadas.
- Splits ImageNet sem cache compatível geram FGSM e gravam `adversarial_examples.npy`.
- Os resultados são agregados por soma dos contadores.
- Não existem experimentos públicos auxiliares.
- Não existem diretórios intermediários permanentes.
- O CSV possui exatamente:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
```

- O JSON possui os mesmos valores do CSV.
- Nenhum relatório adicional é produzido.
- Nenhum diagnóstico é produzido.
- Nenhum artefato morto é criado.
- Toda a lógica permanece compatível com o runner centralizado do projeto.
