# Spec 2 — Refatorar Tabelas 7 e 8 para fluxo genérico de avaliação de candidatos de filtro

## Status

Proposto.

## Contexto

Os experimentos das Tabelas 7 e 8 fazem parte da etapa intermediária de escolha do filtro de suavização usado posteriormente no filtro final da Tabela 9.

O fluxo conceitual é:

```text
Table 6 -> valida quantização adaptativa
Table 7 -> testa filtros de smoothing em imagens de alta entropia
Table 8 -> valida os melhores filtros de smoothing no validation set
Table 9 -> avalia o filtro final combinando quantização + smoothing
```

A Tabela 7 avalia múltiplas máscaras de suavização espacial, como:

* `cross mask`
* `diamond mask`
* `box mask`

Em diferentes tamanhos:

* `3x3`
* `5x5`
* `7x7`
* `9x9`

A Tabela 8 reaproveita os melhores candidatos da Tabela 7 e valida esses filtros no conjunto de validação.

Hoje, a tendência é criar scripts separados para cada tabela, duplicando:

* carregamento de dados;
* geração FGSM;
* cálculo de métricas;
* escrita de resultados.

Esta spec propõe transformar Tabela 7 e Tabela 8 em duas configurações diferentes de um mesmo runner genérico de avaliação de candidatos de filtro.

---

## Problema

As Tabelas 7 e 8 possuem fluxo quase idêntico:

1. Carregar dataset/slice configurado.
2. Restaurar modelo.
3. Gerar exemplos adversariais com FGSM.
4. Calcular predições limpas e adversariais.
5. Aplicar um conjunto de filtros candidatos.
6. Calcular TP, FN, FP, recall, precision e F1.
7. Escrever resultados.

A diferença principal entre elas é:

* **Table 7**: busca ampla de filtros em training/high-entropy samples.
* **Table 8**: validação de subconjunto dos melhores filtros.

Portanto, não faz sentido manter dois scripts com lógica própria. O correto é criar um runner reutilizável que receba os filtros via YAML.

---

## Objetivo

Criar um fluxo genérico para avaliação de candidatos de filtro de suavização, permitindo reproduzir as Tabelas 7 e 8 com o mesmo código e configurações diferentes.

Ao final desta spec:

* Table 7 e Table 8 devem usar o mesmo runner.
* Os filtros devem ser declarados em YAML.
* O código deve suportar `cross`, `diamond` e `box`.
* O ataque FGSM deve ser gerado uma única vez por execução e reutilizado para todos os filtros.
* As saídas devem ser padronizadas em CSV e JSON.
* O resultado da Table 8 deve preparar a escolha do filtro final usado na Table 9.

---

## Fora de escopo

Esta spec não deve:

* refatorar ainda a Table 6;
* refatorar ainda a Table 9;
* alterar a fórmula de cálculo das métricas;
* alterar a implementação matemática dos filtros;
* alterar a geração FGSM;
* alterar o modelo usado;
* remover completamente referências ao artigo em documentação histórica;
* criar CLI final definitivo.

A limpeza completa de YAML e outputs será tratada na Spec 4.

---

## Relação com a Spec 1

Esta spec assume que a infraestrutura criada na Spec 1 já existe ou será usada durante a implementação:

* `deepdetector.io.paths`
* `deepdetector.io.config`
* `deepdetector.io.result_writers`
* `deepdetector.experiments.metadata`

Portanto, o runner das Tabelas 7 e 8 deve usar:

* `load_yaml_config`
* `resolve_project_path`
* `ensure_dir`
* `write_experiment_outputs`
* `build_experiment_payload`

Em vez de implementar essa infraestrutura localmente.

---

## Proposta de solução

Criar um runner genérico de candidatos de filtro:

```text
src/deepdetector/experiments/filter_candidate_runner.py
```

Criar uma factory para filtros:

```text
src/deepdetector/filters/factory.py
```

Criar configs específicas:

```text
configs/experiments/table_7_smoothing_filter_search.yaml
configs/experiments/table_8_smoothing_filter_validation.yaml
```

Opcionalmente, criar wrappers finos:

```text
scripts/experiments/table_7.py
scripts/experiments/table_8.py
```

Esses scripts não devem conter lógica experimental complexa. Eles devem apenas carregar config e chamar o runner.

---

## Estrutura proposta

```text
src/deepdetector/
  experiments/
    __init__.py
    fgsm_context.py
    filter_candidate_runner.py

  filters/
    factory.py

configs/
  experiments/
    table_7_smoothing_filter_search.yaml
    table_8_smoothing_filter_validation.yaml

scripts/
  experiments/
    table_7.py
    table_8.py
```

---

# 1. Contexto FGSM reutilizável

Criar:

```text
src/deepdetector/experiments/fgsm_context.py
```

Com uma estrutura para armazenar dados preparados:

```python
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FGSMEvaluationContext:
    graph: dict[str, Any]
    images: np.ndarray
    labels: np.ndarray
    adversarial_images: np.ndarray
    clean_predictions: np.ndarray
    adversarial_predictions: np.ndarray
```

E uma função:

```python
def prepare_fgsm_context(config: dict) -> FGSMEvaluationContext:
    """
    Load dataset, restore model, generate FGSM examples once,
    and compute clean/adversarial predictions.
    """
```

## Regras esperadas

* O FGSM deve ser gerado uma única vez por execução.
* As predições limpas devem ser calculadas uma única vez.
* As predições adversariais devem ser calculadas uma única vez.
* O contexto deve ser reutilizado por todos os filtros candidatos.
* A sessão/grafo deve ser fechada adequadamente ao final da execução.

---

# 2. Factory de filtros

Criar:

```text
src/deepdetector/filters/factory.py
```

Com função:

```python
from typing import Callable

import numpy as np

FilterFn = Callable[[np.ndarray], np.ndarray]


def build_filter_from_config(config: dict) -> tuple[str, FilterFn, dict]:
    """
    Build one filter from a YAML config.

    Returns:
        filter_name: stable name used in outputs
        filter_fn: callable image -> filtered_image
        metadata: normalized filter metadata
    """
```

A factory deve suportar inicialmente:

* `cross_mean`
* `diamond_mean`
* `box_mean`

## Exemplo de entrada

```yaml
name: cross_7x7
type: cross_mean
radius: 3
```

## Exemplo de saída normalizada

```json
{
  "filter_name": "cross_7x7",
  "filter_type": "cross_mean",
  "mask_type": "cross",
  "mask_size": 7,
  "radius": 3
}
```

---

## Regras para cada filtro

### Cross mean

```yaml
name: cross_7x7
type: cross_mean
radius: 3
```

Deve chamar:

```python
cross_mean_filter(image, radius=3)
```

E normalizar:

```text
mask_size = 2 * radius + 1
mask_type = cross
```

### Diamond mean

```yaml
name: diamond_5x5
type: diamond_mean
radius: 2
```

Deve chamar:

```python
diamond_mean_filter(image, radius=2)
```

E normalizar:

```text
mask_size = 2 * radius + 1
mask_type = diamond
```

### Box mean

```yaml
name: box_5x5
type: box_mean
kernel_size: 5
```

Deve chamar:

```python
box_mean_filter(image, kernel_size=5)
```

E normalizar:

```text
mask_size = kernel_size
mask_type = box
```

---

## Validações

* `radius` deve ser inteiro positivo.
* `kernel_size` deve ser inteiro positivo e ímpar.
* `type` desconhecido deve gerar `ValueError`.
* `name` ausente deve gerar `ValueError`.

---

# 3. Runner genérico de candidatos

Criar:

```text
src/deepdetector/experiments/filter_candidate_runner.py
```

Com função principal:

```python
def run_filter_candidate_experiment(config: dict) -> list[dict]:
    """
    Run a filter candidate evaluation experiment.

    Used by Table 7 and Table 8.
    """
```

## Fluxo esperado

```python
def run_filter_candidate_experiment(config):
    context = prepare_fgsm_context(config)

    rows = []

    try:
        for filter_config in config["filters"]:
            filter_name, filter_fn, filter_metadata = build_filter_from_config(filter_config)

            metrics = evaluate_filter_on_existing_adversarial(
                graph=context.graph,
                images=context.images,
                labels=context.labels,
                adv_images=context.adversarial_images,
                clean_pred=context.clean_predictions,
                adv_pred=context.adversarial_predictions,
                filter_fn=filter_fn,
                exclude_invalid_pairs=config["evaluation"]["exclude_invalid_pairs"],
            )

            rows.append({
                **filter_metadata,
                **metrics,
            })

        write outputs...

    finally:
        close graph/session...
```

## Regras esperadas

* O runner não deve saber se está rodando Table 7 ou Table 8.
* A identidade da tabela vem do YAML, via `experiment_id`.
* O runner deve avaliar qualquer lista de filtros configurada.
* O runner deve gerar apenas CSV e JSON.
* O runner não deve gerar Markdown.
* O runner não deve conter valores hardcoded do artigo.
* O runner deve reaproveitar `evaluate_filter_on_existing_adversarial`.

---

# 4. Config da Table 7

Criar:

```text
configs/experiments/table_7_smoothing_filter_search.yaml
```

## Exemplo sugerido

```yaml
experiment_id: table_7_smoothing_filter_search

dataset:
  name: mnist
  split: test
  start: 0
  end: 4500
  high_entropy_only: true
  entropy_threshold:
    min: 5.0

model:
  name: mnist_m1
  checkpoint_dir: artifacts/models/mnist/m1/clean_baseline/checkpoints

attack:
  name: fgsm
  epsilon: 0.2
  clip_min: 0.0
  clip_max: 1.0

base_filter:
  name: scalar_quantization
  intervals: 6

filters:
  - name: cross_3x3
    type: cross_mean
    radius: 1

  - name: cross_5x5
    type: cross_mean
    radius: 2

  - name: cross_7x7
    type: cross_mean
    radius: 3

  - name: cross_9x9
    type: cross_mean
    radius: 4

  - name: diamond_3x3
    type: diamond_mean
    radius: 1

  - name: diamond_5x5
    type: diamond_mean
    radius: 2

  - name: diamond_7x7
    type: diamond_mean
    radius: 3

  - name: diamond_9x9
    type: diamond_mean
    radius: 4

  - name: box_3x3
    type: box_mean
    kernel_size: 3

  - name: box_5x5
    type: box_mean
    kernel_size: 5

  - name: box_7x7
    type: box_mean
    kernel_size: 7

  - name: box_9x9
    type: box_mean
    kernel_size: 9

evaluation:
  exclude_invalid_pairs: false
  batch_size: 256

output:
  dir: results/experiments/table_7_smoothing_filter_search
  csv: metrics.csv
  json: metrics.json
```

---

# 5. Config da Table 8

Criar:

```text
configs/experiments/table_8_smoothing_filter_validation.yaml
```

## Exemplo sugerido

```yaml
experiment_id: table_8_smoothing_filter_validation

dataset:
  name: mnist
  split: test
  start: 4500
  end: 5500
  high_entropy_only: true
  entropy_threshold:
    min: 5.0

model:
  name: mnist_m1
  checkpoint_dir: artifacts/models/mnist/m1/clean_baseline/checkpoints

attack:
  name: fgsm
  epsilon: 0.2
  clip_min: 0.0
  clip_max: 1.0

base_filter:
  name: scalar_quantization
  intervals: 6

filters:
  - name: cross_5x5
    type: cross_mean
    radius: 2

  - name: cross_7x7
    type: cross_mean
    radius: 3

  - name: diamond_5x5
    type: diamond_mean
    radius: 2

  - name: diamond_7x7
    type: diamond_mean
    radius: 3

  - name: box_5x5
    type: box_mean
    kernel_size: 5

evaluation:
  exclude_invalid_pairs: false
  batch_size: 256

output:
  dir: results/experiments/table_8_smoothing_filter_validation
  csv: metrics.csv
  json: metrics.json
```

---

## Observação sobre dataset e alta entropia

No artigo, as Tabelas 7 e 8 focam nos samples de alta entropia, isto é, aqueles com entropia maior que `5.0`.

Portanto, o runner precisa suportar:

```yaml
high_entropy_only: true
entropy_threshold:
  min: 5.0
```

A filtragem por entropia deve ocorrer antes da avaliação dos filtros.

## Fluxo esperado

1. Carregar slice.
2. Calcular entropia das imagens.
3. Filtrar apenas imagens com `entropy > 5.0`.
4. Gerar FGSM para esse subconjunto.
5. Avaliar candidatos.

A função responsável por preparar o contexto deve aplicar esse filtro se configurado.

---

# Contrato de saída

## CSV da Table 7 e Table 8

Formato mínimo:

```csv
filter_name,filter_type,mask_type,mask_size,radius,kernel_size,TP,FN,FP,recall_percent,precision_percent,f1_percent
cross_3x3,cross_mean,cross,3,1,,0,0,0,0.0,0.0,0.0
box_5x5,box_mean,box,5,,5,0,0,0,0.0,0.0,0.0
```

## Campos obrigatórios

* `filter_name`
* `filter_type`
* `mask_type`
* `mask_size`
* `radius`
* `kernel_size`
* `TP`
* `FN`
* `FP`
* `recall_percent`
* `precision_percent`
* `f1_percent`

---

## JSON da Table 7 e Table 8

Formato mínimo:

```json
{
  "experiment_id": "table_7_smoothing_filter_search",
  "config": {},
  "metrics": [
    {
      "filter_name": "cross_7x7",
      "filter_type": "cross_mean",
      "mask_type": "cross",
      "mask_size": 7,
      "radius": 3,
      "kernel_size": null,
      "TP": 0,
      "FN": 0,
      "FP": 0,
      "recall_percent": 0.0,
      "precision_percent": 0.0,
      "f1_percent": 0.0
    }
  ],
  "extra": {
    "num_filters": 12,
    "selection_stage": "search"
  }
}
```

---

## Como a Table 8 prepara a Table 9

A Table 8 deve permitir identificar o melhor filtro candidato a partir de `f1_percent`.

A seleção automática do melhor filtro ainda pode ficar fora de escopo, mas o JSON deve conter informação suficiente para a próxima etapa.

## Exemplo de `extra` opcional

```json
{
  "best_filter_by_f1": {
    "filter_name": "cross_7x7",
    "filter_type": "cross_mean",
    "mask_type": "cross",
    "mask_size": 7,
    "radius": 3,
    "f1_percent": 95.54
  }
}
```

Se a seleção automática for implementada nesta spec, ela deve ser simples:

```python
best_filter = max(rows, key=lambda row: row["f1_percent"])
```

Mas a Table 9 ainda será refatorada em spec própria.

---

# Mudanças esperadas em scripts

Criar scripts finos:

```text
scripts/experiments/table_7.py
scripts/experiments/table_8.py
```

## Exemplo

```python
from deepdetector.io.config import load_yaml_config
from deepdetector.io.paths import resolve_project_path
from deepdetector.experiments.filter_candidate_runner import run_filter_candidate_experiment


def main() -> int:
    config_path = resolve_project_path("configs/experiments/table_7_smoothing_filter_search.yaml")
    config = load_yaml_config(config_path)
    run_filter_candidate_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Esses scripts não devem conter:

* lógica de carregamento manual do modelo;
* geração FGSM;
* cálculo de métricas;
* escrita manual de CSV;
* escrita de Markdown;
* valores fixos do artigo.

---

# Critérios de aceitação

## Runner

* Existe `src/deepdetector/experiments/fgsm_context.py`.
* Existe `src/deepdetector/experiments/filter_candidate_runner.py`.
* `prepare_fgsm_context` gera FGSM uma única vez por execução.
* `run_filter_candidate_experiment` avalia uma lista de filtros vinda do YAML.
* O runner usa `evaluate_filter_on_existing_adversarial`.
* O runner gera apenas `metrics.csv` e `metrics.json`.
* O runner não gera Markdown.
* O runner não contém valores hardcoded do artigo.

## Filtros

* Existe `src/deepdetector/filters/factory.py`.
* A factory suporta `cross_mean`.
* A factory suporta `diamond_mean`.
* A factory suporta `box_mean`.
* A factory normaliza `filter_name`.
* A factory normaliza `filter_type`.
* A factory normaliza `mask_type`.
* A factory normaliza `mask_size`.
* A factory valida `radius`.
* A factory valida `kernel_size`.
* Tipo de filtro desconhecido gera `ValueError`.

## Configs

* Existe `configs/experiments/table_7_smoothing_filter_search.yaml`.
* Existe `configs/experiments/table_8_smoothing_filter_validation.yaml`.
* Table 7 define os 12 candidatos: `cross`, `diamond` e `box` em `3x3`, `5x5`, `7x7` e `9x9`.
* Table 8 define os 5 candidatos superiores.
* As configs usam `output.csv: metrics.csv`.
* As configs usam `output.json: metrics.json`.

## Outputs

* CSV contém `filter_name`.
* CSV contém `filter_type`.
* CSV contém `mask_type`.
* CSV contém `mask_size`.
* CSV contém `TP`, `FN`, `FP`.
* CSV contém `recall_percent`, `precision_percent`, `f1_percent`.
* JSON contém `experiment_id`.
* JSON contém `config`.
* JSON contém `metrics`.
* JSON contém `extra`.
* Nenhum `.md` é criado.

## Testes

* Teste da factory para `cross_mean`.
* Teste da factory para `diamond_mean`.
* Teste da factory para `box_mean`.
* Teste da factory com tipo inválido.
* Teste da factory com `radius` inválido.
* Teste da factory com `kernel_size` par.
* Teste do runner com dois filtros mockados.
* Teste garantindo que FGSM/contexto é preparado uma única vez.
* Teste garantindo que outputs CSV/JSON são criados.
* Teste garantindo que nenhum Markdown é criado.

## Sugestão de estrutura de testes

```text
tests/
  filters/
    test_filter_factory.py

  experiments/
    test_fgsm_context.py
    test_filter_candidate_runner.py
```

---

# Riscos

## Risco 1 — Regerar FGSM para cada filtro

Se o runner for implementado de forma ingênua, pode acabar gerando adversariais dentro do loop de filtros.

### Mitigação

* Criar `prepare_fgsm_context`.
* Testar que o contexto é preparado uma única vez.
* Usar `evaluate_filter_on_existing_adversarial`.

---

## Risco 2 — Misturar Table 7 e Table 8 com lógica especial

Se o código tiver `if table_7` ou `if table_8`, a duplicação volta.

### Mitigação

* O runner não deve conhecer número da tabela.
* A diferença deve estar apenas no YAML.

---

## Risco 3 — Ambiguidade entre radius e mask_size

Filtros `cross` e `diamond` usam `radius`; filtro `box` usa `kernel_size`.

### Mitigação

* Normalizar tudo para `mask_size`.
* Manter `radius` e `kernel_size` no output para rastreabilidade.
* Validar que `mask_size = 2 * radius + 1` quando aplicável.

---

## Risco 4 — Filtro de alta entropia alterar amostras avaliadas

As Tabelas 7 e 8 dependem de avaliar high-entropy samples. Se isso não for aplicado corretamente, os resultados podem divergir.

### Mitigação

* Colocar `high_entropy_only` explicitamente no YAML.
* Registrar no JSON quantas imagens foram mantidas após o filtro de entropia.
* Testar a filtragem de entropia com dados sintéticos.

---

# Plano de implementação sugerido

1. Criar `src/deepdetector/filters/factory.py`.
2. Implementar suporte a `cross_mean`, `diamond_mean` e `box_mean`.
3. Criar testes da factory.
4. Criar `src/deepdetector/experiments/fgsm_context.py`.
5. Implementar `prepare_fgsm_context`.
6. Adicionar suporte a `high_entropy_only`.
7. Criar `src/deepdetector/experiments/filter_candidate_runner.py`.
8. Implementar loop genérico de filtros candidatos.
9. Integrar escrita de outputs CSV/JSON usando a infraestrutura da Spec 1.
10. Criar YAML da Table 7.
11. Criar YAML da Table 8.
12. Criar scripts finos para Table 7 e Table 8.
13. Criar testes do runner.
14. Validar que os outputs não incluem Markdown.
15. Validar que Table 7 e Table 8 usam o mesmo runner.

---

# Definição de pronto

A spec é considerada concluída quando:

* Table 7 e Table 8 usam o mesmo runner genérico;
* os filtros são definidos por YAML;
* `cross`, `diamond` e `box` são suportados;
* o FGSM é gerado uma única vez por execução;
* os resultados são gerados apenas em CSV e JSON;
* nenhum Markdown ou diagnóstico é criado;
* os testes cobrem factory, runner, outputs e validações principais;
* o resultado da Table 8 contém dados suficientes para justificar o filtro final usado na Table 9.
