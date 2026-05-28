# Spec 3 — Refatorar Tabelas 6 e 9 para runner único de avaliação FGSM por split

## Status

Proposto.

## Contexto

As Tabelas 6 e 9 possuem fluxo experimental muito semelhante.

A diferença conceitual entre elas é o filtro avaliado:

```text
Table 6 -> adaptive quantization
Table 9 -> proposed detection filter
```

A Tabela 6 avalia a estratégia de quantização adaptativa em **Training** e **Validation**.

A Tabela 9 avalia o filtro final proposto, combinando quantização adaptativa e suavização espacial, também em **Training** e **Validation**.

Portanto, as duas tabelas devem compartilhar o mesmo runner experimental. A diferença deve estar apenas na configuração do filtro.

---

## Problema

Hoje a tendência é cada tabela possuir um script próprio com lógica repetida:

* carregar YAML;
* resolver paths;
* carregar slices;
* restaurar modelo;
* gerar FGSM;
* aplicar filtro;
* calcular métricas;
* escrever outputs.

Isso gera duplicação e aumenta o risco de divergência entre os experimentos.

Exemplo de risco:

* Table 6 calcula TP/FN/FP de uma forma;
* Table 9 calcula TP/FN/FP de outra forma.

Ou:

* Table 6 usa uma regra de slice;
* Table 9 usa outra regra sem perceber.

Como ambas avaliam filtros contra FGSM em splits definidos, elas devem usar um único fluxo.

---

## Objetivo

Criar um runner único para experimentos de avaliação FGSM por split, reutilizável pelas Tabelas 6 e 9.

Ao final desta spec:

* Table 6 e Table 9 usam o mesmo runner.
* O runner recebe os splits via YAML.
* O runner recebe o filtro via YAML.
* O runner gera apenas CSV e JSON.
* A lógica de cálculo de métricas permanece centralizada.
* A diferença entre Table 6 e Table 9 fica restrita ao filtro configurado.
* Não há geração de Markdown ou diagnóstico.

---

## Fora de escopo

Esta spec não deve:

* refatorar Table 7 e Table 8;
* alterar a implementação matemática dos filtros;
* alterar a geração FGSM;
* alterar o modelo base;
* alterar a fórmula de TP, FN, FP, recall, precision ou F1;
* limpar todo o YAML do projeto;
* remover todas as referências históricas ao artigo;
* implementar CLI final definitivo.

A limpeza completa de YAML, outputs antigos e referências ao artigo será tratada na Spec 4.

---

## Relação com specs anteriores

Esta spec assume que a Spec 1 já criou a infraestrutura comum:

* `deepdetector.io.paths`
* `deepdetector.io.config`
* `deepdetector.io.result_writers`
* `deepdetector.experiments.metadata`

Também assume que a Spec 2 já introduziu ou iniciou a ideia de factory de filtros:

* `deepdetector.filters.factory`

Caso a factory da Spec 2 ainda não exista, esta spec deve criar o mínimo necessário para suportar:

* `adaptive_quantization`
* `proposed_detection_filter`

---

## Fluxo esperado

O runner único deve executar o seguinte fluxo:

1. Ler config.
2. Resolver paths.
3. Restaurar modelo.
4. Para cada split configurado:

   1. Carregar imagens e labels.
   2. Gerar FGSM para aquele split.
   3. Aplicar filtro configurado.
   4. Calcular TP, FN, FP, recall, precision e F1.
   5. Adicionar linha ao resultado.
5. Fechar sessão/grafo.
6. Escrever `metrics.csv`.
7. Escrever `metrics.json`.

---

## Estrutura proposta

```text
src/deepdetector/
  experiments/
    split_runner.py
    fgsm_split_runner.py

  filters/
    factory.py

configs/
  experiments/
    table_6_adaptive_quantization.yaml
    table_9_proposed_filter.yaml

scripts/
  experiments/
    table_6.py
    table_9.py
```

---

# 1. Runner FGSM por split

Criar:

```text
src/deepdetector/experiments/fgsm_split_runner.py
```

Com função principal:

```python
from typing import Any, Dict, List


def run_fgsm_split_experiment(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run an FGSM filter evaluation experiment over configured dataset splits.

    Used by Table 6 and Table 9.
    """
```

## Fluxo sugerido

```python
def run_fgsm_split_experiment(config):
    graph = create_restored_mnist_graph(...)
    filter_name, filter_fn, filter_metadata = build_filter_from_config(config["filter"])
    rows = []

    try:
        for split in config["dataset"]["slices"]:
            images, labels = load_mnist_test_slice(split["start"], split["end"])

            metrics = evaluate_filter_on_images(
                graph=graph,
                images=images,
                labels=labels,
                epsilon=config["attack"]["epsilon"],
                filter_fn=filter_fn,
                clip_min=config["attack"]["clip_min"],
                clip_max=config["attack"]["clip_max"],
                exclude_invalid_pairs=config["evaluation"]["exclude_invalid_pairs"],
            )

            rows.append({
                "split": split["name"],
                **metrics,
            })

        write outputs...

    finally:
        close_graph(graph)
```

---

# 2. Responsabilidades do runner

O runner deve ser responsável por:

* iterar sobre os splits configurados;
* carregar dados de cada split;
* gerar FGSM para cada split;
* aplicar o filtro configurado;
* calcular métricas usando funções já existentes;
* montar rows padronizadas;
* escrever CSV e JSON via infraestrutura da Spec 1.

O runner não deve:

* saber se está executando Table 6 ou Table 9;
* conter valores fixos do artigo;
* gerar Markdown;
* gerar diagnóstico;
* definir filtros hardcoded por tabela;
* duplicar cálculo de métricas.

---

# 3. Factory de filtros

A factory de filtros deve suportar pelo menos dois filtros para esta spec:

* `adaptive_quantization`
* `proposed_detection_filter`

## Exemplo

```python
def build_filter_from_config(config):
    filter_type = config["type"]

    if filter_type == "adaptive_quantization":
        return (
            config["name"],
            adaptive_quantization_filter,
            {
                "filter_name": config["name"],
                "filter_type": "adaptive_quantization",
            },
        )

    if filter_type == "proposed_detection_filter":
        return (
            config["name"],
            proposed_detection_filter,
            {
                "filter_name": config["name"],
                "filter_type": "proposed_detection_filter",
            },
        )

    raise ValueError(f"Unknown filter type: {filter_type}")
```

## Observação

Nesta spec, a implementação interna de `proposed_detection_filter` pode continuar usando os valores atuais:

```text
entropy < 4.0 -> 2 intervals
4.0 <= entropy < 5.0 -> 4 intervals
entropy >= 5.0 -> 6 intervals + cross 7x7 smoothing
```

A parametrização completa desses valores no YAML pode ficar para a Spec 4, se necessário.

---

# 4. Config da Table 6

Criar:

```text
configs/experiments/table_6_adaptive_quantization.yaml
```

## Exemplo sugerido

```yaml
experiment_id: table_6_adaptive_quantization

dataset:
  name: mnist
  split: test
  slices:
    - name: Training
      start: 0
      end: 4500
    - name: Validation
      start: 4500
      end: 5500

model:
  name: mnist_m1
  checkpoint_dir: artifacts/models/mnist/m1/clean_baseline/checkpoints

attack:
  name: fgsm
  epsilon: 0.2
  clip_min: 0.0
  clip_max: 1.0

filter:
  name: adaptive_quantization
  type: adaptive_quantization

evaluation:
  exclude_invalid_pairs: false
  batch_size: 256

output:
  dir: results/experiments/table_6_adaptive_quantization
  csv: metrics.csv
  json: metrics.json
```

---

# 5. Config da Table 9

Criar:

```text
configs/experiments/table_9_proposed_filter.yaml
```

## Exemplo sugerido

```yaml
experiment_id: table_9_proposed_filter

dataset:
  name: mnist
  split: test
  slices:
    - name: Training
      start: 0
      end: 4500
    - name: Validation
      start: 4500
      end: 5500

model:
  name: mnist_m1
  checkpoint_dir: artifacts/models/mnist/m1/clean_baseline/checkpoints

attack:
  name: fgsm
  epsilon: 0.2
  clip_min: 0.0
  clip_max: 1.0

filter:
  name: proposed_detection_filter
  type: proposed_detection_filter

evaluation:
  exclude_invalid_pairs: false
  batch_size: 256

output:
  dir: results/experiments/table_9_proposed_filter
  csv: metrics.csv
  json: metrics.json
```

---

# 6. Contrato de saída

As Tabelas 6 e 9 devem gerar o mesmo schema de CSV.

## CSV

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
Training,0,0,0,0.0,0.0,0.0
Validation,0,0,0,0.0,0.0,0.0
```

## Campos obrigatórios

* `split`
* `TP`
* `FN`
* `FP`
* `recall_percent`
* `precision_percent`
* `f1_percent`

## Campos opcionais, se úteis

* `F`
* `RTP`
* `RTP_percent`
* `n_total`
* `clean_errors`

Para esta spec, o schema mínimo deve ser o contrato principal. Campos adicionais só devem entrar se forem úteis para rastreabilidade e estiverem também no JSON.

---

## JSON

Formato mínimo:

```json
{
  "experiment_id": "table_6_adaptive_quantization",
  "config": {},
  "metrics": [
    {
      "split": "Training",
      "TP": 0,
      "FN": 0,
      "FP": 0,
      "recall_percent": 0.0,
      "precision_percent": 0.0,
      "f1_percent": 0.0
    },
    {
      "split": "Validation",
      "TP": 0,
      "FN": 0,
      "FP": 0,
      "recall_percent": 0.0,
      "precision_percent": 0.0,
      "f1_percent": 0.0
    }
  ],
  "extra": {
    "filter": {
      "filter_name": "adaptive_quantization",
      "filter_type": "adaptive_quantization"
    }
  }
}
```

---

# 7. Scripts finos

Criar scripts finos para compatibilidade operacional:

```text
scripts/experiments/table_6.py
scripts/experiments/table_9.py
```

## Exemplo

```python
from deepdetector.io.config import load_yaml_config
from deepdetector.io.paths import resolve_project_path
from deepdetector.experiments.fgsm_split_runner import run_fgsm_split_experiment


def main() -> int:
    config_path = resolve_project_path(
        "configs/experiments/table_6_adaptive_quantization.yaml"
    )
    config = load_yaml_config(config_path)
    run_fgsm_split_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Esses scripts não devem conter:

* lógica de modelo;
* lógica de dataset;
* geração FGSM;
* cálculo de métricas;
* escrita manual de CSV;
* escrita de Markdown;
* constantes de comparação com artigo.

---

# 8. Compatibilidade temporária com scripts antigos

Durante a migração, scripts antigos podem permanecer como wrappers de compatibilidade.

Exemplo:

```text
scripts/article_reproduction/table_6.py
```

Pode virar apenas:

```python
from scripts.experiments.table_6 import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Ou pode ser marcado como deprecated.

Critério importante:

> A lógica experimental deve existir em um único lugar.

---

# 9. Integração futura com CLI

Esta spec não precisa implementar o CLI final, mas o runner deve ser compatível com um comando futuro como:

```bash
deepdetector reproduce table-6 --config configs/experiments/table_6_adaptive_quantization.yaml
deepdetector reproduce table-9 --config configs/experiments/table_9_proposed_filter.yaml
```

Portanto, o runner não deve depender de argumentos globais ou de paths hardcoded.

---

# Critérios de aceitação

## Runner

* Existe `src/deepdetector/experiments/fgsm_split_runner.py`.
* Existe `run_fgsm_split_experiment(config)`.
* O runner itera sobre `dataset.slices`.
* O runner carrega cada slice configurado.
* O runner restaura o modelo uma única vez por execução.
* O runner aplica o filtro configurado em `filter`.
* O runner calcula métricas usando função central existente.
* O runner fecha a sessão/grafo ao final.
* O runner gera apenas CSV e JSON.
* O runner não gera Markdown.
* O runner não gera diagnóstico.
* O runner não possui valores hardcoded do artigo.

## Filtros

* A factory suporta `adaptive_quantization`.
* A factory suporta `proposed_detection_filter`.
* Tipo de filtro desconhecido gera `ValueError`.
* A factory retorna `filter_name`.
* A factory retorna `filter_type`.
* A factory retorna `filter_fn`.
* A factory retorna metadata normalizada.

## Configs

* Existe `configs/experiments/table_6_adaptive_quantization.yaml`.
* Existe `configs/experiments/table_9_proposed_filter.yaml`.
* Table 6 usa `filter.type: adaptive_quantization`.
* Table 9 usa `filter.type: proposed_detection_filter`.
* Ambas usam os mesmos slices `Training` e `Validation`.
* Ambas usam `output.csv: metrics.csv`.
* Ambas usam `output.json: metrics.json`.

## Scripts

* Existe wrapper fino para Table 6.
* Existe wrapper fino para Table 9.
* Wrappers não implementam lógica experimental.
* Scripts antigos, se mantidos, apenas delegam para o runner novo.

## Outputs

* `metrics.csv` contém `split`.
* `metrics.csv` contém `TP`.
* `metrics.csv` contém `FN`.
* `metrics.csv` contém `FP`.
* `metrics.csv` contém `recall_percent`.
* `metrics.csv` contém `precision_percent`.
* `metrics.csv` contém `f1_percent`.
* `metrics.json` contém `experiment_id`.
* `metrics.json` contém `config`.
* `metrics.json` contém `metrics`.
* `metrics.json` contém metadata do filtro.
* Nenhum `.md` é criado.

## Testes

* Teste do runner com dois splits mockados.
* Teste garantindo que o modelo/grafo é criado uma única vez.
* Teste garantindo que o filtro é construído pela factory.
* Teste garantindo que `dataset.slices` ausente gera erro claro.
* Teste garantindo que filtro desconhecido gera erro claro.
* Teste garantindo que CSV e JSON são criados.
* Teste garantindo que nenhum Markdown é criado.
* Teste garantindo que Table 6 e Table 9 usam o mesmo runner.

---

## Sugestão de estrutura de testes

```text
tests/
  experiments/
    test_fgsm_split_runner.py

  filters/
    test_filter_factory_split_filters.py
```

---

# Riscos

## Risco 1 — Table 6 e Table 9 divergirem por configuração

Mesmo com runner único, configs diferentes podem causar divergência indesejada.

### Mitigação

* Criar teste comparando os campos comuns das duas configs.
* Garantir que slices, modelo e ataque sejam iguais.
* Permitir diferença apenas em `experiment_id`, `filter` e `output`.

---

## Risco 2 — Mudança acidental no cálculo das métricas

A refatoração não deve mudar TP, FN, FP, recall, precision ou F1.

### Mitigação

* Reutilizar `evaluate_filter_on_images`.
* Não reimplementar cálculo de métricas no runner.
* Testar com dados sintéticos ou mocks.

---

## Risco 3 — Fechamento incorreto da sessão TensorFlow

Como o runner restaura o modelo uma vez e itera sobre splits, é importante fechar o grafo mesmo se algum split falhar.

### Mitigação

* Usar `try/finally`.
* Testar caminho de erro com mock.

---

## Risco 4 — Manter lógica duplicada nos scripts antigos

Se os scripts antigos continuarem com lógica própria, a duplicação permanece.

### Mitigação

* Transformar scripts antigos em wrappers.
* Ou marcar como deprecated.
* Garantir que a lógica experimental exista apenas em `fgsm_split_runner.py`.

---

# Plano de implementação sugerido

1. Criar `src/deepdetector/experiments/fgsm_split_runner.py`.
2. Implementar `run_fgsm_split_experiment(config)`.
3. Adicionar suporte a `adaptive_quantization` na factory de filtros.
4. Adicionar suporte a `proposed_detection_filter` na factory de filtros.
5. Criar config `table_6_adaptive_quantization.yaml`.
6. Criar config `table_9_proposed_filter.yaml`.
7. Criar wrapper fino `scripts/experiments/table_6.py`.
8. Criar wrapper fino `scripts/experiments/table_9.py`.
9. Adaptar scripts antigos para delegarem ao novo runner ou marcá-los como deprecated.
10. Garantir saída `metrics.csv` e `metrics.json`.
11. Remover geração de Markdown nesses fluxos.
12. Criar testes do runner.
13. Criar testes da factory.
14. Validar que Table 6 e Table 9 usam o mesmo fluxo.

---

# Definição de pronto

A spec é considerada concluída quando:

* existe um runner único para experimentos FGSM por split;
* Table 6 e Table 9 usam esse runner;
* a diferença entre Table 6 e Table 9 está apenas no filtro configurado;
* os outputs são padronizados em CSV e JSON;
* nenhum Markdown ou diagnóstico é gerado;
* a lógica de cálculo de métricas permanece centralizada e inalterada;
* os testes cobrem runner, filtros, configs e outputs.
