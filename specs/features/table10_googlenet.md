````markdown
# Spec — Implementar Table 10 / ImageNet / GoogLeNet

## 1. Objetivo

Implementar o grupo `googlenet` da **Table 10**, referente aos experimentos em **ImageNet** com o modelo **GoogLeNet**.

Este grupo contém as linhas:

| No. | Attack/Model | Dataset |
|---:|---|---|
| 5 | `FGSM (ε=1/255)/GoogLeNet` | ImageNet |
| 6 | `FGSM (ε=2/255)/GoogLeNet` | ImageNet |
| 7 | `DeepFool/GoogLeNet` | ImageNet |

A implementação deve seguir boas práticas:

- evitar criar lógica repetida;
- reutilizar módulos existentes de métricas, outputs, filtros, ataques, loaders e runners;
- não criar `manifest.json`;
- não criar scripts paralelos específicos para GoogLeNet;
- não fazer referência a resultados antigos;
- manter o schema da Table 10 compatível com o artigo.

---

## 2. Estado atual esperado

O experimento já deve existir no `configs/experiments.yaml` como grupo da Table 10.

A configuração atual esperada é equivalente a:

```yaml
table_10_googlenet:
  kind: table_10_group
  output_dir: results/experiments/table_10/googlenet
  dataset:
    name: imagenet
  model:
    name: googlenet
  attack: {}
  evaluation: {}
  model_group: googlenet
  dataset_label: ImageNet
  rows:
    - "no": 5
      attack_model: "FGSM (ε=1/255)/GoogLeNet"
      status: planned
      attack:
        name: fgsm
        epsilon: 0.00392156862745098

    - "no": 6
      attack_model: "FGSM (ε=2/255)/GoogLeNet"
      status: planned
      attack:
        name: fgsm
        epsilon: 0.00784313725490196

    - "no": 7
      attack_model: "DeepFool/GoogLeNet"
      status: planned
      attack:
        name: deepfool
````

Antes da implementação, ajustar o `output_dir` para seguir a organização por dataset/modelo:

```yaml
output_dir: results/experiments/table_10/imagenet/googlenet
```

---

## 3. Escopo

Esta implementação cobre:

* registro e execução do grupo `table_10_googlenet`;
* geração de `metrics.csv`;
* geração de `metrics.json`;
* uso do schema oficial da Table 10;
* integração com o runner/CLI existente;
* preparação do fluxo para as linhas 5, 6 e 7;
* suporte estrutural para resultados reais quando os ataques estiverem implementados.

Nesta etapa, o foco é implementar o **fluxo do grupo GoogLeNet** de forma limpa e integrada, aproveitando o que já existe.

---

## 4. Fora do escopo

Não implementar nesta spec:

* `manifest.json`;
* CaffeNet;
* Inception v3;
* agregado geral da Table 10;
* scripts novos dentro de `scripts/`;
* relatórios `.md`;
* diagnóstico extra;
* referência a resultados antigos;
* cópia de lógica já existente de CSV/JSON, métricas ou filtros.

---

## 5. Saídas esperadas

O comando:

```bash
python scripts/run_experiment.py --experiment table_10_googlenet
```

deve gerar:

```text
results/experiments/table_10/imagenet/googlenet/
├── metrics.csv
└── metrics.json
```

Não deve gerar:

```text
manifest.json
summary.md
diagnostics.md
report.md
```

---

## 6. Schema oficial do grupo GoogLeNet

O `metrics.csv` deve seguir exatamente o schema:

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
```

Mapeamento com a Table 10 do artigo:

| Campo do artigo | Campo no CSV/JSON |
| --------------- | ----------------- |
| `No.`           | `no`              |
| `Attack/Model`  | `attack_model`    |
| `Dataset`       | `dataset`         |
| `#F`            | `num_failures`    |
| `TP`            | `tp`              |
| `FN`            | `fn`              |
| `FP`            | `fp`              |
| `RTP`           | `rtp`             |
| `RTP%`          | `rtp_percent`     |
| `Recall`        | `recall`          |
| `Precision`     | `precision`       |
| `F1`            | `f1`              |

---

## 7. Conteúdo inicial esperado

Enquanto as linhas ainda não forem computadas experimentalmente, os campos métricos devem ficar vazios no CSV e `null` no JSON.

### 7.1. `metrics.csv`

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
5,FGSM (ε=1/255)/GoogLeNet,ImageNet,,,,,,,,,
6,FGSM (ε=2/255)/GoogLeNet,ImageNet,,,,,,,,,
7,DeepFool/GoogLeNet,ImageNet,,,,,,,,,
```

### 7.2. `metrics.json`

```json
{
  "table": 10,
  "dataset_group": "imagenet",
  "model_group": "googlenet",
  "rows": [
    {
      "no": 5,
      "attack_model": "FGSM (ε=1/255)/GoogLeNet",
      "dataset": "ImageNet",
      "num_failures": null,
      "tp": null,
      "fn": null,
      "fp": null,
      "rtp": null,
      "rtp_percent": null,
      "recall": null,
      "precision": null,
      "f1": null
    },
    {
      "no": 6,
      "attack_model": "FGSM (ε=2/255)/GoogLeNet",
      "dataset": "ImageNet",
      "num_failures": null,
      "tp": null,
      "fn": null,
      "fp": null,
      "rtp": null,
      "rtp_percent": null,
      "recall": null,
      "precision": null,
      "f1": null
    },
    {
      "no": 7,
      "attack_model": "DeepFool/GoogLeNet",
      "dataset": "ImageNet",
      "num_failures": null,
      "tp": null,
      "fn": null,
      "fp": null,
      "rtp": null,
      "rtp_percent": null,
      "recall": null,
      "precision": null,
      "f1": null
    }
  ]
}
```

---

## 8. Regras de implementação

## 8.1. Não duplicar lógica de escrita de arquivos

Antes de criar qualquer função nova para escrever CSV/JSON, verificar se já existe módulo equivalente, por exemplo:

```text
src/deepdetector/evaluation/outputs.py
```

ou equivalente no projeto.

A implementação da Table 10 deve reutilizar funções existentes de escrita sempre que possível.

Se o projeto já tiver algo como:

```python
write_metrics_csv(...)
write_metrics_json(...)
save_metrics(...)
```

usar essas funções.

Criar funções novas somente se não existir alternativa reutilizável.

---

## 8.2. Não duplicar lógica de métricas

A Table 10 usa métricas já presentes em outros experimentos:

```text
TP
FN
FP
RTP
Recall
Precision
F1
```

Se já existir módulo como:

```text
src/deepdetector/evaluation/metrics.py
```

ou função equivalente, reaproveitar.

A implementação do grupo GoogLeNet não deve recalcular fórmulas manualmente se já existir função comum.

---

## 8.3. Não duplicar lógica de filtros

Quando a execução real for conectada, o grupo GoogLeNet deve usar o filtro já existente no projeto.

Não implementar novamente:

* entropia;
* quantização;
* smoothing;
* filtro proposto;
* comparação `C(x)` vs `C(T(x))`.

Essas responsabilidades devem permanecer nos módulos já existentes de `filters/` ou equivalentes.

---

## 8.4. Não criar scripts novos

Não criar arquivos como:

```text
scripts/table_10_googlenet.py
scripts/imagenet_googlenet_table10.py
scripts/run_table10_googlenet.py
```

A execução deve ser feita apenas pelo CLI atual:

```bash
python scripts/run_experiment.py --experiment table_10_googlenet
```

---

## 8.5. Não criar `manifest.json`

O grupo deve gerar somente:

```text
metrics.csv
metrics.json
```

Não gerar:

```text
manifest.json
```

---

## 9. Estrutura recomendada de código

A implementação deve encaixar a Table 10 no mesmo padrão das demais tabelas.

Estrutura sugerida:

```text
src/deepdetector/
└── evaluation/
    ├── metrics.py
    ├── outputs.py
    ├── runner.py
    └── tables/
        ├── table_6.py
        ├── table_9.py
        └── table_10.py
```

Se a pasta `tables/` já existir, usar ela.

Se o projeto já tiver outro padrão para runners de tabelas, seguir o padrão atual em vez de criar um novo.

---

## 10. Módulo `table_10.py`

Criar ou atualizar:

```text
src/deepdetector/evaluation/tables/table_10.py
```

Responsabilidade do módulo:

* receber a configuração do grupo `table_10_googlenet`;
* montar as linhas no schema da Table 10;
* despachar execução real quando uma linha estiver implementada;
* manter linhas planejadas com métricas vazias;
* salvar `metrics.csv` e `metrics.json`;
* reutilizar funções comuns de outputs.

---

## 11. API sugerida

A API abaixo é uma sugestão. Adaptar ao padrão já existente no projeto.

```python
from __future__ import annotations

from typing import Any


TABLE_10_SCHEMA = [
    "no",
    "attack_model",
    "dataset",
    "num_failures",
    "tp",
    "fn",
    "fp",
    "rtp",
    "rtp_percent",
    "recall",
    "precision",
    "f1",
]


def run_table_10_group(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Executa ou materializa um grupo da Table 10.

    Para linhas com status planned ou blocked, gera linha com métricas vazias.
    Para linhas com status implemented, deve chamar o executor real da linha,
    quando disponível.

    Retorna uma lista de linhas no schema oficial da Table 10.
    """
```

---

## 12. Função para construir linha vazia

Criar uma função pequena, pura e testável:

```python
def build_pending_table_10_row(
    *,
    no: int,
    attack_model: str,
    dataset: str,
) -> dict[str, Any]:
    return {
        "no": no,
        "attack_model": attack_model,
        "dataset": dataset,
        "num_failures": None,
        "tp": None,
        "fn": None,
        "fp": None,
        "rtp": None,
        "rtp_percent": None,
        "recall": None,
        "precision": None,
        "f1": None,
    }
```

Essa função evita repetição e facilita testes.

---

## 13. Função para normalizar resultados computados

Quando a execução real de uma linha for implementada, o resultado deve ser normalizado para o schema oficial.

```python
def normalize_table_10_result(
    *,
    no: int,
    attack_model: str,
    dataset: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Converte o resultado bruto de uma execução para o schema oficial da Table 10.
    """
```

Exemplo de saída:

```python
{
    "no": 5,
    "attack_model": "FGSM (ε=1/255)/GoogLeNet",
    "dataset": "ImageNet",
    "num_failures": 270,
    "tp": 841,
    "fn": 98,
    "fp": 88,
    "rtp": 718,
    "rtp_percent": 85.37,
    "recall": 89.56,
    "precision": 90.53,
    "f1": 90.04,
}
```

---

## 14. Fluxo do `run_table_10_group`

Pseudocódigo:

```python
def run_table_10_group(config):
    rows = []

    for row_config in config["rows"]:
        if row_config["status"] != "implemented":
            rows.append(
                build_pending_table_10_row(
                    no=row_config["no"],
                    attack_model=row_config["attack_model"],
                    dataset=config["dataset_label"],
                )
            )
            continue

        raw_result = run_table_10_row(
            group_config=config,
            row_config=row_config,
        )

        rows.append(
            normalize_table_10_result(
                no=row_config["no"],
                attack_model=row_config["attack_model"],
                dataset=config["dataset_label"],
                result=raw_result,
            )
        )

    save_table_10_outputs(
        rows=rows,
        output_dir=config["output_dir"],
    )

    return rows
```

---

## 15. Função `save_table_10_outputs`

Antes de criar essa função, verificar se já existe função genérica em `outputs.py`.

Preferência:

```python
save_metrics_csv(rows, path, schema=TABLE_10_SCHEMA)
save_metrics_json(payload, path)
```

Caso não exista, criar algo genérico no módulo comum de outputs, e não dentro do `table_10.py`.

Exemplo aceitável em módulo comum:

```python
def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ...


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ...
```

Então `table_10.py` usa:

```python
write_csv_rows(csv_path, rows, TABLE_10_SCHEMA)
write_json(json_path, payload)
```

Isso evita duplicação futura.

---

## 16. Integração com o runner existente

Localizar o ponto em que os experimentos são despachados por `kind`.

Adicionar suporte para:

```yaml
kind: table_10_group
```

Exemplo:

```python
if experiment_config["kind"] == "table_10_group":
    return run_table_10_group(experiment_config)
```

Não criar um fluxo separado só para `googlenet`.

O mesmo runner deve servir futuramente para:

```text
table_10_mnist_m1
table_10_mnist_m2
table_10_googlenet
table_10_caffenet
table_10_inception_v3
```

---

## 17. Ajuste de configuração

Atualizar apenas o necessário no grupo `table_10_googlenet`.

De:

```yaml
output_dir: results/experiments/table_10/googlenet
```

Para:

```yaml
output_dir: results/experiments/table_10/imagenet/googlenet
```

Manter:

```yaml
kind: table_10_group
model_group: googlenet
dataset_label: ImageNet
```

As linhas continuam:

```yaml
rows:
  - "no": 5
    attack_model: "FGSM (ε=1/255)/GoogLeNet"
    status: planned
    attack:
      name: fgsm
      epsilon: 0.00392156862745098

  - "no": 6
    attack_model: "FGSM (ε=2/255)/GoogLeNet"
    status: planned
    attack:
      name: fgsm
      epsilon: 0.00784313725490196

  - "no": 7
    attack_model: "DeepFool/GoogLeNet"
    status: planned
    attack:
      name: deepfool
```

---

## 18. Comando esperado

```bash
python scripts/run_experiment.py --experiment table_10_googlenet
```

Resultado esperado:

```text
results/experiments/table_10/imagenet/googlenet/metrics.csv
results/experiments/table_10/imagenet/googlenet/metrics.json
```

---

## 19. Testes

## 19.1. Teste do schema

Criar ou atualizar teste:

```text
tests/test_table_10_group.py
```

Teste esperado:

```python
from deepdetector.evaluation.tables.table_10 import TABLE_10_SCHEMA


def test_table_10_schema_matches_paper_fields():
    assert TABLE_10_SCHEMA == [
        "no",
        "attack_model",
        "dataset",
        "num_failures",
        "tp",
        "fn",
        "fp",
        "rtp",
        "rtp_percent",
        "recall",
        "precision",
        "f1",
    ]
```

---

## 19.2. Teste de linha pendente

```python
from deepdetector.evaluation.tables.table_10 import build_pending_table_10_row


def test_build_pending_table_10_row():
    row = build_pending_table_10_row(
        no=5,
        attack_model="FGSM (ε=1/255)/GoogLeNet",
        dataset="ImageNet",
    )

    assert row == {
        "no": 5,
        "attack_model": "FGSM (ε=1/255)/GoogLeNet",
        "dataset": "ImageNet",
        "num_failures": None,
        "tp": None,
        "fn": None,
        "fp": None,
        "rtp": None,
        "rtp_percent": None,
        "recall": None,
        "precision": None,
        "f1": None,
    }
```

---

## 19.3. Teste do grupo GoogLeNet

```python
from deepdetector.evaluation.tables.table_10 import run_table_10_group


def test_table_10_googlenet_group_generates_three_rows(tmp_path):
    config = {
        "kind": "table_10_group",
        "output_dir": str(tmp_path),
        "dataset": {"name": "imagenet"},
        "model": {"name": "googlenet"},
        "model_group": "googlenet",
        "dataset_label": "ImageNet",
        "rows": [
            {
                "no": 5,
                "attack_model": "FGSM (ε=1/255)/GoogLeNet",
                "status": "planned",
                "attack": {"name": "fgsm", "epsilon": 1 / 255},
            },
            {
                "no": 6,
                "attack_model": "FGSM (ε=2/255)/GoogLeNet",
                "status": "planned",
                "attack": {"name": "fgsm", "epsilon": 2 / 255},
            },
            {
                "no": 7,
                "attack_model": "DeepFool/GoogLeNet",
                "status": "planned",
                "attack": {"name": "deepfool"},
            },
        ],
    }

    rows = run_table_10_group(config)

    assert [row["no"] for row in rows] == [5, 6, 7]
    assert all(row["dataset"] == "ImageNet" for row in rows)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "manifest.json").exists()
```

---

## 19.4. Teste de integração do `kind`

Criar ou atualizar teste do runner geral para garantir que `kind: table_10_group` é roteado corretamente.

Exemplo conceitual:

```python
def test_runner_dispatches_table_10_group():
    ...
```

O teste deve verificar que o runner chama `run_table_10_group` para configs com:

```yaml
kind: table_10_group
```

---

## 20. Critérios de aceite

* [ ] `table_10_googlenet` usa `output_dir: results/experiments/table_10/imagenet/googlenet`.
* [ ] O comando `python scripts/run_experiment.py --experiment table_10_googlenet` executa sem erro.
* [ ] O grupo gera `metrics.csv`.
* [ ] O grupo gera `metrics.json`.
* [ ] O grupo não gera `manifest.json`.
* [ ] O `metrics.csv` segue o schema oficial da Table 10.
* [ ] O `metrics.json` contém `table`, `dataset_group`, `model_group` e `rows`.
* [ ] As linhas geradas são exatamente `No. 5`, `No. 6` e `No. 7`.
* [ ] Campos métricos de linhas `planned` ficam vazios no CSV.
* [ ] Campos métricos de linhas `planned` ficam `null` no JSON.
* [ ] A implementação reutiliza funções existentes de output sempre que possível.
* [ ] A implementação não duplica cálculo de métricas.
* [ ] A implementação não duplica lógica de filtros.
* [ ] A implementação não cria scripts novos.
* [ ] A implementação não usa resultados antigos.
* [ ] A implementação cria um runner genérico para grupos da Table 10, não algo exclusivo para GoogLeNet.

---

## 21. Definition of Done

Esta etapa estará concluída quando:

1. `table_10_googlenet` estiver configurado com o diretório:

```text
results/experiments/table_10/imagenet/googlenet
```

2. existir suporte a `kind: table_10_group` no runner;

3. o módulo da Table 10 conseguir materializar as linhas planejadas;

4. `metrics.csv` e `metrics.json` forem gerados corretamente;

5. nenhum `manifest.json` for gerado;

6. os testes da Table 10 passarem;

7. a implementação puder ser reutilizada futuramente para:

```text
table_10_mnist_m1
table_10_mnist_m2
table_10_caffenet
table_10_inception_v3
```

sem duplicar lógica.

```
```
