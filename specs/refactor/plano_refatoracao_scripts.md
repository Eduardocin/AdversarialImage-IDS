# Plano de Refatoração — `scripts/article_reproduction`

## Contexto

Após análise dos scripts da pasta `scripts/article_reproduction`, foram identificados três problemas centrais:

1. **Redundância de código**: funções quase idênticas aparecem em mais de um arquivo.
2. **Nomes de arquivo inconsistentes**: há mistura de prefixos como `table_`, `generate_table_`, nomes genéricos e sufixos pouco padronizados.
3. **Acoplamento excessivo**: os scripts acumulam responsabilidades de CLI, I/O, YAML, resolução de paths, carregamento de modelos, geração/carregamento de adversariais e escrita de resultados.

Este plano foca em duas frentes:

1. Padronizar os nomes dos arquivos dentro de `scripts/article_reproduction`.
2. Extrair código duplicado para módulos reutilizáveis em `src/deepdetector/`.

---

## 1. Padronização de Nomes de Arquivo

### Convenção proposta

```text
<dataset>_table_<N>[_<variante>].py
```

### Regras adotadas

- O prefixo deve ser o dataset: `imagenet_` ou `mnist_`.
- O número da tabela deve ter dois dígitos: `04`, `07`, `08`, `10`.
- A variante é opcional e deve explicitar modelo, ataque ou cenário quando necessário.
- Evitar verbos no nome do arquivo, como `generate_`, quando o arquivo representa uma reprodução de tabela.
- Scripts utilitários de geração podem existir separadamente, mas não devem ser confundidos com scripts de reprodução de tabela.

### Renomeação proposta

| Arquivo atual | Arquivo proposto | Motivo |
|---|---|---|
| `table_4_imagenet.py` | `imagenet_table_04.py` | Dataset como prefixo e número com zero-padding. |
| `table_7_imagenet.py` | `imagenet_table_07.py` | Mesmo padrão da pasta. |
| `table_8_imagenet.py` | `imagenet_table_08.py` | Mesmo padrão da pasta. |
| `table_10.py` | `mnist_table_10_m1_fgsm.py` | O nome atual é genérico; o script cobre MNIST, modelo M1 e FGSM. |
| `generate_table_10_m2_cw.py` | `mnist_table_10_m2_cw.py` | O script está associado ao bloco MNIST/M2/CW da Table 10. |

### Observação sobre a Table 10

A Table 10 do artigo não representa apenas CW. Ela agrega vários blocos de ataque/modelo/dataset, como FGSM, DeepFool, CW-L2 e CW-L∞.

Portanto, os scripts atuais cobrem apenas partes da Table 10:

| Bloco | Script atual | Nome proposto |
|---|---|---|
| MNIST + M1 + FGSM | `table_10.py` | `mnist_table_10_m1_fgsm.py` |
| MNIST + M2 + CW | `generate_table_10_m2_cw.py` | `mnist_table_10_m2_cw.py` |

Caso novos blocos da Table 10 sejam implementados no futuro, seguir o mesmo padrão:

```text
imagenet_table_10_googlenet_fgsm.py
imagenet_table_10_googlenet_deepfool.py
imagenet_table_10_caffenet_deepfool.py
imagenet_table_10_inceptionv3_cw.py
```

---

## 2. Extração de Código Duplicado

As funções abaixo aparecem em dois ou mais scripts com implementações praticamente iguais. Todas devem ser movidas para módulos em `src/deepdetector/`.

O objetivo é deixar os scripts de `scripts/article_reproduction` como arquivos pequenos, responsáveis principalmente por:

- parsear argumentos de linha de comando;
- chamar funções reutilizáveis de `src/deepdetector/`;
- imprimir os caminhos dos artefatos gerados.

---

### 2.1 `_resolve_path` → `deepdetector.io.paths.resolve_project_path`

**Presente em:**

- `table_7_imagenet.py`
- `table_8_imagenet.py`
- `generate_table_10_m2_cw.py`

**Problema:**  
A função aparece com corpo praticamente idêntico em múltiplos scripts.

**Destino proposto:**

```text
src/deepdetector/io/paths.py
```

**Assinatura proposta:**

```python
from pathlib import Path


def resolve_project_path(
    path_value: str | None,
    project_root: Path,
) -> Path | None:
    """Resolve a path relative to the project root."""
    if path_value in (None, ""):
        return None

    path = Path(str(path_value))
    if path.is_absolute():
        return path

    return project_root / path
```

---

### 2.2 `load_config` → `deepdetector.io.config.load_yaml_config`

**Presente em:**

- `table_7_imagenet.py`
- `table_8_imagenet.py`
- `generate_table_10_m2_cw.py`

**Observação:**  
`table_4_imagenet.py` já importa `load_yaml_config` de `deepdetector.io.config`, então a extração deve consolidar esse caminho em vez de criar mais uma função isolada.

**Destino proposto:**

```text
src/deepdetector/io/config.py
```

**Ação proposta:**

- Manter uma única função `load_yaml_config`.
- Remover os `load_config` inline dos scripts.
- Consolidar o fallback manual de YAML que hoje aparece em `table_8_imagenet.py`, caso o projeto queira manter compatibilidade sem `PyYAML`.

**Uso esperado nos scripts:**

```python
from deepdetector.io.config import load_yaml_config

config = load_yaml_config(config_path)
```

---

### 2.3 `build_model` → helper compartilhado para ImageNet

**Presente em:**

- `table_7_imagenet.py`
- `table_8_imagenet.py`

**Problema:**  
A função é idêntica nos dois scripts e instancia `GoogLeNetCaffeWrapper` a partir do YAML.

**Destino proposto:**

Opção preferencial:

```text
src/deepdetector/models/imagenet_wrappers.py
```

Ou, caso seja mais adequado manter lógica de experimento fora de `models`:

```text
src/deepdetector/experiments/table4_imagenet_runner.py
```

mas exportando como helper compartilhado.

**Nome sugerido:**

```python
build_googlenet_caffe_model(config, project_root)
```

**Uso esperado nos scripts:**

```python
from deepdetector.models.imagenet_wrappers import build_googlenet_caffe_model

model = build_googlenet_caffe_model(config, project_root=PROJECT_ROOT)
```

---

### 2.4 `filter_clean_baseline_images` → `deepdetector.evaluation.imagenet_utils`

**Presente em:**

- `table_7_imagenet.py`
- `table_8_imagenet.py`

**Problema:**  
Mesma assinatura e mesmo corpo nos dois scripts.

**Destino proposto:**

```text
src/deepdetector/evaluation/imagenet_utils.py
```

**Responsabilidade:**

Filtrar imagens que o modelo já erra antes do ataque, mantendo apenas as corretamente classificadas no baseline limpo.

**Uso esperado nos scripts:**

```python
from deepdetector.evaluation.imagenet_utils import filter_clean_baseline_images

images, labels, selected_indices, clean_summary = filter_clean_baseline_images(
    model=model,
    images=images,
    labels=labels,
)
```

---

### 2.5 `load_adversarial_images`, `generate_adversarial_images` e `adversarial_images_for_run`

**Presente em:**

- `table_7_imagenet.py`
- `table_8_imagenet.py`

**Problema:**  
O fluxo de carregar adversariais salvas, gerar FGSM quando necessário, validar shape e salvar `.npy` é idêntico nos dois scripts.

**Destino proposto:**

```text
src/deepdetector/attacks/adversarial_loader.py
```

**Funções a extrair:**

```python
load_adversarial_images(...)
generate_adversarial_images(...)
adversarial_images_for_run(...)
```

**Responsabilidade:**

Centralizar a decisão:

1. usar `--adv-path`, se fornecido;
2. usar `attack.adversarial_path`, se existir;
3. gerar adversariais via FGSM, se possível;
4. salvar em `attack.save_adversarial_path`, se configurado.

---

### 2.6 Helpers privados de ImageNet

**Funções:**

```python
_read_rgb_image
_article_model_inputs
_predict_label
_class_image_rows
_epsilon_normalized
_epsilon_255
_label_to_int
```

**Presente em:**

- `table_7_imagenet.py`
- `table_8_imagenet.py`

**Destino proposto:**

Dividir por responsabilidade:

```text
src/deepdetector/data/imagenet.py
```

Para funções de dados:

```python
read_rgb_image
class_image_rows
```

```text
src/deepdetector/evaluation/imagenet_utils.py
```

Para funções de avaliação/preparação:

```python
article_model_inputs
predict_label
label_to_int
epsilon_normalized
epsilon_255
```

**Observação:**  
Se `epsilon_normalized` e `epsilon_255` forem usados também por MNIST ou outros ataques, eles podem ir para:

```text
src/deepdetector/attacks/config.py
```

---

### 2.7 `write_pivot_csv` → `deepdetector.io.csv_writer.write_pivot_csv`

**Presente em:**

- `table_7_imagenet.py`
- `table_8_imagenet.py`

**Problema:**  
As funções são quase idênticas. A diferença principal é o conjunto de colunas.

**Destino proposto:**

```text
src/deepdetector/io/csv_writer.py
```

**Assinatura proposta:**

```python
from pathlib import Path
from typing import Any, Sequence


def write_pivot_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    metric_rows: Sequence[tuple[str, str]] | None = None,
) -> Path:
    ...
```

**Uso esperado na Table 7:**

```python
TABLE_07_COLUMNS = [
    "cross_3x3",
    "cross_5x5",
    "cross_7x7",
    "cross_9x9",
    "diamond_3x3",
    "diamond_5x5",
    "diamond_7x7",
    "diamond_9x9",
    "box_3x3",
    "box_5x5",
    "box_7x7",
    "box_9x9",
]
```

**Uso esperado na Table 8:**

```python
TABLE_08_COLUMNS = [
    "cross_5x5",
    "cross_7x7",
    "diamond_5x5",
    "diamond_7x7",
    "box_5x5",
]
```

---

### 2.8 `output_dir_from_config` → `deepdetector.io.config.output_dir_from_config`

**Presente em:**

- `table_7_imagenet.py`
- `table_8_imagenet.py`

**Problema:**  
A função é idêntica nos dois scripts.

**Destino proposto:**

```text
src/deepdetector/io/config.py
```

**Assinatura proposta:**

```python
from pathlib import Path
from typing import Any


def output_dir_from_config(
    config: dict[str, Any],
    override: str | None,
    project_root: Path,
    default_output_dir: Path,
) -> Path:
    ...
```

**Responsabilidade:**

Resolver diretório de saída usando a seguinte prioridade:

1. argumento `--output-dir`;
2. `outputs.results_dir` ou `output.results_dir` no YAML;
3. diretório padrão do script.

---

## 3. Resultado Esperado nos Scripts

Após a extração, os scripts em `scripts/article_reproduction` devem ficar menores e mais consistentes.

Exemplo desejado para `imagenet_table_07.py`:

```python
"""Reproduce ImageNet Table 7 spatial smoothing filter metrics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deepdetector.io.config import load_yaml_config, output_dir_from_config
from deepdetector.models.imagenet_wrappers import build_googlenet_caffe_model
from deepdetector.experiments.article_reproduction.imagenet_table_07 import run_table_07


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "article_reproduction" / "imagenet_table_07.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "imagenet" / "article_reproduction"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adv-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()

    config = load_yaml_config(Path(args.config))
    output_dir = output_dir_from_config(
        config=config,
        override=args.output_dir,
        project_root=PROJECT_ROOT,
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )

    result = run_table_07(
        config=config,
        output_dir=output_dir,
        adv_path=args.adv_path,
        dry_run=args.dry_run,
        project_root=PROJECT_ROOT,
    )

    for key, value in result.items():
        print(f"{key}={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 4. Ordem Recomendada de Implementação

### Etapa 1 — Renomear arquivos

Renomear os scripts mantendo o comportamento atual:

```text
table_4_imagenet.py        -> imagenet_table_04.py
table_7_imagenet.py        -> imagenet_table_07.py
table_8_imagenet.py        -> imagenet_table_08.py
table_10.py                -> mnist_table_10_m1_fgsm.py
generate_table_10_m2_cw.py -> mnist_table_10_m2_cw.py
```

Opcionalmente, manter wrappers antigos temporários para compatibilidade:

```python
"""Deprecated. Use imagenet_table_07.py instead."""
```

---

### Etapa 2 — Extrair helpers de I/O

Criar ou atualizar:

```text
src/deepdetector/io/paths.py
src/deepdetector/io/config.py
src/deepdetector/io/csv_writer.py
```

Extrair:

```text
_resolve_path
load_config
output_dir_from_config
write_pivot_csv
```

---

### Etapa 3 — Extrair helpers de ImageNet

Criar ou atualizar:

```text
src/deepdetector/data/imagenet.py
src/deepdetector/evaluation/imagenet_utils.py
src/deepdetector/attacks/adversarial_loader.py
```

Extrair:

```text
_read_rgb_image
_class_image_rows
_article_model_inputs
_predict_label
_label_to_int
_epsilon_normalized
_epsilon_255
filter_clean_baseline_images
load_adversarial_images
generate_adversarial_images
adversarial_images_for_run
```

---

### Etapa 4 — Atualizar scripts renomeados

Substituir funções inline por imports.

Exemplo:

```python
from deepdetector.io.paths import resolve_project_path
from deepdetector.io.config import load_yaml_config, output_dir_from_config
from deepdetector.io.csv_writer import write_pivot_csv
from deepdetector.evaluation.imagenet_utils import filter_clean_baseline_images
from deepdetector.attacks.adversarial_loader import adversarial_images_for_run
```

---

### Etapa 5 — Validar comportamento

Executar pelo menos:

```bash
python scripts/article_reproduction/imagenet_table_07.py --dry-run
python scripts/article_reproduction/imagenet_table_08.py --dry-run
python scripts/article_reproduction/mnist_table_10_m1_fgsm.py --help
python scripts/article_reproduction/mnist_table_10_m2_cw.py --help
```

Validar que:

- os argumentos continuam funcionando;
- os paths de saída continuam corretos;
- os CSVs preservam o formato esperado;
- os status JSON continuam sendo gerados;
- os scripts antigos, caso mantidos como wrappers, apontam para os novos nomes.

---

## 5. Escopo Desta Refatoração

Este plano **não** propõe, neste momento:

- criar um runner único para todas as tabelas;
- reestruturar toda a árvore de experimentos;
- alterar a lógica matemática dos filtros;
- modificar os resultados esperados das tabelas;
- implementar blocos faltantes da Table 10;
- mudar a API pública do pacote além dos helpers extraídos.

O foco é:

1. padronizar nomes em `scripts/article_reproduction`;
2. reduzir duplicação;
3. mover lógica compartilhada para `src/deepdetector`;
4. manter o comportamento atual dos scripts.
