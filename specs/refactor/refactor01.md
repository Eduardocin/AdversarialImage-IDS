# Spec 1 — Centralizar infraestrutura de experimento

## Status

Proposto.

## Contexto

Os experimentos atuais de reprodução possuem scripts independentes para cada tabela. Esse formato funcionou para avançar rapidamente, mas gerou duplicação em pontos centrais do código, principalmente:

- descoberta da raiz do projeto;
- inclusão manual de `src` no `sys.path`;
- resolução de paths relativos ao projeto;
- carregamento de YAML;
- criação de diretórios de saída;
- escrita de arquivos de resultado;
- geração de Markdown como artefato de execução;
- definição local de nomes e caminhos de output.

Essa duplicação aumenta o risco de inconsistência entre os experimentos das Tabelas 3, 4, 6, 7, 8, 9 e 10.

Antes de refatorar os fluxos específicos de cada tabela, é necessário centralizar a infraestrutura comum.

---

## Problema

Atualmente, cada script de experimento tende a funcionar como um mini-runner próprio.

Isso faz com que mudanças simples, como alterar o formato de saída, a forma de resolver paths ou a validação de YAML, precisem ser replicadas em múltiplos arquivos.

Exemplos de responsabilidades que não deveriam ficar repetidas em scripts de tabela:

```python
PROJECT_ROOT = next(...)
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG = ...

def _resolve_path(...):
    ...

def load_config(...):
    ...
```
Além disso, a escrita de resultados ainda está acoplada a formatos diferentes, principalmente CSV e Markdown.
O objetivo do projeto passa a ser padronizar os artefatos de execução em apenas:
- metrics.csv
- metrics.json

---

## Objetivo
Criar uma infraestrutura central reutilizável para experimentos, removendo duplicação de configuração, paths e outputs dos scripts individuais.
Ao final desta spec, os scripts de experimento devem delegar a infraestrutura comum para módulos internos do pacote deepdetector, em vez de implementar localmente:
- carregamento de config;
- resolução de paths;
- criação de diretórios;
- escrita de resultados;
- montagem mínima de metadata do experimento.

---

##  Fora de escopo
Esta spec não deve refatorar ainda a lógica específica das Tabelas 6, 7, 8 ou 9.
Também não faz parte desta spec:
- alterar a matemática dos filtros;
- alterar a geração de adversariais FGSM;
- alterar o cálculo de TP, FN, FP, recall, precision ou F1;
- implementar o runner único das Tabelas 6 e 9;
- implementar o runner de candidatos das Tabelas 7 e 8;
- limpar completamente o YAML;
- remover todas as referências ao artigo em documentação histórica.
Esses pontos devem ser tratados nas próximas specs.

---

## Proposta de solução
Criar módulos centrais para infraestrutura de experimento:
```text
src/deepdetector/
  io/
    __init__.py
    paths.py
    config.py
    result_writers.py

  experiments/
    __init__.py
    metadata.py
```
---

# 1. Módulo de paths
Criar: `src/deepdetector/io/paths.py`

Com funções como:
```python
from pathlib import Path
from typing import Optional, Union

def get_project_root() -> Path:
    """Return the repository root containing pyproject.toml."""

def resolve_project_path(path_value: Optional[Union[str, Path]]) -> Optional[Path]:
    """Resolve a path relative to the project root when it is not absolute."""

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory if needed and return it as Path."""
```

### Regras esperadas
- get_project_root() deve procurar o diretório que contém pyproject.toml.
- resolve_project_path(None) deve retornar None.
- Paths absolutos devem ser preservados.
- Paths relativos devem ser resolvidos a partir da raiz do projeto.
- Scripts não devem mais inserir manualmente src no sys.path.

---

## 2. Módulo de configuração
Criar: `src/deepdetector/io/config.py`

Com funções como:
```python
from pathlib import Path
from typing import Any, Dict

def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config and ensure the root object is a mapping."""

def get_config_section(config: Dict[str, Any], section: str, default=None):
    """Return a config section with an optional default."""
``` 

### Regras esperadas
- YAML vazio deve gerar ValueError.
- YAML cujo root não seja mapping deve gerar ValueError.
- O erro deve informar qual arquivo/config está inválido.
- Scripts não devem mais implementar load_config local.

---

## 3. Módulo de escrita de resultados
Criar: `src/deepdetector/io/result_writers.py`

Com funções como:
```python
from pathlib import Path
from typing import Any, Dict, Sequence

def write_metrics_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    """Write experiment metrics to CSV with stable column order."""

def write_metrics_json(
    path: Path,
    payload: Dict[str, Any],
) -> Path:
    """Write experiment metadata and metrics to JSON."""

def write_experiment_outputs(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    csv_fields: Sequence[str],
    metadata: Dict[str, Any],
    csv_name: str = "metrics.csv",
    json_name: str = "metrics.json",
) -> Dict[str, Path]:
    """Write the standard CSV and JSON outputs for an experiment."""
```

### Regras esperadas
- O CSV deve preservar exatamente a ordem de csv_fields.
- O JSON deve ser indentado e legível.
- write_experiment_outputs deve gerar somente dois arquivos por padrão:
    * metrics.csv;
    * metrics.json.
- O módulo não deve gerar Markdown.
- O módulo não deve conhecer detalhes das tabelas do artigo.

---

## 4. Metadados de experimento
Criar: `src/deepdetector/experiments/metadata.py`

Com utilitário simples para montar payloads JSON padronizados:
```python
from typing import Any, Dict, Sequence

def build_experiment_payload(
    experiment_id: str,
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the standard JSON payload for experiment outputs."""
```

Payload mínimo esperado:
```json
{
  "experiment_id": "table_6_adaptive_quantization",
  "config": {},
  "metrics": [],
  "extra": {}
}
```

---

## Mudanças esperadas nos scripts existentes
Os scripts de tabela devem parar de definir localmente:
- PROJECT_ROOT;
- SRC_ROOT;
- injeção de src no sys.path;
- _resolve_path;
- load_config;
- escrita manual de CSV/Markdown.
O padrão esperado passa a ser:
```python
from deepdetector.io.config import load_yaml_config
from deepdetector.io.paths import resolve_project_path, ensure_dir
from deepdetector.io.result_writers import write_experiment_outputs
from deepdetector.experiments.metadata import build_experiment_payload
``` 

Nesta spec, não é obrigatório migrar todos os scripts de tabela.
A migração mínima deve cobrir pelo menos um script representativo para validar a infraestrutura. O candidato preferencial é:
scripts/article_reproduction/table_6.py

Motivo:
- é simples;
- já usa YAML;
- já gera CSV e Markdown;
- será usado nas próximas specs como base para o runner único das Tabelas 6 e 9.

---

## Contrato de saída
Todo experimento migrado nesta spec deve gerar apenas:
- metrics.csv
- metrics.json

Não deve gerar:
- *.md
- summary.md
- diagnostic.md
- comparison.md

---

## Exemplo de output esperado
Estrutura sugerida:
```text
results/
  experiments/
    table_6_adaptive_quantization/
      metrics.csv
      metrics.json
```
- metrics.csv
```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
Training,0,0,0,0.0,0.0,0.0
Validation,0,0,0,0.0,0.0,0.0
```

metrics.json
```json
{
  "experiment_id": "table_6_adaptive_quantization",
  "config": {
    "dataset": {
      "name": "mnist"
    },
    "attack": {
      "name": "fgsm",
      "epsilon": 0.2
    }
  },
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
  "extra": {}
}
```

--- 

## Critérios de aceitação
### Infraestrutura
- Existe src/deepdetector/io/__init__.py.
- Existe src/deepdetector/io/paths.py.
- Existe src/deepdetector/io/config.py.
- Existe src/deepdetector/io/result_writers.py.
- Existe src/deepdetector/experiments/__init__.py.
- Existe src/deepdetector/experiments/metadata.py.
- load_yaml_config valida que o YAML possui root mapping.
- resolve_project_path resolve paths relativos a partir da raiz do projeto.
- write_experiment_outputs gera CSV e JSON no diretório informado.
- write_experiment_outputs não gera Markdown.

### Migração mínima
- Pelo menos um script de experimento usa a nova infraestrutura.
- O script migrado não possui _resolve_path local.
- O script migrado não possui load_config local.
- O script migrado não injeta src manualmente no sys.path.
- O script migrado não chama write_markdown_table.
- O script migrado gera metrics.csv e metrics.json.

### Testes
- Teste para load_yaml_config com YAML válido.
- Teste para load_yaml_config com YAML vazio.
- Teste para load_yaml_config com YAML cujo root não é mapping.
- Teste para resolve_project_path com path absoluto.
- Teste para resolve_project_path com path relativo.
- Teste para resolve_project_path(None).
- Teste para write_metrics_csv preservando ordem das colunas.
- Teste para write_metrics_json gerando JSON válido.
- Teste para write_experiment_outputs gerando apenas CSV e JSON.
- Teste para garantir que o output padrão não cria arquivos .md.

---

## Sugestão de estrutura de testes
```text
tests/
  io/
    test_config.py
    test_paths.py
    test_result_writers.py

  experiments/
    test_metadata.py
```

---

## Riscos
### Risco 1 — Quebrar execução por import
Alguns scripts hoje inserem `src` manualmente no `sys.path`.
Remover isso pode quebrar execução direta caso o pacote não esteja instalado em modo editável.
Mitigação:
- manter documentação recomendando: `pip install -e .`
- preferir execução via: `python -m ...`

### Risco 2 — Mudança de nomes de output
Scripts antigos podem gerar arquivos com nomes específicos, como:
- table_6_adaptive_quantization.csv
- table_6_adaptive_quantization.md

Mitigação:
- nesta spec, permitir configurar csv_name e json_name;
- nas specs seguintes, padronizar definitivamente para:
    * metrics.csv
    * metrics.json

### Risco 3 — Misturar refatoração de infraestrutura com lógica experimental
Esta spec não deve alterar o cálculo das métricas nem a implementação dos filtros.
Mitigação:
- testes devem comparar apenas estrutura de output e comportamento de infraestrutura;
- alterações nos runners das tabelas devem ficar nas próximas specs.

---

## Plano de implementação sugerido
1. Criar pacote deepdetector.io.
2. Implementar paths.py.
3. Implementar config.py.
4. Implementar result_writers.py.
5. Criar pacote deepdetector.experiments, se ainda não existir.
6. Implementar experiments/metadata.py.
7. Criar testes unitários para os módulos novos.
8. Migrar scripts/article_reproduction/table_6.py para usar a infraestrutura nova.
9. Remover geração de Markdown do script migrado.
10. Validar que o script ainda gera as mesmas métricas no CSV.
11. Confirmar que o diretório de saída contém apenas CSV e JSON.

---

## Definição de pronto
A spec é considerada concluída quando:
- existe infraestrutura central para config/path/output;
- pelo menos um script de experimento foi migrado para essa infraestrutura;
- os testes cobrem os módulos novos;
- o script migrado gera apenas CSV e JSON;
- nenhum Markdown é gerado como artefato de execução do script migrado;
- a lógica de cálculo experimental permanece inalterada.
