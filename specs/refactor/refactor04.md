# Spec 4 — Limpeza destrutiva e simplificação dos experimentos

## Status

Proposto.

## Contexto

As Specs 1, 2 e 3 já criaram uma infraestrutura mais centralizada para os experimentos das Tabelas 6, 7, 8 e 9.

Porém, essa refatoração também pode ter aumentado a quantidade de arquivos, scripts, wrappers e configs. Isso resolve parte da duplicação interna, mas ainda deixa o projeto com ruído operacional.

A intenção original da refatoração não era apenas criar uma arquitetura mais organizada. Era também:

```text
remover redundância
melhorar o fluxo
excluir código desnecessário
reduzir arquivos
deixar a execução objetiva
```

Portanto, a Spec 4 deve ser uma etapa de limpeza destrutiva.

O foco agora é reduzir, consolidar e remover.

---

## Objetivo

Simplificar a estrutura criada nas specs anteriores, removendo arquivos redundantes e mantendo apenas o fluxo mínimo necessário para executar os experimentos das Tabelas 6, 7, 8 e 9.

A Spec 4 deve garantir:

- menos scripts;
- menos configs;
- menos outputs;
- menos wrappers;
- menos referências ao artigo no runtime;
- menos caminhos para executar a mesma coisa.

A refatoração só deve ser considerada concluída se o projeto ficar mais simples do que estava após as Specs 1, 2 e 3.

---

## Princípios

- Não criar arquivos novos sem necessidade clara.
- Preferir remover arquivos a criar novas camadas.
- Preferir um ponto de entrada único a vários scripts por tabela.
- Preferir um YAML consolidado a vários YAMLs quase iguais, se isso reduzir complexidade.
- Manter apenas CSV e JSON como outputs oficiais.
- Remover Markdown, diagnóstico e comparação hardcoded com o artigo.
- Remover scripts antigos que só duplicam fluxo.
- Remover configs antigas que já foram substituídas.
- Remover wrappers desnecessários.
- A quantidade total de arquivos relacionados aos experimentos deve diminuir.

---

## Problema

Após as Specs 1, 2 e 3, o projeto pode ter ficado com arquivos como:

```text
src/deepdetector/io/
src/deepdetector/experiments/
src/deepdetector/filters/factory.py

scripts/experiments/table_6.py
scripts/experiments/table_7.py
scripts/experiments/table_8.py
scripts/experiments/table_9.py

configs/experiments/table_6_*.yaml
configs/experiments/table_7_*.yaml
configs/experiments/table_8_*.yaml
configs/experiments/table_9_*.yaml

scripts/article_reproduction/table_*.py
configs/article_reproduction/*.yaml
results/**/*.md
```

Mesmo que parte desses arquivos seja tecnicamente “organizada”, o fluxo fica mais difícil de entender.

O projeto precisa deixar claro:

- onde configuro;
- onde executo;
- onde fica a lógica comum;
- onde ficam os resultados.

---

## Fora de escopo

Esta spec não deve alterar:

- cálculo de métricas;
- implementação dos filtros;
- geração FGSM;
- modelo usado;
- valores de epsilon;
- slices experimentais;
- resultados esperados.

Esta spec também não deve criar novos fluxos experimentais.

---

# Direção desejada

A estrutura final deve ser objetiva.

## Estrutura preferencial

```text
configs/
  experiments.yaml

scripts/
  run_experiment.py
  README.md
  dev/
    smoke_test.py

src/deepdetector/
  experiments/
    runner.py
```

Ou, se o projeto ficar mais simples com um arquivo único:

```text
src/deepdetector/
  experiments.py
```

A decisão entre `src/deepdetector/experiments/runner.py` e `src/deepdetector/experiments.py` deve priorizar simplicidade.

Não criar uma árvore grande de módulos se a lógica couber de forma clara em um arquivo.

---

# Redundância em scripts

A pasta `scripts/` deve ser tratada como interface operacional, não como local de implementação de experimentos.

Após as Specs 1, 2 e 3, podem existir múltiplos scripts que apenas delegam para runners internos. Isso ainda é redundância, mesmo que a lógica principal esteja no `src`.

A Spec 4 deve consolidar a execução dos experimentos das Tabelas 6–9 em um único ponto de entrada:

```bash
python scripts/run_experiment.py --experiment table_6
python scripts/run_experiment.py --experiment table_7
python scripts/run_experiment.py --experiment table_8
python scripts/run_experiment.py --experiment table_9
```

Scripts por tabela devem ser removidos.

Caso seja necessário manter compatibilidade temporária, eles devem ser wrappers depreciados, sem lógica própria, e não devem aparecer na documentação principal.

## Remover ou depreciar

```text
scripts/experiments/table_6.py
scripts/experiments/table_7.py
scripts/experiments/table_8.py
scripts/experiments/table_9.py

scripts/article_reproduction/table_6.py
scripts/article_reproduction/table_7.py
scripts/article_reproduction/table_8.py
scripts/article_reproduction/table_9.py
```

Também revisar scripts relacionados a tabelas antigas, como:

```text
scripts/article_reproduction/table_3.py
scripts/article_reproduction/table_4.py
scripts/article_reproduction/table_4_mnist.py
scripts/article_reproduction/table_10.py
```

Se eles não fizerem parte do fluxo atual, devem ser removidos, depreciados ou movidos para uma pasta claramente legada.

## Estrutura desejada para scripts

```text
scripts/
  run_experiment.py
  README.md
  dev/
    smoke_test.py
```

## Regra

Nenhum script antigo deve conter lógica de:

- dataset;
- modelo;
- ataque;
- filtro;
- métrica;
- escrita de resultado;
- comparação com artigo;
- diagnóstico;
- Markdown.

Scripts devem apenas chamar o fluxo central, quando realmente forem necessários.

---

# Fluxos que devem permanecer

O projeto precisa suportar apenas dois tipos de fluxo para as Tabelas 6–9.

---

## 1. `split_eval`

Usado por:

- Table 6;
- Table 9.

Avalia um filtro em múltiplos splits.

Exemplo:

```yaml
table_6:
  kind: split_eval
  filter: adaptive_quantization
  slices:
    - name: Training
      start: 0
      end: 4500
    - name: Validation
      start: 4500
      end: 5500
```

---

## 2. `filter_grid`

Usado por:

- Table 7;
- Table 8.

Avalia múltiplos filtros candidatos em um slice.

Exemplo:

```yaml
table_7:
  kind: filter_grid
  slice:
    name: Training
    start: 0
    end: 4500
  high_entropy_only: true
  filters:
    - cross_3x3
    - cross_5x5
    - cross_7x7
    - cross_9x9
    - diamond_3x3
    - diamond_5x5
    - diamond_7x7
    - diamond_9x9
    - box_3x3
    - box_5x5
    - box_7x7
    - box_9x9
```

---

# Configuração mínima recomendada

Consolidar em um único arquivo, se isso reduzir complexidade:

```text
configs/experiments.yaml
```

## Exemplo

```yaml
defaults:
  dataset: mnist
  model: mnist_m1
  checkpoint_dir: artifacts/models/mnist/m1/clean_baseline/checkpoints
  attack: fgsm
  epsilon: 0.2
  clip_min: 0.0
  clip_max: 1.0
  batch_size: 256
  exclude_invalid_pairs: false

experiments:
  table_6:
    kind: split_eval
    output_dir: results/experiments/table_6
    filter: adaptive_quantization
    slices:
      - name: Training
        start: 0
        end: 4500
      - name: Validation
        start: 4500
        end: 5500

  table_7:
    kind: filter_grid
    output_dir: results/experiments/table_7
    slice:
      name: Training
      start: 0
      end: 4500
    high_entropy_only: true
    entropy_min: 5.0
    base_filter:
      name: scalar_quantization
      intervals: 6
    filters:
      - cross_3x3
      - cross_5x5
      - cross_7x7
      - cross_9x9
      - diamond_3x3
      - diamond_5x5
      - diamond_7x7
      - diamond_9x9
      - box_3x3
      - box_5x5
      - box_7x7
      - box_9x9

  table_8:
    kind: filter_grid
    output_dir: results/experiments/table_8
    slice:
      name: Validation
      start: 4500
      end: 5500
    high_entropy_only: true
    entropy_min: 5.0
    base_filter:
      name: scalar_quantization
      intervals: 6
    filters:
      - cross_5x5
      - cross_7x7
      - diamond_5x5
      - diamond_7x7
      - box_5x5

  table_9:
    kind: split_eval
    output_dir: results/experiments/table_9
    filter: proposed_filter
    slices:
      - name: Training
        start: 0
        end: 4500
      - name: Validation
        start: 4500
        end: 5500
```

Se manter YAMLs separados for mais claro na implementação atual, isso é permitido, mas deve ser justificado.

## Regra principal

> Não manter múltiplos YAMLs quase iguais se um YAML consolidado resolver melhor.

---

# Ponto de entrada único

Criar ou consolidar em:

```text
scripts/run_experiment.py
```

## Uso esperado

```bash
python scripts/run_experiment.py --experiment table_6
python scripts/run_experiment.py --experiment table_7
python scripts/run_experiment.py --experiment table_8
python scripts/run_experiment.py --experiment table_9
```

O script deve:

1. ler `configs/experiments.yaml`;
2. selecionar o experimento solicitado;
3. mesclar `defaults` com a config do experimento;
4. chamar o runner apropriado com base em `kind`.

## Exemplo conceitual

```python
if experiment["kind"] == "split_eval":
    run_split_eval(experiment)

elif experiment["kind"] == "filter_grid":
    run_filter_grid(experiment)

else:
    raise ValueError(f"Unknown experiment kind: {experiment['kind']}")
```

---

# Código central mínimo

Consolidar a lógica experimental em poucos lugares.

## Estrutura aceitável

```text
src/deepdetector/experiments/runner.py
```

Com funções:

```python
run_experiment(name, config)
run_split_eval(config)
run_filter_grid(config)
```

Ou:

```text
src/deepdetector/experiments.py
```

Se isso deixar o projeto mais simples.

O importante é evitar uma árvore grande de módulos para uma lógica pequena.

---

# O que deve ser removido

## 1. Scripts por tabela

Remover ou depreciar:

```text
scripts/experiments/table_6.py
scripts/experiments/table_7.py
scripts/experiments/table_8.py
scripts/experiments/table_9.py
```

Preferência: remover.

Se mantidos temporariamente, devem ser wrappers mínimos e depreciados.

---

## 2. Scripts antigos de reprodução

Remover ou depreciar:

```text
scripts/article_reproduction/table_3.py
scripts/article_reproduction/table_4.py
scripts/article_reproduction/table_4_mnist.py
scripts/article_reproduction/table_6.py
scripts/article_reproduction/table_10.py
```

E quaisquer scripts equivalentes de Table 7, 8 e 9, se existirem.

---

## 3. YAMLs redundantes

Se `configs/experiments.yaml` for adotado, remover:

```text
configs/experiments/table_6_adaptive_quantization.yaml
configs/experiments/table_7_smoothing_filter_search.yaml
configs/experiments/table_8_smoothing_filter_validation.yaml
configs/experiments/table_9_proposed_filter.yaml
```

Remover configs antigas substituídas:

```text
configs/article_reproduction/*.yaml
```

---

## 4. Outputs Markdown

Remover outputs versionados como:

```text
results/**/*.md
results/**/summary.md
results/**/diagnostic.md
results/**/comparison.md
```

Novos experimentos não devem gerar `.md`.

---

## 5. Código de Markdown

Remover funções não usadas como:

```text
write_markdown_table
```

Remover prints como:

```text
results_md=...
comparison_md=...
```

---

## 6. Comparação hardcoded com artigo

Remover constantes e lógica como:

```text
ARTICLE_TABLE_10 = {...}
article_TP
article_FN
article_FP
article_f1_percent
Delta F1
Reference
Our
```

O runtime deve gerar métricas locais, não comparação com valores do artigo.

---

## 7. Diagnóstico automático

Remover geração de:

```text
diagnostic
diagnostico
report
summary textual
comparison textual
```

Se relatório for necessário no futuro, deve ser outro comando separado, não parte do experimento.

---

# O que pode permanecer

Podem permanecer, se realmente forem usados e simplificarem o código:

```text
deepdetector.io.config
deepdetector.io.paths
deepdetector.io.result_writers
deepdetector.filters.factory
```

Mas se forem wrappers muito pequenos e usados em apenas um lugar, considerar consolidar no runner.

## Regra

> Se o arquivo existe só para uma função trivial usada uma vez, remover ou consolidar.

---

# Critérios objetivos de simplificação

A PR desta spec deve demonstrar redução real.

Medir antes e depois:

```bash
find scripts -type f | wc -l
find configs -type f | wc -l
find src/deepdetector -type f | wc -l
find results -name "*.md" | wc -l
```

A expectativa é:

- reduzir número de scripts relacionados às tabelas;
- reduzir número de YAMLs relacionados às tabelas;
- zerar ou reduzir drasticamente Markdown em `results`;
- remover código de comparação com artigo do runtime.

---

# Contrato de output final

Cada experimento deve gerar:

```text
results/experiments/<experiment_id>/metrics.csv
results/experiments/<experiment_id>/metrics.json
```

Nada além disso.

---

## Table 6 e Table 9

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
Training,0,0,0,0.0,0.0,0.0
Validation,0,0,0,0.0,0.0,0.0
```

---

## Table 7 e Table 8

```csv
filter_name,filter_type,mask_type,mask_size,TP,FN,FP,recall_percent,precision_percent,f1_percent
cross_7x7,cross_mean,cross,7,0,0,0,0.0,0.0,0.0
```

---

## JSON comum

```json
{
  "experiment_id": "table_7",
  "kind": "filter_grid",
  "config": {},
  "metrics": []
}
```

---

# Testes esperados

A Spec 4 deve priorizar testes de limpeza e contrato.

## Testes mínimos

- `scripts/run_experiment.py --experiment table_6` resolve config correta.
- `scripts/run_experiment.py --experiment table_7` resolve config correta.
- `scripts/run_experiment.py --experiment table_8` resolve config correta.
- `scripts/run_experiment.py --experiment table_9` resolve config correta.
- Experimento inválido gera erro claro.
- `kind` inválido gera erro claro.
- Output contém `metrics.csv`.
- Output contém `metrics.json`.
- Output não contém `.md`.
- CSV da Table 6/9 segue schema esperado.
- CSV da Table 7/8 segue schema esperado.
- Config consolidada possui `defaults`.
- Config consolidada possui `experiments.table_6`.
- Config consolidada possui `experiments.table_7`.
- Config consolidada possui `experiments.table_8`.
- Config consolidada possui `experiments.table_9`.

---

## Testes de ausência

Criar testes ou checks para garantir que runners e scripts principais não contêm:

```text
write_markdown_table
ARTICLE_TABLE_
Delta F1
comparison_md
results_md
diagnostic
diagnostico
summary.md
```

---

# Critérios de aceitação

## Simplificação estrutural

- Existe um único ponto de entrada ativo para Table 6–9.
- O ponto de entrada ativo é `scripts/run_experiment.py`.
- Não existe um script ativo por tabela com lógica própria.
- Scripts por tabela foram removidos ou depreciados.
- Scripts antigos não possuem lógica experimental.
- Scripts antigos não aparecem na documentação principal.
- Nenhum script antigo gera CSV, JSON, Markdown ou diagnóstico por conta própria.
- Existe um YAML consolidado ou uma estrutura de YAMLs comprovadamente mínima.
- A quantidade de scripts relacionados a experimentos diminuiu.
- A quantidade de configs relacionadas a experimentos diminuiu ou foi justificada.
- Arquivos novos criados nas specs anteriores que não agregam valor foram removidos ou consolidados.

## Outputs

- Experimentos geram apenas `metrics.csv` e `metrics.json`.
- Nenhum `.md` é gerado.
- Nenhum diagnóstico textual é gerado.
- Nenhum output compara automaticamente com valores do artigo.

## Runtime

- Runtime não contém `ARTICLE_TABLE_*`.
- Runtime não calcula `Delta F1`.
- Runtime não escreve colunas `Reference`/`Our` como padrão.
- Runtime não depende de `article_reproduction`.
- Runtime não possui wrappers desnecessários.

## Código

- Funções duplicadas foram removidas.
- Runners antigos foram removidos ou consolidados.
- Scripts antigos foram removidos ou depreciados.
- Configs antigas foram removidas ou depreciadas.
- O cálculo de métricas continua centralizado.
- A implementação dos filtros não foi alterada.

## Documentação

- `README` ou `scripts/README.md` mostra apenas o fluxo novo.
- Comandos antigos foram removidos da documentação principal.
- A documentação deixa claro que outputs oficiais são CSV e JSON.
- Qualquer referência ao artigo fica em documentação, não no runtime.

---

# Plano de implementação sugerido

1. Fazer inventário dos arquivos criados nas Specs 1, 2 e 3.
2. Listar scripts atuais relacionados às Tabelas 6–9.
3. Identificar quais scripts possuem lógica própria e quais são apenas wrappers.
4. Criar ou consolidar `scripts/run_experiment.py`.
5. Fazer `run_experiment.py` executar Table 6–9 por `--experiment`.
6. Consolidar configs em `configs/experiments.yaml`, se isso reduzir duplicação.
7. Consolidar runners redundantes.
8. Remover scripts por tabela, se não forem necessários.
9. Remover scripts antigos de `article_reproduction`, ou marcá-los como deprecated.
10. Remover YAMLs antigos substituídos.
11. Remover geração de Markdown.
12. Remover funções de Markdown não usadas.
13. Remover constantes hardcoded do artigo.
14. Remover geração de diagnóstico.
15. Remover outputs `.md` versionados.
16. Atualizar documentação operacional.
17. Criar testes de contrato.
18. Criar testes de ausência de `.md` e strings proibidas.
19. Comparar quantidade de arquivos antes/depois.
20. Rodar experimentos principais.
21. Validar que os resultados continuam sendo gerados em CSV/JSON.

---

# Definição de pronto

A Spec 4 está pronta quando a refatoração deixa o projeto mais simples do que após as Specs 1–3.

Critérios finais:

- um ponto de entrada principal;
- menos scripts;
- menos configs;
- menos outputs;
- menos wrappers;
- menos referências ao artigo no runtime;
- nenhum Markdown gerado;
- nenhum diagnóstico automático;
- CSV + JSON como únicos artefatos.