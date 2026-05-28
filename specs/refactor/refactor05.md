# Spec 6 — Simplificar e unificar os experimentos das Tabelas 3–6

## Status

Proposto.

---

## Contexto

Após as refatorações anteriores, o fluxo das Tabelas 6–9 começou a ficar mais padronizado, principalmente com o uso de um ponto de entrada único e de `configs/experiments.yaml`.

Agora o foco deve voltar para as Tabelas 3–6, porque elas ainda representam a primeira parte da reprodução do artigo e parecem ter muita redundância entre scripts, configs e lógica de avaliação.

O objetivo desta spec é simplificar esse bloco inicial:

```text
Table 3 -> comparação de quantização uniforme vs não uniforme
Table 4 -> variação do número de intervalos da quantização escalar
Table 5 -> análise/seleção intermediária relacionada a filtros/quantização
Table 6 -> avaliação da quantização adaptativa
```

A ideia não é criar vários arquivos novos. A ideia é **unificar o fluxo, remover duplicação e deixar a reprodução das Tabelas 3–6 objetiva**.

---

## Objetivo

Criar um fluxo mínimo e unificado para executar as Tabelas 3–6.

Ao final desta spec:

- Tabelas 3–6 devem rodar pelo mesmo ponto de entrada usado pelos demais experimentos.
- Scripts antigos específicos dessas tabelas devem ser removidos, depreciados ou convertidos em wrappers sem lógica.
- YAMLs individuais repetidos devem ser consolidados em `configs/experiments.yaml`.
- Outputs devem ser padronizados em `metrics.csv` e `metrics.json`.
- Markdown, diagnóstico e comparação hardcoded com o artigo não devem fazer parte do runtime.
- A lógica comum de carregamento de modelo, dataset, FGSM, filtros, métricas e outputs deve ficar centralizada.
- A quantidade total de scripts/configs relacionados às Tabelas 3–6 deve diminuir.

---

## Problema

As Tabelas 3–6 tendem a repetir o mesmo esqueleto experimental:

```text
carregar config
resolver paths
carregar MNIST
restaurar modelo
gerar FGSM
aplicar um ou mais filtros
calcular métricas
escrever CSV
escrever Markdown
```

A diferença principal entre elas está no conjunto de filtros/estratégias avaliadas:

```text
Table 3 -> lista curta de filtros de quantização
Table 4 -> grid de intervalos de quantização escalar
Table 5 -> variação intermediária, se existir no código atual
Table 6 -> quantização adaptativa em splits Training/Validation
```

Portanto, a solução não deve ser manter um script por tabela com lógica própria.

O correto é representar essas tabelas como variações de poucos tipos de experimento.

---

## Princípios

1. Não criar um runner específico para cada tabela.
2. Não criar um YAML específico para cada tabela se `configs/experiments.yaml` resolver melhor.
3. Não manter scripts antigos com lógica própria.
4. Não gerar Markdown.
5. Não gerar diagnóstico.
6. Não comparar automaticamente com valores hardcoded do artigo.
7. Não duplicar geração FGSM para cada filtro quando ela puder ser reutilizada.
8. Não duplicar path/config/output handling.
9. Manter a lógica matemática dos filtros inalterada.
10. Reduzir a quantidade total de arquivos relacionados às Tabelas 3–6.

---

## Escopo

Esta spec cobre:

- scripts das Tabelas 3–6;
- YAMLs das Tabelas 3–6;
- integração dessas tabelas no `configs/experiments.yaml`;
- integração dessas tabelas no `scripts/run_experiment.py`;
- padronização dos outputs;
- remoção de Markdown e diagnósticos;
- remoção de redundância em scripts e configs;
- reaproveitamento de runners já existentes quando possível.

Esta spec não deve alterar:

- implementação matemática dos filtros;
- cálculo de métricas;
- geração FGSM;
- arquitetura do modelo;
- valores experimentais definidos para reproduzir o artigo;
- datasets usados na reprodução.

---

# 1. Inventário obrigatório

Antes de alterar, levantar os arquivos atuais relacionados às Tabelas 3–6.

Comandos sugeridos:

```bash
find scripts -type f | grep -E "table_3|table_4|table_5|table_6|article_reproduction"
find configs -type f | grep -E "table_3|table_4|table_5|table_6|mnist"
find results -type f | grep -E "table_3|table_4|table_5|table_6|article_reproduction"
```

Classificar cada arquivo em:

```text
oficial
redundante
legado
remover
migrar
```

Exemplo:

| Arquivo | Categoria | Ação |
| --- | --- | --- |
| `scripts/article_reproduction/table_3.py` | legado/redundante | migrar para runner central e remover |
| `scripts/article_reproduction/table_4_mnist.py` | legado/redundante | migrar para runner central e remover |
| `scripts/article_reproduction/table_6.py` | legado/redundante | substituir por `run_experiment.py --experiment table_6` |
| `configs/article_reproduction/mnist_table_3.yaml` | redundante | migrar para `configs/experiments.yaml` |
| `configs/article_reproduction/mnist_table_4.yaml` | redundante | migrar para `configs/experiments.yaml` |
| `configs/article_reproduction/mnist_table_6.yaml` | redundante | migrar/remover |
| `results/mnist/article_reproduction/table_*.md` | output antigo | remover se não for usado em teste |

A PR deve deixar claro o que foi migrado e o que foi removido.

---

# 2. Tipos mínimos de experimento

As Tabelas 3–6 não precisam de quatro fluxos diferentes.

Elas podem ser representadas por dois tipos principais:

```text
filter_grid
split_eval
```

## 2.1. `filter_grid`

Usado por tabelas que avaliam múltiplos filtros ou múltiplas variações de um filtro sobre o mesmo conjunto de imagens.

Candidatos:

```text
Table 3
Table 4
possivelmente Table 5
```

Exemplos:

```text
Table 3 -> scalar_quantization_2, nonuniform_quantization, nonuniform_quantization_legacy
Table 4 -> scalar_quantization com intervalos 2–10
Table 5 -> filtros candidatos, se existir no código atual
```

Fluxo:

```text
1. carregar slice/dataset
2. restaurar modelo
3. gerar FGSM uma vez
4. calcular clean_pred e adv_pred uma vez
5. iterar sobre filtros
6. avaliar cada filtro usando os mesmos adversariais
7. escrever metrics.csv e metrics.json
```

## 2.2. `split_eval`

Usado por tabelas que avaliam um filtro em múltiplos splits.

Candidatos:

```text
Table 6
```

Fluxo:

```text
1. restaurar modelo uma vez
2. para cada split:
   - carregar imagens
   - gerar FGSM
   - aplicar filtro
   - calcular métricas
3. escrever metrics.csv e metrics.json
```

---

# 3. Configuração em `configs/experiments.yaml`

Adicionar as Tabelas 3–6 à configuração consolidada.

Exemplo sugerido:

```yaml
experiments:
  table_3:
    kind: filter_grid
    output_dir: results/experiments/table_3
    dataset:
      name: mnist
      split: test
      start: 0
      end: 4500
    attack:
      name: fgsm
      epsilon: 0.2
      clip_min: 0.0
      clip_max: 1.0
    filters:
      - name: scalar_quantization_2
        type: scalar_quantization
        intervals: 2
      - name: nonuniform_quantization
        type: nonuniform_quantization
      - name: nonuniform_quantization_legacy
        type: nonuniform_quantization_legacy

  table_4:
    kind: filter_grid
    output_dir: results/experiments/table_4
    dataset:
      name: mnist
      split: test
      start: 0
      end: 4500
    attack:
      name: fgsm
      epsilon: 0.2
      clip_min: 0.0
      clip_max: 1.0
    filters:
      - name: scalar_quantization_2
        type: scalar_quantization
        intervals: 2
      - name: scalar_quantization_3
        type: scalar_quantization
        intervals: 3
      - name: scalar_quantization_4
        type: scalar_quantization
        intervals: 4
      - name: scalar_quantization_5
        type: scalar_quantization
        intervals: 5
      - name: scalar_quantization_6
        type: scalar_quantization
        intervals: 6
      - name: scalar_quantization_7
        type: scalar_quantization
        intervals: 7
      - name: scalar_quantization_8
        type: scalar_quantization
        intervals: 8
      - name: scalar_quantization_9
        type: scalar_quantization
        intervals: 9
      - name: scalar_quantization_10
        type: scalar_quantization
        intervals: 10

  table_5:
    kind: filter_grid
    output_dir: results/experiments/table_5
    enabled: false
    note: "Enable only after confirming the exact Table 5 flow in the current code/article mapping."

  table_6:
    kind: split_eval
    output_dir: results/experiments/table_6
    filter:
      name: adaptive_quantization
      type: adaptive_quantization
    slices:
      - name: Training
        start: 0
        end: 4500
      - name: Validation
        start: 4500
        end: 5500
```

## Observação sobre Table 5

Se o código atual não tiver uma Table 5 implementada, não inventar fluxo.

A ação correta é:

```text
1. mapear se existe script/config/resultado da Table 5;
2. se existir, migrar para filter_grid ou split_eval;
3. se não existir, registrar como não implementada;
4. não criar runner novo só para preencher lacuna.
```

---

# 4. Runner único para `filter_grid`

Se já existir um runner para Table 7/8 que avalia múltiplos filtros, reaproveitar esse runner.

Não criar outro runner para Table 3/4.

O runner `filter_grid` deve ser genérico o suficiente para suportar:

```text
quantization filters
smoothing filters
nonuniform filters
scalar quantization intervals
```

O contrato esperado:

```python
run_filter_grid(config)
```

Deve suportar filtros como:

```yaml
- name: scalar_quantization_6
  type: scalar_quantization
  intervals: 6

- name: nonuniform_quantization
  type: nonuniform_quantization

- name: cross_7x7
  type: cross_mean
  radius: 3
```

A factory de filtros deve centralizar a construção dos filtros.

---

# 5. Factory de filtros mínima

A factory deve suportar os filtros necessários para Tabelas 3–8.

Tipos mínimos:

```text
scalar_quantization
nonuniform_quantization
nonuniform_quantization_legacy
adaptive_quantization
cross_mean
diamond_mean
box_mean
proposed_filter
```

Mas a factory deve continuar simples.

Evitar criar uma classe para cada filtro.

Preferir função simples:

```python
def build_filter(config):
    filter_type = config["type"]

    if filter_type == "scalar_quantization":
        ...

    if filter_type == "nonuniform_quantization":
        ...

    if filter_type == "adaptive_quantization":
        ...

    ...
```

Retorno esperado:

```python
filter_name, filter_fn, metadata
```

---

# 6. Outputs padronizados

## Table 3

CSV esperado:

```csv
filter_name,filter_type,TP,FN,FP,recall_percent,precision_percent,f1_percent
scalar_quantization_2,scalar_quantization,0,0,0,0.0,0.0,0.0
nonuniform_quantization,nonuniform_quantization,0,0,0,0.0,0.0,0.0
```

## Table 4

CSV esperado:

```csv
filter_name,filter_type,intervals,interval_size,TP,FN,FP,recall_percent,precision_percent,f1_percent
scalar_quantization_2,scalar_quantization,2,128,0,0,0,0.0,0.0,0.0
scalar_quantization_3,scalar_quantization,3,85,0,0,0,0.0,0.0,0.0
```

## Table 6

CSV esperado:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
Training,0,0,0,0.0,0.0,0.0
Validation,0,0,0,0.0,0.0,0.0
```

## JSON comum

Todos devem gerar:

```json
{
  "experiment_id": "table_4",
  "kind": "filter_grid",
  "config": {},
  "metrics": []
}
```

---

# 7. Scripts

As Tabelas 3–6 devem rodar pelo mesmo ponto de entrada:

```bash
python scripts/run_experiment.py --experiment table_3
python scripts/run_experiment.py --experiment table_4
python scripts/run_experiment.py --experiment table_5
python scripts/run_experiment.py --experiment table_6
```

Não criar:

```text
scripts/experiments/table_3.py
scripts/experiments/table_4.py
scripts/experiments/table_5.py
scripts/experiments/table_6.py
```

Scripts antigos devem ser removidos ou depreciados:

```text
scripts/article_reproduction/table_3.py
scripts/article_reproduction/table_4.py
scripts/article_reproduction/table_4_mnist.py
scripts/article_reproduction/table_6.py
```

Se mantidos temporariamente, devem ser wrappers mínimos e não devem aparecer na documentação principal.

---

# 8. YAMLs antigos

Após migrar para `configs/experiments.yaml`, remover ou depreciar:

```text
configs/article_reproduction/mnist_table_3.yaml
configs/article_reproduction/mnist_table_4.yaml
configs/article_reproduction/mnist_table_6.yaml
configs/mnist_table_3.yaml
configs/mnist_table_4.yaml
configs/mnist_table_6.yaml
```

Não manter YAML antigo “só por garantia” se ele não for usado por teste ou documentação oficial.

---

# 9. Remoção de Markdown e comparação com artigo

Remover do fluxo das Tabelas 3–6:

```text
write_markdown_table
format_percent usado apenas para Markdown
results_md
comparison_md
table_*.md
Reference
Our
Delta
ARTICLE_TABLE_*
```

O runtime deve apenas executar e salvar métricas locais.

Comparação com valores do artigo deve ficar fora do runner, em documentação ou análise manual.

---

# 10. Critérios de aceitação

## Execução

- [ ] `python scripts/run_experiment.py --experiment table_3` executa ou falha com erro claro se assets/modelo não existirem.
- [ ] `python scripts/run_experiment.py --experiment table_4` executa ou falha com erro claro se assets/modelo não existirem.
- [ ] `python scripts/run_experiment.py --experiment table_6` executa pelo fluxo unificado.
- [ ] Table 5 é migrada se existir; se não existir, fica explicitamente marcada como não implementada.

## Config

- [ ] `configs/experiments.yaml` contém `table_3`.
- [ ] `configs/experiments.yaml` contém `table_4`.
- [ ] `configs/experiments.yaml` contém `table_6`.
- [ ] Configs antigas equivalentes foram removidas ou depreciadas.
- [ ] Não há duplicação de `checkpoint_dir`, `epsilon`, `clip_min`, `clip_max` sem necessidade.
- [ ] Defaults comuns são reaproveitados.

## Scripts

- [ ] Não existe script ativo por tabela com lógica própria.
- [ ] Scripts antigos das Tabelas 3–6 foram removidos ou viraram wrappers depreciados.
- [ ] Documentação principal usa apenas `scripts/run_experiment.py`.

## Runner

- [ ] Table 3 usa `filter_grid`.
- [ ] Table 4 usa `filter_grid`.
- [ ] Table 6 usa `split_eval`.
- [ ] O `filter_grid` reutiliza adversariais e predições para todos os filtros.
- [ ] Não há runner específico para Table 3.
- [ ] Não há runner específico para Table 4.
- [ ] Não há runner específico para Table 6 se `split_eval` já existir.

## Outputs

- [ ] Outputs vão para `results/experiments/table_3/`.
- [ ] Outputs vão para `results/experiments/table_4/`.
- [ ] Outputs vão para `results/experiments/table_6/`.
- [ ] Cada experimento gera `metrics.csv`.
- [ ] Cada experimento gera `metrics.json`.
- [ ] Nenhum `.md` é gerado.

## Código

- [ ] Não há `ARTICLE_TABLE_*` no runtime das Tabelas 3–6.
- [ ] Não há `write_markdown_table` no fluxo das Tabelas 3–6.
- [ ] Não há `results_md` ou `comparison_md`.
- [ ] Cálculo de métricas não foi reimplementado.
- [ ] Filtros não tiveram a matemática alterada.

---

# 11. Testes sugeridos

## Testes de config

- [ ] `configs/experiments.yaml` possui `table_3`.
- [ ] `configs/experiments.yaml` possui `table_4`.
- [ ] `configs/experiments.yaml` possui `table_6`.
- [ ] `table_3.kind == filter_grid`.
- [ ] `table_4.kind == filter_grid`.
- [ ] `table_6.kind == split_eval`.

## Testes de factory

- [ ] `scalar_quantization` é construído corretamente.
- [ ] `nonuniform_quantization` é construído corretamente.
- [ ] `nonuniform_quantization_legacy` é construído corretamente.
- [ ] `adaptive_quantization` é construído corretamente.
- [ ] tipo desconhecido gera erro claro.

## Testes de runner

- [ ] `filter_grid` avalia múltiplos filtros.
- [ ] `filter_grid` gera FGSM uma única vez por execução.
- [ ] `split_eval` avalia múltiplos splits.
- [ ] outputs são `metrics.csv` e `metrics.json`.

## Testes de ausência

- [ ] scripts principais não contêm `write_markdown_table`.
- [ ] scripts principais não contêm `ARTICLE_TABLE_`.
- [ ] scripts principais não contêm `comparison_md`.
- [ ] scripts principais não contêm `results_md`.

---

# 12. Plano de implementação sugerido

1. Inventariar scripts/configs/results das Tabelas 3–6.
2. Confirmar o papel exato da Table 5 no código atual.
3. Adicionar `table_3`, `table_4` e `table_6` em `configs/experiments.yaml`.
4. Reaproveitar defaults existentes.
5. Expandir `filter_grid` para suportar filtros de quantização.
6. Expandir factory para suportar `scalar_quantization`, `nonuniform_quantization` e `nonuniform_quantization_legacy`.
7. Garantir que `filter_grid` reutiliza FGSM/predições.
8. Garantir que `split_eval` cobre Table 6.
9. Remover scripts antigos das Tabelas 3–6 ou transformá-los em wrappers depreciados.
10. Remover YAMLs antigos das Tabelas 3–6.
11. Remover geração Markdown desses fluxos.
12. Atualizar `scripts/README.md`.
13. Atualizar README principal, se necessário.
14. Criar testes de config/factory/runner.
15. Rodar os comandos oficiais das Tabelas 3, 4 e 6.
16. Registrar Table 5 como migrada ou não implementada.

---

# Definição de pronto

A Spec 6 está concluída quando:

```text
Table 3 roda pelo runner unificado
Table 4 roda pelo runner unificado
Table 6 roda pelo runner unificado
Table 5 está mapeada, migrada ou explicitamente marcada como não implementada
scripts antigos foram removidos ou depreciados
YAMLs redundantes foram removidos
outputs são apenas CSV/JSON
não há Markdown no runtime
não há comparação hardcoded com o artigo
a quantidade total de arquivos relacionados às Tabelas 3–6 diminuiu
```

O resultado esperado é um fluxo inicial de reprodução mais simples, onde as Tabelas 3–6 usam os mesmos blocos centrais já usados pelas Tabelas 7–9.