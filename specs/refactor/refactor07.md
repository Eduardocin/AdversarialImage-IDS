# Spec 7 — Limpeza destrutiva de arquivos desnecessários

## Status

Proposto.

---

## Contexto

Após várias etapas de refatoração, o projeto ganhou novos runners, novas configs, novos scripts, novas pastas e novos fluxos. Parte disso ajudou a padronizar a reprodução das tabelas, mas também deixou o repositório com muitos arquivos que provavelmente não são mais necessários.

O objetivo agora é fazer uma limpeza agressiva, mas controlada.

Esta spec não tem como foco criar novos fluxos. O foco é **excluir, arquivar ou consolidar tudo que não é necessário para o objetivo atual do projeto**.

O objetivo atual do projeto é:

```text
1. reproduzir os resultados principais do artigo;
2. manter uma base simples para experimentos adicionais;
3. evitar múltiplos caminhos para fazer a mesma coisa.
```

---

## Objetivo

Remover arquivos, scripts, configs, outputs e módulos que não são mais necessários após a refatoração.

Ao final desta spec:

- o projeto deve ter menos arquivos;
- deve existir um caminho oficial claro de execução;
- configs antigas e duplicadas devem ser removidas;
- scripts antigos e redundantes devem ser removidos ou movidos para legado;
- outputs antigos versionados devem ser removidos;
- código morto deve ser apagado;
- documentação deve apontar apenas para o fluxo atual;
- qualquer coisa mantida como legado deve estar claramente isolada.

---

## Princípios

1. Se não é usado pelo fluxo oficial, teste ou documentação atual, deve ser removido.
2. Se é legado, deve ficar em uma área explicitamente legada ou ser removido.
3. Se existem dois arquivos fazendo quase a mesma coisa, manter apenas um.
4. Se um YAML repete outro YAML, consolidar ou remover.
5. Se um script só chama outro script sem agregar valor, remover.
6. Se um resultado pode ser regenerado, não precisa estar versionado.
7. Se um output antigo é `.md`, `summary`, `diagnostic` ou `comparison`, remover.
8. Se uma função não é importada em nenhum fluxo oficial, remover.
9. Se um módulo existe apenas por causa de uma refatoração anterior, mas não simplifica o fluxo, consolidar.
10. A PR deve demonstrar redução real de arquivos.

---

## Fora de escopo

Esta spec não deve alterar:

- matemática dos filtros;
- cálculo de métricas;
- geração FGSM;
- arquitetura dos modelos;
- resultados experimentais esperados;
- datasets;
- pesos/checkpoints necessários para rodar experimentos.
- Não altere qualquer coisa referentes a table 10

Esta spec também não deve criar novos experimentos.

---

# 1. Inventário obrigatório

Antes de remover qualquer arquivo, fazer um inventário.

## Comandos sugeridos

```bash
find configs -type f | sort
find scripts -type f | sort
find src/deepdetector -type f | sort
find tests -type f | sort
find results -type f | sort
find reproduction_notes -type f | sort
```

Também procurar arquivos suspeitos:

```bash
find . -type f | grep -E "summary|diagnostic|comparison|article_reproduction|table_.*\\.md|\\.bak|old|legacy|tmp|debug"
find results -type f | grep -E "\\.md$|summary|diagnostic|comparison"
find configs -type f | grep -E "article|table_|mnist_|imagenet_|attack|filter|training"
find scripts -type f | grep -E "article|table_|mnist|imagenet|old|legacy"
```

Classificar cada arquivo em uma das categorias:

```text
manter
remover
mover_para_legacy
consolidar
avaliar
```

A PR deve incluir uma tabela ou checklist com a decisão.

Exemplo:

| Arquivo/Pasta | Decisão | Motivo |
| --- | --- | --- |
| `configs/experiments.yaml` | manter | config oficial |
| `configs/article_reproduction/` | remover | substituído por `configs/experiments.yaml` |
| `scripts/run_experiment.py` | manter | ponto de entrada oficial |
| `scripts/article_reproduction/` | remover/mover | fluxo legado |
| `results/**/*.md` | remover | outputs antigos regeneráveis |
| `reproduction_notes/` | avaliar | documentação histórica |

---

# 2. Definir o que é oficial

Antes de excluir, declarar explicitamente o fluxo oficial.

## Fluxo oficial mínimo

```text
configs/experiments.yaml
scripts/run_experiment.py
src/deepdetector/experiments/runner.py
src/deepdetector/filters/
src/deepdetector/evaluation/
src/deepdetector/paths.py ou src/deepdetector/io/paths.py
tests/
```

## Comando oficial

```bash
python scripts/run_experiment.py --experiment table_3
python scripts/run_experiment.py --experiment table_4
python scripts/run_experiment.py --experiment table_6
python scripts/run_experiment.py --experiment table_7
python scripts/run_experiment.py --experiment table_8
python scripts/run_experiment.py --experiment table_9
```

Se algum experimento ainda não estiver migrado, ele deve ser marcado como:

```text
não migrado
legado
fora do fluxo oficial
```

---

# 3. Remoção de configs desnecessárias

## Objetivo

Reduzir configs duplicadas e manter apenas as configurações oficiais.

## Manter

Preferencialmente:

```text
configs/experiments.yaml
```

Opcionalmente, se realmente necessário:

```text
configs/assets.yaml
```

## Remover ou migrar

Remover depois de migrar o conteúdo útil para `configs/experiments.yaml`:

```text
configs/article_reproduction/
configs/attacks/
configs/filters/
configs/training/
configs/mnist_fgsm.yaml
configs/imagenet_googlenet_fgsm.yaml
configs/mnist_table_3.yaml
configs/mnist_table_4.yaml
configs/mnist_table_6.yaml
configs/imagenet_*.yaml
```

## Regra

Um YAML só deve existir se:

```text
é lido por scripts/run_experiment.py
ou é lido por um utilitário oficial
ou é usado em teste
ou é documentação intencional
```

Caso contrário, remover.

---

# 4. Remoção de scripts desnecessários

## Objetivo

Reduzir scripts paralelos e manter apenas pontos de entrada claros.

## Manter

Preferencialmente:

```text
scripts/run_experiment.py
scripts/README.md
scripts/dev/smoke_test.py
```

Opcionalmente:

```text
scripts/prepare_assets.py
```

se houver necessidade real de preparar MNIST/ImageNet/assets.

## Remover ou mover para legado

Avaliar e remover/mover:

```text
scripts/article_reproduction/
scripts/experiments/table_*.py
scripts/mnist/
scripts/imagenet/
```

## Regra

Um script só deve permanecer em `scripts/` se for um ponto de entrada oficial ou utilitário claro.

Scripts que apenas fazem uma variação de experimento devem ser substituídos por:

```bash
python scripts/run_experiment.py --experiment <nome>
```

## Scripts legados

Se for necessário manter algum script antigo, mover para:

```text
scripts/legacy/
```

com aviso claro no topo:

```python
"""Deprecated legacy script. Prefer scripts/run_experiment.py."""
```

Mas a preferência é remover, não arquivar.

---

# 5. Remoção de outputs versionados desnecessários

## Objetivo

Remover outputs antigos que podem ser regenerados.

## Remover

```text
results/**/*.md
results/**/summary.md
results/**/diagnostic.md
results/**/comparison.md
results/**/table_*.md
results/mnist/article_reproduction/
results/imagenet/article_reproduction/
results/mnist/fgsm_smoke/
results/mnist/fgsm/
results/mnist/detector/
results/mnist/entropy/
```

Atenção: remover apenas se esses diretórios forem outputs regeneráveis e não forem usados como fixtures de teste.

## Manter

Se necessário para testes pequenos, mover para:

```text
tests/fixtures/
```

com tamanho mínimo.

## Regra

`results/` não deve ser fonte de verdade do projeto.

O resultado oficial deve ser gerado por execução:

```text
results/experiments/<experiment_id>/metrics.csv
results/experiments/<experiment_id>/metrics.json
```

---

# 6. Remoção de documentação histórica redundante

## Objetivo

Reduzir notas antigas que não são mais usadas ou que duplicam README/specs.

Avaliar:

```text
reproduction_notes/
docs/
README antigo
scripts/README.md
```

## Manter

Documentação que explica:

```text
como rodar o fluxo oficial
como preparar assets
qual experimento corresponde a qual tabela
decisões importantes de reprodução
```

## Remover ou arquivar

Documentação que:

```text
descreve scripts antigos
aponta para configs antigas
explica outputs Markdown antigos
duplica informações de specs atuais
contém planos obsoletos
```

## Sugestão

Manter no máximo:

```text
README.md
scripts/README.md
docs/paper_mapping.md
docs/decisions.md
```

Evitar várias notas de reprodução soltas se elas não forem consultadas.

---

# 7. Remoção de código morto

## Objetivo

Apagar funções e módulos não usados.

Procurar:

```text
write_markdown_table
format_percent
percent_delta
ARTICLE_TABLE_
ARTICLE_OUTPUT_DIR
comparison_md
results_md
diagnostic
summary
paper_reference
reference_values
```

Também procurar módulos antigos:

```text
src/deepdetector/evaluation/article_reproduction.py
src/deepdetector/experiments/table_*.py
src/deepdetector/experiments/*_old.py
src/deepdetector/io/* se usado só uma vez
```

## Regra

Se uma função não é usada pelo fluxo oficial nem por testes, remover.

Se um módulo inteiro virou apenas compatibilidade, remover ou consolidar.

---

# 8. Remoção de paths antigos

## Objetivo

Eliminar paths antigos ou inconsistentes.

Remover referências a:

```text
results/mnist/article_reproduction
results/imagenet/article_reproduction
ARTICLE_OUTPUT_DIR
configs/article_reproduction
scripts/article_reproduction
table_*.md
summary.md
diagnostic.md
comparison.md
```

Os paths oficiais devem ser:

```text
configs/experiments.yaml
results/experiments/<experiment_id>/
artifacts/
```

---

# 9. Limpeza de testes

## Objetivo

Atualizar testes para o fluxo oficial.

Remover ou adaptar testes que dependem de:

```text
scripts/article_reproduction/
configs/article_reproduction/
outputs .md
ARTICLE_TABLE_*
results antigos
```

Testes devem validar:

```text
config oficial
runner oficial
factory de filtros
outputs CSV/JSON
ausência de Markdown
ausência de paths antigos
```

## Regra

Não manter teste para fluxo que foi removido.

Se o teste existe apenas para manter legado vivo, remover junto com o legado.

---

# 10. Checklist de remoção

## Configs

- [ ] Remover configs antigas substituídas por `configs/experiments.yaml`.
- [ ] Remover configs de attack/filter/training se não forem necessárias.
- [ ] Remover configs `article_reproduction`.
- [ ] Remover configs de tabelas antigas duplicadas.
- [ ] Garantir que README não referencia configs removidas.

## Scripts

- [ ] Remover scripts por tabela.
- [ ] Remover scripts `article_reproduction`.
- [ ] Remover scripts MNIST/ImageNet que foram substituídos.
- [ ] Mover utilitários necessários para `scripts/prepare_assets.py` ou manter com justificativa.
- [ ] Garantir que README não referencia scripts removidos.

## Results

- [ ] Remover `.md` de `results`.
- [ ] Remover summaries antigos.
- [ ] Remover diagnostics antigos.
- [ ] Remover comparisons antigos.
- [ ] Remover diretórios antigos de article reproduction.
- [ ] Garantir que outputs oficiais são regeneráveis.

## Código

- [ ] Remover `write_markdown_table`.
- [ ] Remover `ARTICLE_TABLE_*`.
- [ ] Remover `ARTICLE_OUTPUT_DIR`.
- [ ] Remover `percent_delta` se usado apenas para comparação com artigo.
- [ ] Remover funções de relatório Markdown.
- [ ] Remover módulos inteiros sem uso.

## Docs

- [ ] Atualizar README principal.
- [ ] Atualizar `scripts/README.md`.
- [ ] Remover instruções antigas.
- [ ] Manter apenas documentação útil e atual.

---

# 11. Testes/checagens obrigatórias

## Checagens de arquivos

```bash
find results -name "*.md"
find . -type f | grep -E "diagnostic|summary|comparison"
find scripts -type f | grep "article_reproduction"
find configs -type f | grep "article_reproduction"
```

Esses comandos devem retornar vazio ou apenas arquivos explicitamente justificados.

## Checagens de strings

```bash
grep -R "ARTICLE_TABLE_" src scripts configs tests
grep -R "ARTICLE_OUTPUT_DIR" src scripts configs tests
grep -R "write_markdown_table" src scripts configs tests
grep -R "comparison_md" src scripts configs tests
grep -R "results_md" src scripts configs tests
grep -R "configs/article_reproduction" .
grep -R "scripts/article_reproduction" .
```

Essas strings não devem aparecer no runtime oficial.

## Execução mínima

```bash
python scripts/run_experiment.py --help
python scripts/run_experiment.py --experiment table_3
python scripts/run_experiment.py --experiment table_4
python scripts/run_experiment.py --experiment table_6
python scripts/run_experiment.py --experiment table_7
python scripts/run_experiment.py --experiment table_8
python scripts/run_experiment.py --experiment table_9
```

Se algum experimento depender de checkpoint/assets ausentes, o erro deve ser claro.

---

# 12. Critérios de aceitação

## Redução real

A PR deve mostrar contagem antes/depois:

```bash
find configs -type f | wc -l
find scripts -type f | wc -l
find results -type f | wc -l
find results -name "*.md" | wc -l
find src/deepdetector -type f | wc -l
```

A expectativa é reduzir:

```text
configs
scripts
outputs versionados
Markdowns
código morto
```

Se algum número aumentar, justificar.

---

## Fluxo oficial

- [ ] Existe um único caminho oficial para rodar experimentos.
- [ ] O caminho oficial está documentado.
- [ ] Scripts antigos não aparecem na documentação principal.
- [ ] Configs antigas não aparecem na documentação principal.
- [ ] Outputs oficiais são `metrics.csv` e `metrics.json`.

---

## Remoções

- [ ] Configs duplicadas foram removidas.
- [ ] Scripts redundantes foram removidos.
- [ ] Outputs antigos foram removidos.
- [ ] Código morto foi removido.
- [ ] Paths antigos foram removidos.
- [ ] Testes antigos foram atualizados ou removidos.

---

## Segurança

- [ ] Nenhum checkpoint/modelo necessário foi removido.
- [ ] Nenhum dataset pequeno necessário para teste foi removido.
- [ ] Nenhum fixture usado por teste foi removido sem substituição.
- [ ] Nenhum código usado pelo fluxo oficial foi removido.
- [ ] Testes passam após limpeza.

---

# 13. Plano de implementação sugerido

1. Criar inventário de arquivos.
2. Classificar arquivos por categoria.
3. Atualizar README para apontar somente para fluxo oficial.
4. Remover configs obsoletas.
5. Remover scripts obsoletos.
6. Remover outputs antigos regeneráveis.
7. Remover código Markdown/diagnóstico/comparação com artigo.
8. Remover paths antigos.
9. Atualizar testes.
10. Rodar grep de strings proibidas.
11. Rodar testes.
12. Rodar comandos oficiais ou validar erro claro.
13. Registrar contagem antes/depois na PR.

---

# 14. Definição de pronto

A Spec 7 está concluída quando:

```text
o repositório tem menos arquivos desnecessários
o fluxo oficial está claro
configs duplicadas foram removidas
scripts redundantes foram removidos
outputs antigos foram removidos
código morto foi removido
paths antigos foram removidos
documentação aponta apenas para o fluxo atual
testes foram atualizados
```

O resultado esperado é um projeto mais enxuto, com menos ruído e menor custo de manutenção.