# Spec 8 — Reorganizacao de evaluation e deduplicacao de helpers

## Status

Proposto.

---

## Objetivo

Reorganizar `src/deepdetector/evaluation` para reduzir ruido estrutural e
eliminar duplicacoes evidentes de helpers, mantendo o comportamento identico.

---

## Contexto

A pasta `evaluation` hoje mistura:

- helpers genericos de metricas e avaliacao,
- fluxos especificos das tabelas ImageNet,
- materializacao oficial da Table 10,
- funcoes repetidas entre Table 4/6 e Table 7/8,
- helpers duplicados de Keras learning phase.

Existe uma subpasta `evaluation/tables` apenas para `table_10.py`, o que
introduz uma camada extra sem ganho de organizacao real. Alem disso, ha
duplicacoes de:

- conversoes de labels/datasets;
- conversoes HWC/CHW e escala 0-255;
- calculo de precision/recall/F1;
- helpers de input/preprocessamento Caffe;
- helper de learning phase do Keras.

---

## Regras de negocio

1. Nao alterar a semantica das metricas nem a logica de avaliacao.
2. Nao alterar o schema de outputs nem os caminhos de resultados.
3. Manter compatibilidade com o runner da Table 10.
4. Evitar introduzir dependencias novas ou ciclos de importacao.
5. Todos os erros e validacoes de entrada devem manter comportamento
   equivalente (ou mais explicito) sem mudar os casos de falha.

---

## Requisitos funcionais

### 1) Reorganizacao de `evaluation/tables`

- Mover `src/deepdetector/evaluation/tables/table_10.py` para
  `src/deepdetector/evaluation/table_10.py`.
- Remover a subpasta `evaluation/tables/` se ela ficar vazia.
- Atualizar todos os imports internos para o novo caminho.

### 2) Helpers compartilhados

Criar um ou mais modulos de suporte em `evaluation` para substituir duplicacoes
atuais. Sugestao minima:

- `evaluation/imagenet_eval_utils.py` (nome ajustavel)
  - `label_to_int()`
  - `iter_dataset()`
  - `image_to_chw_255()` e `restore_from_chw_255()`
  - `predict_label()` (wrapper para `predict_caffe_label` com checagem de shape)
  - `metrics_prf1()` para precision/recall/F1
  - `article_model_input()` (wrapper para preprocessamento Caffe quando usado)

- `evaluation/keras_utils.py` (nome ajustavel)
  - `learning_phase_feed()` reutilizado por `adversarial.py` e
    `article_reproduction.py`.

### 3) Deduplicacao nos modulos existentes

- Table 7 e Table 8 devem consumir os helpers comuns de dataset/conversao
  e metricas.
- Table 4 e Table 6 devem consumir helpers comuns para `predict_label`,
  `article_model_input` e `metrics_prf1`.
- `adversarial.py` e `article_reproduction.py` devem usar o helper comum
  de learning phase.

### 4) Documentacao e export

- Ajustar `evaluation/__init__.py` caso seja usado para expor symbols.
- Atualizar README somente se mencionar caminhos ou responsabilidades de
  `evaluation/tables`.

---

## Requisitos nao funcionais

- Mudancas pequenas e rastreaveis.
- Mantem estilo atual (tipagem existente, docstrings curtas).
- Sem novos warnings ou imports de TensorFlow/CleverHans fora do necessario.
- Preferir helpers puros e sem estado.

---

## Criterios de aceitacao

1. Nao existe mais `src/deepdetector/evaluation/tables/` (ou a pasta fica
   apenas com `__init__.py` e sem uso).
2. `TABLE_10_SCHEMA` e `run_table_10_group` continuam acessiveis pelo
   novo caminho e o runner usa o import atualizado.
3. Table 7 e Table 8 usam um unico conjunto de helpers para:
   - conversao de labels,
   - iteracao de dataset,
   - conversao HWC/CHW e escala 0-255,
   - calculo de precision/recall/F1.
4. Table 4 e Table 6 usam helpers comuns para:
   - preprocessamento Caffe quando aplicavel,
   - predicao unica,
   - calculo de precision/recall/F1.
5. `adversarial.py` e `article_reproduction.py` compartilham
   `learning_phase_feed()`.
6. Testes existentes passam sem alterar resultados esperados.
7. Qualquer novo helper tem testes focados (quando fizer sentido) para:
   - conversao CHW/HWC,
   - iteracao de dataset com/sem adversarial,
   - calculo de metricas.

---

## Casos de erro

- Dataset com shapes incompatíveis deve continuar gerando `ValueError`.
- Conversoes HWC/CHW com layout invalido devem manter a falha explicita.
- `table_10` deve falhar com mensagens equivalentes se faltar config.
- `learning_phase_feed()` deve retornar `{}` quando Keras nao estiver
  disponivel.

---

## Fora de escopo

- Alterar matematica dos filtros.
- Modificar output de qualquer experimento.
- Reescrever experiment runners.
- Migrar configs ou scripts fora de evaluation.
- Introduzir nova estrutura de pacotes fora de `evaluation/`.
