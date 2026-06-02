# Investigacao: CW L2 kappa=0.5 / Inception v3 / ImageNet na Table 10

## Contexto

Esta investigacao parte do log descrito no texto anexado:

```text
clean_errors=7
attack_failures=0
tp=93
fn=0
fp=93
```

A linha de referencia do artigo para **CW L2 (kappa=0.5) / Inception v3 / ImageNet** e:

```text
#F=0, TP=100, FN=0, FP=2, RTP=98, Recall=100%, Precision=98.04%, F1=99.01%
```

O objetivo aqui nao foi corrigir codigo, mas identificar os pontos mais provaveis de divergencia entre a reproducao atual e a linha 15 da Table 10.

## Status apos correcao

O filtro `article_final_detection_filter` foi atualizado para preservar explicitamente entradas Inception em `[-0.5, 0.5]`. O filtro agora converte essa escala centralizada para `[0, 255]` durante entropia, quantizacao e suavizacao, e restaura para `[-0.5, 0.5]` antes de retornar.

Validacao executada:

```text
pytest tests/test_table9.py::test_article_final_filter_is_registered_and_preserves_scales -q
pytest tests/test_table9.py -q
pytest tests/test_table10_runner.py::test_table10_inception_v3_row_computes_cw_metrics -q
```

Todos os comandos passaram. A causa de `FP=93` deve ser reavaliada em uma nova execucao real da row 15, porque a hipotese de incompatibilidade de escala do filtro foi corrigida.

## Resumo executivo

A suspeita inicial do texto colado era que `FP` poderia estar sendo incrementado junto com `TP`. No estado atual do codigo, essa suspeita **nao se confirma diretamente**: o avaliador da Table 10 calcula `false_positive` usando a imagem limpa filtrada, nao a adversarial.

O achado mais forte antes da correcao era outro: o filtro oficial (`proposed_detection_filter`) nao tratava corretamente tensores Inception no intervalo `[-0.5, 0.5]`. Ele identificava como "normalizada" apenas imagem com `min >= 0` e `max <= 1`. Como o pipeline Inception entrega imagens pre-processadas com valores negativos, o filtro passava a tratar esses tensores como se fossem escala 0-255, clipeava valores negativos para zero e podia alterar praticamente todas as imagens limpas. Esse ponto foi corrigido no filtro.

Ha tambem um segundo problema metodologico: depois do pull mais recente, a configuracao nao usa mais `dataset.n_samples: 100` nem amostragem aleatoria; ela usa quotas fixas por classe (`zebra=40`, `panda=40`, `cab=20`) e `shuffle: false`. Ainda assim, o avaliador descarta `clean_errors` sem reposicao. Se 7 imagens limpas forem classificadas errado, a execucao continua terminando com 93 pares validos. Isso explica `tp=93` em vez de `tp=100`, mesmo com ataque sem falhas.

## Evidencias no codigo

### 1. O FP nao esta duplicado diretamente do TP

No avaliador oficial da Table 10, o registro por amostra e montado em `src/deepdetector/evaluation/tables/table_10.py`:

```python
"detected": bool(filtered_adv_pred != adv_pred),
"corrected": bool(filtered_adv_pred == true_label),
"false_positive": bool(filtered_clean_pred != clean_pred),
```

Ou seja:

- `TP` depende de `filtered_adv_pred != adv_pred`;
- `FP` depende de `filtered_clean_pred != clean_pred`;
- nao ha, nesse ponto, incremento explicito de `FP` junto com `TP`.

O agregador em `src/deepdetector/evaluation/detector_metrics.py` tambem usa esses campos separados:

```python
if detected:
    counts["TP"] += 1
...
if false_positive:
    counts["FP"] += 1
```

Conclusao: se o log real vem desse fluxo, `fp=93` significa que o detector esta marcando 93 imagens limpas como adversariais, nao que `FP` esta sendo somado automaticamente junto com `TP`.

### 2. O filtro oficial parece incompatibilizado com entrada Inception `[-0.5, 0.5]`

A spec de Inception v3 exige `dataset.value_range: [-0.5, 0.5]`, e a config atual declara isso em `configs/experiments.yaml`:

```yaml
table_10_inception_v3:
  dataset:
    image_size: 299
    value_range: [-0.5, 0.5]
    shuffle: false
    class_quotas:
      zebra: 40
      panda: 40
      cab: 20
```

O loader da Table 10 chama `model.preprocess(...)`, e o wrapper Inception implementa:

```python
return (resized - 0.5).astype(np.float32)
```

Entao as imagens limpas avaliadas pelo detector chegam em `[-0.5, 0.5]`.

O problema aparece no filtro final em `src/deepdetector/filters/article_final.py`:

```python
def _is_normalized(image):
    return min(image) >= 0.0 and max(image) <= 1.0
```

Para Inception, qualquer pixel negativo faz `_is_normalized(...)` retornar `False`. Depois disso, o filtro passa por caminhos que assumem escala 0-255 e fazem clipe em `[0, 255]`. Na pratica, valores negativos da imagem Inception podem virar zero, e a imagem filtrada deixa de estar no mesmo dominio do modelo.

Impacto esperado:

- `filtered_clean_pred` muda em massa;
- `false_positive = filtered_clean_pred != clean_pred` vira `True` para quase todas as imagens limpas;
- `FP` cresce junto com o numero de pares validos, como no log `fp=93`.

Esta foi a principal hipotese para o resultado observado e agora deve ser validada novamente com a row 15 completa.

### 3. A selecao atual nao reabastece as 100 amostras apos `clean_errors`

Depois do pull mais recente, a config define 100 imagens por quotas fixas de classe:

```yaml
class_order: [zebra, panda, cab]
class_quotas:
  zebra: 40
  panda: 40
  cab: 20
```

O loader monta essas 100 candidatas antes da avaliacao. Depois, no loop de avaliacao:

```python
if clean_pred != true_label:
    discarded_clean_error = True
    continue
```

Essas amostras descartadas nao sao substituidas por novas candidatas. Portanto, se 7 das 100 imagens forem `clean_errors`, a execucao avalia apenas 93 ataques efetivos. Isso bate exatamente com:

```text
clean_errors=7
tp=93
fn=0
attack_failures=0
```

Para reproduzir a linha 15 do artigo, ha duas possibilidades metodologicas que precisam ficar explicitas:

1. selecionar 100 imagens ja corretamente classificadas pelo Inception v3 antes do ataque;
2. ou continuar sorteando candidatos ate obter 100 ataques efetivos, registrando quantos candidatos foram descartados.

O comportamento atual faz uma selecao deterministica de 100 candidatas por classe e depois descarta erros limpos, logo nao garante denominador 100 para `TP + FN`.

### 4. O CW L2 esta defense-unaware no fluxo atual

O ataque e gerado antes da aplicacao do filtro:

```python
adversarial_image = generate_attack(...)
...
filtered_clean = filter_fn(clean_image)
filtered_adv = filter_fn(adversarial_image)
```

Isso esta alinhado com a Table 10 normal do artigo, nao com a avaliacao defense-aware. Portanto, este nao parece ser o problema principal para a linha 15.

### 5. O dominio do CW L2 parece correto e agora esta explicito nas rows

`generate_cw_l2_attack` usa por padrao:

```python
clip_min=-0.5
clip_max=0.5
```

Depois do pull, as rows CW L2 do Inception tambem declaram explicitamente:

```yaml
clip_min: -0.5
clip_max: 0.5
```

Esse ponto parece coerente com o dominio esperado pelo wrapper Inception e nao parece explicar o `FP=93`.

## Diagnostico provavel do log

O log:

```text
clean_errors=7
attack_failures=0
tp=93
fn=0
fp=93
```

e consistente com este fluxo:

1. O runner carrega 100 imagens por quotas fixas: 40 zebra, 40 panda e 20 cab.
2. O Inception erra 7 imagens limpas.
3. Restam 93 imagens limpas corretas.
4. O CW L2 consegue mudar a predicao nas 93.
5. O detector detecta todos os 93 adversariais, portanto `tp=93`, `fn=0`.
6. O filtro tambem muda a predicao das 93 imagens limpas, provavelmente por tratar `[-0.5, 0.5]` como escala 0-255, portanto `fp=93`.

Isso explicaria tanto o denominador 93 quanto a precision de aproximadamente 50%.

## Gaps de teste

Os testes atuais cobrem que a row Inception chama o avaliador e que `FP` pode ser zero em um dummy model. Eles nao cobrem os dois pontos criticos da reproducao real:

- o filtro final preserva dominio e shape quando recebe imagem Inception em `[-0.5, 0.5]`;
- a Table 10 Inception garante 100 amostras validas, nao apenas 100 candidatas preselecionadas por quota.

Tambem nao ha um teste especifico que simule `filtered_clean_pred != clean_pred` para todas as amostras limpas por distorcao de escala do filtro.

## Recomendacoes

1. Corrigir ou adaptar o filtro para imagens Inception `[-0.5, 0.5]`. **Status: aplicado.**
   - Opcao segura: converter `[-0.5, 0.5]` para `[0, 1]` antes do filtro e converter de volta para `[-0.5, 0.5]` depois.
   - Evitar passar tensor Inception diretamente para `article_final_detection_filter` sem wrapper de escala.

2. Atualizar a especificacao antes da correcao, porque isso muda comportamento esperado.
   - A spec atual ainda descreve `dataset.n_samples: 100`, `dataset.shuffle: true` e seed de amostragem, enquanto a config/testes atuais usam quotas fixas e `shuffle: false`.
   - Definir explicitamente se a Table 10 Inception deve avaliar 100 candidatas preselecionadas por quota ou 100 amostras limpas corretamente classificadas.
   - Para aproximar a linha 15 do artigo, o esperado deve ser 100 amostras validas/clean-correct.

3. Adicionar testes unitarios focados.
   - Teste do filtro/wrapper: entrada `[-0.5, 0.5]` deve sair em `[-0.5, 0.5]`.
   - Teste do runner: quando ha clean errors, o loader/avaliador deve reabastecer ate obter 100 validas ou registrar explicitamente que o denominador efetivo e menor.
   - Teste de config/spec: alinhar `specs/features/table10_inception_v3.md` com a decisao final sobre quotas fixas versus amostragem.
   - Teste de metrica: `FP` deve depender apenas de `filtered_clean_pred != clean_pred`.

4. Registrar diagnosticos por amostra em execucoes longas.
   - Salvar `sample_index`, `true_label`, `clean_pred`, `adv_pred`, `filtered_clean_pred`, `filtered_adv_pred`, `detected`, `false_positive`, `discard_reason`.
   - Isso permitiria confirmar rapidamente se todos os `FP` vieram da imagem limpa filtrada.

## Checklist de validacao sugerido

Depois da correcao, a row 15 deve ser validada nesta ordem:

1. Rodar smoke test com 2 a 5 imagens e imprimir min/max de `clean_image`, `filtered_clean`, `adversarial_image`, `filtered_adv`.
2. Confirmar que imagens Inception permanecem em `[-0.5, 0.5]` depois do filtro. **Status unitario: coberto por teste.**
3. Confirmar que `FP` cai substancialmente antes de comparar com o artigo.
4. Confirmar se a execucao usa 100 amostras validas ou documentar denominador menor.
5. Rodar a row 15 completa.

## Conclusao

A causa mais provavel original do `fp=93` nao era duplicacao direta de `TP` como `FP`, mas sim incompatibilidade de escala entre o filtro final e o pre-processamento Inception. Essa incompatibilidade foi corrigida e precisa ser validada em execucao real. A causa mais provavel de `tp=93` continua sendo a selecao de 100 candidatas por quota seguida do descarte de 7 `clean_errors`, sem reposicao.

Para reproduzir a linha 15 do artigo, o proximo passo pendente e especificar e implementar a selecao de 100 amostras limpas corretamente classificadas antes de contabilizar os ataques, ou documentar explicitamente o denominador efetivo quando houver `clean_errors`.
