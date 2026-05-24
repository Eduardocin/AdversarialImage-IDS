# SPEC — Filtro de Predição Limpa para Experimentos ImageNet

## Objetivo

Garantir que os experimentos ImageNet do DeepDetector avaliem apenas imagens que o modelo GoogLeNet classifica corretamente antes do ataque.

Essa regra deve ser aplicada antes de gerar métricas da Table 4 e Table 7.

## Contexto

Durante a validação limpa, algumas imagens do treino exe:  classes goldfish e pineapple foram classificadas com labels diferentes do label configurado.

Exemplo:

goldfish:
- configured_label = 1
- matches = 44/50

pineapple:
- configured_label = 953
- matches = 42/50

Essas imagens não devem ser apagadas do dataset, mas devem ser ignoradas durante a avaliação experimental quando `clean_pred != true_label`.

## Justificativa

O objetivo dos experimentos é medir:

imagem limpa corretamente classificada
→ ataque adversarial altera a predição
→ filtro tenta detectar a perturbação

Se a imagem limpa já está classificada incorretamente, o resultado fica ambíguo, porque não é possível separar erro natural do modelo, efeito do ataque e efeito do filtro.

O código original faz essa filtragem com a regra:

```python
if pred_class != true_class:
    continue
````

## Regra obrigatória

Para cada imagem:

```python
clean_pred = model.predict(clean_image)

if clean_pred != true_label:
    skipped_wrong_baseline += 1
    registrar_diagnostico(...)
    continue
```

A imagem deve ser pulada antes da geração/avaliação do ataque.

## Regra do ataque ImageNet

O FGSM ImageNet da reprodução deve seguir o script base `Train_FGSM_ImageNet.py`.

O ataque deve ser Caffe-only:

```text
original_data = transformer.preprocess(...)
pred_class = model.predict(original_data)

if pred_class != true_label:
    skipped_wrong_baseline += 1
    continue

gradient = model.gradient(original_data, pred_class)
adversarial_data = original_data + 1.0 * sign(gradient)
clip adversarial_data to [0, 255]
```

Não usar TensorFlow, CleverHans, placeholders ou logits TF para gerar os adversariais ImageNet da reprodução.

## Onde aplicar

Aplicar esta regra em:

* Table 4 ImageNet
* Table 7 ImageNet
* futuros experimentos ImageNet com FGSM
* diagnósticos de ataque ImageNet

## Onde não aplicar

Não remover fisicamente a imagem do dataset.

Não deletar imagens das pastas.

Não alterar o manifest.

A filtragem deve acontecer apenas em tempo de execução do experimento.

## Contadores obrigatórios

Todo script afetado deve registrar:

* total_images
* clean_correct
* skipped_wrong_baseline
* attack_attempted
* attack_success
* disturbed_failure

## Diagnóstico por imagem

Gerar ou atualizar CSV diagnóstico com colunas:

```text
image_id,
class_name,
true_label,
clean_pred,
clean_correct,
was_skipped,
skip_reason
```

Valores possíveis para `skip_reason`:

```text
none
wrong_clean_prediction
```

## Fluxo correto

```text
1. Carregar imagem limpa.
2. Fazer preprocess Caffe/GoogLeNet.
3. Fazer predição limpa.
4. Se clean_pred != true_label:
      skipped_wrong_baseline += 1
      registrar no diagnóstico
      pular imagem.
5. Se clean_pred == true_label:
      gerar FGSM.
6. Se adv_pred == clean_pred:
      disturbed_failure += 1
      pular avaliação da detecção.
7. Se adv_pred != clean_pred:
      avaliar filtro/detector.
```

## Critérios de aceitação

A implementação é aceita se:

* imagens com `clean_pred != true_label` não entram no cálculo de TP, FN ou FP;
* essas imagens aparecem no diagnóstico como `wrong_clean_prediction`;
* `skipped_wrong_baseline` é maior ou igual ao número de imagens limpas incorretas;
* as imagens não são apagadas do dataset;
* Table 4 e Table 7 exibem `skipped_wrong_baseline` no CSV final;
* o script imprime um resumo com `total_images`, `clean_correct` e `skipped_wrong_baseline`.
* o ataque ImageNet usado por Table 4 e Table 7 é gerado em espaço Caffe `CHW/BGR/0..255`, com `epsilon_255=1.0`;
* o caminho principal de `fgsm_imagenet.py` não importa TensorFlow para gerar FGSM ImageNet da reprodução.

## Exemplo esperado

Se rodarmos 50 imagens de goldfish e apenas 44 forem previstas como label 1:

```text
total_images = 50
clean_correct = 44
skipped_wrong_baseline = 6
```

Somente as 44 imagens corretamente classificadas devem seguir para geração do ataque.

## Fora de escopo

Esta SPEC não deve:

* alterar o dataset;
* modificar labels;
* implementar novos ataques;
* alterar Table 7;
* alterar a quantização;
* alterar o wrapper do GoogLeNet.

O foco é apenas garantir que imagens limpas incorretas sejam puladas corretamente.

