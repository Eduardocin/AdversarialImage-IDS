SPEC — Table 4 ImageNet
1. Objetivo

Implementar e validar a reprodução da Table 4 — ImageNet do artigo.

A Table 4 avalia o desempenho da detecção usando quantização escalar uniforme com diferentes números de intervalos. Para ImageNet, o objetivo é verificar se a quantização consegue detectar exemplos adversariais FGSM ao comparar a predição da imagem adversarial antes e depois do filtro.

Essa SPEC também serve como sanity check obrigatório antes da Table 7.

2. Motivação

A Table 7 apresentou resultado inválido:

tp = 0
fn = 0
n_high_entropy_adversarial = 0
disturbed_failure alto

Isso indica que a geração de adversariais, o modelo, o preprocessamento ou o dataset podem estar incorretos.

Portanto, antes da Table 7, a Table 4 deve responder:

O pipeline ImageNet + GoogLeNet + FGSM está funcionando?
3. Referência experimental

Segundo o artigo, a Table 4 usa ImageNet na fase de escolha de parâmetros do detector. O split de treino do ImageNet é composto pelas classes Goldfish, Pineapple e Clock.

Item	Valor esperado
Dataset	ImageNet local
Split	Training
Classes	Goldfish, Pineapple, Clock
Modelo	BVLC GoogLeNet / Caffe
Ataque	FGSM
Epsilon	1/255
Escala Caffe	epsilon_255 = 1.0
Preprocess Caffe	transpose RGB->CHW, raw_scale=255, channel_swap RGB->BGR, sem mean subtraction
Filtro	Quantização escalar uniforme
Intervalos	2, 3, 4, 5, 6, 7, 8, 9, 10
Métricas	Recall, Precision, F1

No projeto atual, a trilha ImageNet já prevê wrappers e utilitários para Caffe/GoogLeNet, além de scripts de reprodução para tabelas do artigo.

4. Dados esperados

Estrutura local recomendada:

data/imagenet/train/
├── goldfish/
├── pineapple/
└── digital_clock/

Labels esperados:

Classe	Label ImageNet
Goldfish	1
Pineapple	953
Digital clock	530

Esses labels aparecem também no código original da trilha ImageNet, onde as classes são chamadas separadamente para Goldfish, Pineapple e Clock. No layout local atual, a classe Clock deve usar o diretório `digital_clock`, alinhado ao synset ImageNet `n03196217`.

5. Entradas

O script deve aceitar:

python scripts/article_reproduction/table_4_imagenet.py \
  --data-root data/imagenet/train \
  --limit 50

Argumentos mínimos:

Argumento	Descrição
--data-root	Raiz das imagens locais
--limit	Número opcional de imagens para teste rápido
--epsilon	Default 1.0 em escala [0,255]
--output-dir	Default results/imagenet/article_reproduction/
6. Saídas

Arquivo principal:

results/imagenet/article_reproduction/table_4_imagenet.csv

Colunas esperadas:

intervals,tp,fn,fp,recall,precision,f1,
n_clean_total,n_clean_correct,n_attack_success,
disturbed_failure,skipped_wrong_baseline

Também gerar um arquivo de diagnóstico:

results/imagenet/article_reproduction/table_4_imagenet_diagnostics.csv

Colunas:

image_id,class_name,true_label,clean_pred,adv_pred,
clean_correct,attack_success,disturbed_failure,
entropy_clean,entropy_adv,fgsm_linf_255,fgsm_changed_pixels
7. Regras de negócio

Para cada imagem:

1. Carregar PNG local.
2. Aplicar preprocess do Caffe/GoogLeNet.
   O preprocess deve seguir o script original da trilha ImageNet:
   transpose para CHW, raw_scale=255 e channel_swap RGB->BGR.
   Não aplicar subtração de mean file para a Table 4.
3. Predizer imagem limpa.
4. Se clean_pred != true_label:
      skipped_wrong_baseline += 1
      pular ataque.
5. Gerar FGSM com epsilon_255 = 1.0.
   A geração ImageNet deve seguir o script base `Train_FGSM_ImageNet.py`:
   usar Caffe diretamente, calcular o gradiente com `net.backward`, aplicar
   `adversarial_data = original_data + 1.0 * sign(gradient)` no tensor
   preprocessado `CHW/BGR/0..255` e clipar em `[0, 255]`.
   Não usar TensorFlow, CleverHans ou logits de grafo TF para a reprodução ImageNet.
6. Predizer imagem adversarial.
7. Se adv_pred == clean_pred:
      disturbed_failure += 1
      pular avaliação da detecção.
8. Para cada número de intervalos k de 2 a 10:
      aplicar quantização na imagem limpa;
      aplicar quantização na imagem adversarial;
      se C(x_clean) != C(Q(x_clean)): FP += 1
      se C(x_adv) != C(Q(x_adv)): TP += 1
      senão: FN += 1
9. Calcular Recall, Precision e F1.
8. Fórmulas
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * recall * precision / (recall + precision)

Se o denominador for zero:

metric = 0.0
9. Quantização esperada

Para imagem em escala [0,255], a quantização deve usar o número de intervalos k.

A implementação precisa ser consistente com o código original. No código original, a quantização usa um interval, ou seja, um step de intensidade. Para reproduzir os intervalos da Table 4, a conversão prática pode ser:

step = 256 // k

Exemplos:

Intervalos	Step aproximado
2	128
4	64
6	43

O step 43 é importante porque aparece depois na lógica de alta entropia: 6 intervalos para imagens com entropia maior que 5.

10. Critérios de aceitação

A implementação da Table 4 é aceita se:

[ ] O modelo GoogLeNet carrega corretamente.
[ ] O script lê imagens locais PNG.
[ ] O número de clean_correct é maior que zero.
[ ] O número de attack_success é maior que zero.
[ ] disturbed_failure não domina 100% das imagens corretamente classificadas.
[ ] O CSV final tem 9 linhas, uma para cada intervalo de 2 a 10.
[ ] As métricas não são todas zero.
[ ] O intervalo 6 aparece no resultado.
[ ] O script gera diagnóstico por imagem.

Critério específico de sanidade:

Se attack_success = 0, a Table 4 deve falhar explicitamente com mensagem de diagnóstico.

Mensagem esperada:

FGSM did not generate any successful adversarial example.
Check epsilon scale, preprocessing, gradient sign, model wrapper, or labels.
11. Principais riscos
Sintoma	Causa provável
skipped_wrong_baseline muito alto	Labels errados, preprocess errado, modelo errado
disturbed_failure muito alto	FGSM com escala errada, gradiente errado, epsilon pequeno demais
tp=0 e fn=0	Nenhum adversarial válido foi avaliado
fp alto demais	Quantização agressiva ou preprocess inconsistente
Todas as métricas zero	Pipeline de ataque falhou antes da detecção
