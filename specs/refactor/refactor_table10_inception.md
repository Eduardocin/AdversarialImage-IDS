# Refactor - Table_10_Inception_V3
Você é um desenvolvedor Python sênior revisando a reprodução da Table 10 do artigo **“Detecting Adversarial Image Examples in Deep Neural Networks with Adaptive Noise Reduction”**.

## Contexto:
Estou reproduzindo a Table 10, especialmente as linhas de **CW L2 / Inception v3 / ImageNet**, na branch:

https://github.com/Eduardocin/AdversarialImage-IDS/tree/fix/inceptionv3-ampere

O artigo base usa o código OwenSec/DeepDetector. Para CW L2 com Inception v3, os arquivos de referência são:

- `Test/CW/Test_CWL2_ImageNet.py`
- `Test/CW/modified_setup_inception.py`

No código base, o dataset ImageNet/Inception é carregado em `[-0.5, 0.5]`:

```python
img = np.array(
    scipy.misc.imresize(scipy.misc.imread(ff), (299,299)),
    dtype=np.float32
) / 255 - .5
```
E o modelo converte de volta para [0,255] apenas na chamada de predição:

```python
dat = (0.5 + np.reshape(image, ((299,299,3)))) * 255
```

## Objetivo da revisão:
Investigar por que minha reprodução da linha 15 está divergindo do artigo.

Resultado esperado no artigo para a linha 15:
```text
CW L2 (κ=0.5)/Inception v3 | ImageNet
#F=0, TP=100, FN=0, FP=2, RTP=98
Recall=100%, Precision=98.04%, F1=99.01%
```

Resultado observado na minha implementação:
```text
clean_errors=7
attack_failures=0
tp=93
fn=0
fp=93
```
Quero que você revise com foco em dois pontos principais.

---

## 1. Revisar o cálculo das métricas da Table 10

Verifique se a implementação calcula corretamente:

### #F — attack failures

#F deve contar quantas imagens não conseguiram ser perturbadas para adversariais efetivas.

Para CW L2 / Inception:
```python
if adv_pred == original_correct_label:
    attack_failures += 1
    continue
```

Ou seja, se o ataque não mudou a classe prevista para uma classe incorreta, ele não entra em TP/FN.

### TP — True Positive

TP deve ser calculado somente nas imagens adversariais efetivas.

A lógica correta é:
```python
if adv_pred != adv_filtered_pred:
    TP += 1
```
Significado: a imagem adversarial foi detectada porque a predição mudou depois do filtro.

### FN — False Negative

FN também deve ser calculado somente nas imagens adversariais efetivas.

```python
if adv_pred == adv_filtered_pred:
    FN += 1
```

Significado: a adversarial passou pelo detector sem ser marcada.

### FP — False Positive

FP deve ser calculado somente nas imagens limpas/benignas, não nas adversariais.

A lógica correta é:

```python
if clean_pred != clean_filtered_pred:
    FP += 1
```

Significado: uma imagem limpa foi marcada incorretamente como adversarial porque o filtro mudou sua predição.

Atenção especial:
Se o código estiver incrementando FP usando adv_pred ou adv_filtered_pred, está errado.

Um sintoma forte de bug é:
```text
tp=93
fp=93
```

Isso pode indicar que o FP está sendo contado junto com o TP, ou que o filtro está destruindo a escala das imagens limpas e mudando todas as predições filtradas.

### RTP — Recovered True Positive

RTP deve contar adversariais que foram detectadas e cuja versão filtrada voltou para a classe original correta.

```python
if adv_pred != adv_filtered_pred:
    TP += 1
    if adv_filtered_pred == original_correct_label:
        RTP += 1
else:
    FN += 1
```

Não confundir RTP com TP.

TP mede detecção.

RTP mede recuperação da classe original após o filtro.

### Fórmulas esperadas

As fórmulas devem ser:
```python
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * recall * precision / (recall + precision)
rtp_percent = RTP / TP
```
Para a linha 15 do artigo:
```python
recall = 100 / (100 + 0) = 1.0
precision = 100 / (100 + 2) = 0.98039
f1 ≈ 0.9901
rtp_percent = 98 / 100 = 0.98
```

---

## 2. Revisar a seleção das 100 imagens

O artigo usa 100 imagens ImageNet para CW L2 / Inception v3.

Na reprodução, preciso garantir que essas 100 imagens sejam 100 imagens válidas, isto é:

1. pertencem às classes usadas no artigo:
	* Zebra
	* Panda
	* Cab
2. respeitam as quotas da Table 10:
	* 40 zebras
	* 40 pandas
	* 20 cabs
3. são corretamente classificadas pelo Inception v3 antes do ataque.

No código base, a seleção é:
```python
for i in range(40):
    inputs.append(data.zebraData[i])
    targets.append(data.zebraLabel[i])
    sources.append(data.zebraLabel[i].argmax())

for i in range(40):
    inputs.append(data.pandaData[i])
    targets.append(data.pandaLabel[i])
    sources.append(data.pandaLabel[i].argmax())

for i in range(20):
    inputs.append(data.cabData[i])
    targets.append(data.cabLabel[i])
    sources.append(data.cabLabel[i].argmax())
```
Mas na reprodução moderna, se algumas dessas imagens gerarem clean_errors, a avaliação acaba ficando com menos de 100 imagens válidas.

O log atual mostra:
```text
clean_errors=7
tp=93
```
Isso sugere que o código pegou 100 imagens brutas, mas 7 não foram classificadas corretamente pelo modelo antes do ataque. Para reproduzir a Table 10, o ideal é selecionar até obter 100 imagens limpas corretamente classificadas pelo Inception.

A lógica desejada é:
```python
valid_samples = []

for candidate in dataset:
    clean_pred = model.predict(candidate.image)

    if clean_pred != candidate.true_label:
        clean_errors += 1
        continue

    valid_samples.append(candidate)

    if len(valid_samples) == 100:
        break
```
Depois disso, o CW L2 deve ser executado sobre essas 100 imagens válidas.

Critério importante:
clean_errors pode ser logado para diagnóstico, mas não deve reduzir a tabela final para 93 amostras se ainda existem outras imagens disponíveis para completar as 100 válidas.

---

## 3. Revisar compatibilidade do filtro com o domínio do Inception

Além das métricas, revise se o filtro final está compatível com imagens em [-0.5, 0.5].

No código base, o filtro recebe imagens nesse domínio.

Exemplo da quantização oficial:
```python
def scalarQuantization(inputDigit, interval):
    retDigit = (inputDigit + 0.5) * 255
    retDigit //= interval
    retDigit *= interval
    retDigit /= 255.0
    retDigit -= 0.5
    return retDigit
```
A entropia também converte internamente:
```python
expandDigit = np.array((inputDigit + 0.5) * 255, dtype=np.int16)
```
Portanto, se a implementação atual tem algo como:
```python
def _is_normalized(image):
    return image.min() >= 0.0 and image.max() <= 1.0
```
isso pode falhar para Inception, porque imagens em [-0.5, 0.5] têm pixels negativos.

Verifique se o filtro não está fazendo clip(0, 255) diretamente em imagens [-0.5, 0.5], pois isso transforma pixels negativos em zero e pode causar muitos falsos positivos.

Teste esperado:
```python
image = np.random.uniform(-0.5, 0.5, size=(299, 299, 3)).astype(np.float32)
filtered = article_filter(image)

assert filtered.min() >= -0.5 - 1e-6
assert filtered.max() <= 0.5 + 1e-6
```

---

## Entregáveis esperados da revisão
- Identificar onde TP, FN, FP, RTP e #F são calculados.
- Confirmar se FP é calculado usando imagens limpas ou adversariais.
- Confirmar se a avaliação usa exatamente 100 imagens válidas para CW L2 / Inception v3.
- Confirmar se clean_errors apenas registra imagens descartadas ou se reduz indevidamente o total avaliado.
- Confirmar se o filtro preserva o domínio [-0.5, 0.5] para Inception.
- Sugerir patch mínimo e testes unitários para:
	* cálculo correto de FP;
	* seleção de 100 imagens limpas corretamente classificadas;
	* preservação do range [-0.5, 0.5] no filtro;
	* reprodução das fórmulas de Recall, Precision, F1 e RTP%.

Prioridade máxima:
Corrigir qualquer ponto que explique o padrão anômalo:
```text
tp=93
fp=93
clean_errors=7
```
Esse padrão não deve ocorrer se FP estiver sendo calculado corretamente e se o filtro estiver preservando o domínio de entrada do Inception.