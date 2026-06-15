# Experimentos de detecção adversarial em áudio

Este documento descreve o fluxo lógico dos experimentos adicionados na branch `poc/audio-speech-commands` para avaliar uma adaptação do método DeepDetector para o domínio de áudio.

A ideia central foi testar se a estratégia do artigo base, originalmente aplicada a imagens, poderia ser adaptada para comandos de voz representados como espectrogramas log-Mel. O experimento foi mantido como uma prova de conceito isolada, sem integrar ao runner principal do projeto, para permitir ciclos rápidos de validação.

---

## 1. Motivação

O artigo base detecta exemplos adversariais comparando a predição do classificador antes e depois de uma transformação de filtro:

```text
C(x) != C(T(x))  =>  exemplo suspeito/adversarial
C(x) == C(T(x))  =>  exemplo tratado como benigno
```

No domínio de imagens, o filtro combina:

- quantização escalar;
- suavização espacial;
- escolha adaptativa baseada em entropia.

A adaptação para áudio não é direta, porque um espectrograma não é uma imagem comum. Seus eixos possuem significado físico diferente:

```text
Eixo X: tempo
Eixo Y: frequência
Valor da célula: intensidade em dB
```

Por isso, o objetivo dos experimentos foi responder:

1. O método DeepDetector funciona quando aplicado a espectrogramas?
2. A entropia usada no artigo continua sendo uma boa heurística para áudio?
3. Quantização fixa pode ser melhor do que o filtro adaptativo original?
4. Suavizações anisotrópicas, principalmente no eixo de frequência, melhoram a detecção?

---

## 2. Arquivos adicionados ao repositório

### 2.1. Experimento principal

```text
scripts/dev/audio_adversarial_poc.py
```

Script principal da prova de conceito. Ele:

- carrega o dataset Speech Commands via `torchaudio`;
- seleciona as classes `yes`, `no`, `up`, `down`, `left`, `right`;
- converte áudio para espectrograma log-Mel em dB;
- treina/carrega uma CNN residual (`SpeechResCNN`);
- gera exemplos adversariais com FGSM no espaço de espectrogramas em dB;
- aplica um filtro DeepDetector-style;
- calcula métricas alinhadas ao artigo base: `num_failures`, `TP`, `FN`, `FP`, `RTP`, `recall`, `precision` e `F1`;
- calcula também `FPR` como métrica adicional operacional.

---

### 2.2. Calibração inicial do filtro

```text
scripts/dev/audio_filter_calibration_poc.py
```

Script criado para calibrar o filtro herdado do artigo. Ele testa combinações de:

- thresholds de entropia;
- níveis de quantização;
- tipos de smoothing;
- normalização `sample` vs `global`.

O objetivo foi descobrir se os thresholds de imagem poderiam ser reaproveitados ou se o domínio de áudio exigia nova calibração.

---

### 2.3. Sweep de epsilon

```text
scripts/dev/audio_epsilon_sweep_poc.py
```

Script criado para avaliar como o detector se comporta quando a força do ataque FGSM muda.

Ele roda o mesmo modelo e filtro para múltiplos valores de epsilon:

```text
0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0
```

A saída é um CSV/JSON consolidado com métricas por valor de epsilon.

---

### 2.4. Filtro V2: quantização fixa

```text
scripts/dev/audio_filter_v2_fixed_quantization_sweep_poc.py
```

Script criado para testar a hipótese de que a entropia talvez não fosse necessária para espectrogramas.

Ele compara:

- o filtro calibrado por entropia;
- filtros fixos com diferentes valores de quantização;
- smoothings `none`, `mean3`, `cross3`, `cross5`;
- combinação `conservative` vs `replace`.

---

### 2.5. Filtro V3: geometria do espectrograma

```text
scripts/dev/audio_filter_v3_geometry_sweep_poc.py
```

Script criado para testar filtros mais compatíveis com a geometria tempo-frequência do espectrograma.

Ele testa:

- suavização temporal: `time3`, `time5`, `time7`;
- suavização frequencial: `freq3`, `freq5`, `freq7`;
- suavização simétrica: `cross5`, `mean3`;
- combinação `replace` e `weighted`;
- noise gates em dB.

---

### 2.6. Filtro V4: frequência + entropia calibrada

```text
scripts/dev/audio_filter_v4_freq_entropy_sweep_poc.py
```

Script criado para aprofundar o melhor resultado anisotrópico encontrado no V3.

Ele parte da hipótese de que suavizar apenas no eixo de frequência pode ser mais adequado do que aplicar filtros simétricos de imagem.

O V4 testa:

- quantizações fixas `q=5,6,7,8`;
- kernels frequenciais `freq5`, `freq7`, `freq9`, `freq11`;
- `cross5`, `cross7`, `time3`, `time5` como comparação;
- `replace` e `weighted`;
- noise gate;
- entropia recalibrada para a faixa real dos espectrogramas.

---

## 3. Métricas utilizadas

As métricas principais seguem o estilo do artigo base:

| Métrica | Interpretação |
|---|---|
| `num_failures` | ataques que falharam, isto é, o adversarial não enganou o classificador |
| `TP` | adversariais bem-sucedidos detectados pelo filtro |
| `FN` | adversariais bem-sucedidos não detectados |
| `FP` | exemplos limpos marcados incorretamente como adversariais |
| `RTP` | adversariais detectados que foram restaurados para a classe correta após filtragem |
| `recall` | proporção de adversariais detectados |
| `precision` | proporção de alertas que realmente eram adversariais |
| `F1` | média harmônica entre recall e precision |

Além disso, foi adicionada uma métrica operacional:

| Métrica | Interpretação |
|---|---|
| `FPR` | taxa de falso positivo, calculada como `FP / (FP + TN)` |

O `FPR` não é uma métrica central do artigo base, mas foi incluído para avaliar o impacto prático do detector como IDS. Um filtro com recall alto, mas FPR muito alto, não é útil operacionalmente.

---

## 4. Fluxo lógico de experimentação

## 4.1. Experimento inicial: adaptação direta para áudio

### Objetivo

Validar se o pipeline básico funcionava:

```text
áudio -> espectrograma log-Mel -> classificador -> FGSM -> filtro -> métricas
```

### Resultado inicial

Com a primeira CNN simples, o classificador limpo ainda era fraco:

| Métrica | Valor |
|---|---:|
| Clean accuracy | 65.17% |
| Recall | 17.81% |
| Precision | 67.06% |
| F1 | 28.15% |
| FPR | 4.67% |
| RTP% | 77.19% |

### Interpretação

O resultado mostrava que o fluxo funcionava, mas a avaliação do detector ainda não era confiável, porque o classificador base tinha baixa acurácia. Antes de melhorar o filtro, era necessário melhorar o modelo.

---

## 4.2. Melhoria do classificador: SpeechResCNN

### O que mudou

A CNN simples foi substituída por uma arquitetura residual compacta (`SpeechResCNN`). Também foram adicionados:

- `AdamW`;
- `weight_decay`;
- `label_smoothing`;
- scheduler cosseno;
- normalização interna do espectrograma em dB.

### Resultado

Com `12k` amostras de treino, `1200` de teste e `epsilon=2.0`, o classificador passou a ter:

| Métrica | Valor |
|---|---:|
| Clean accuracy | 98.50% |
| Clean correctly classified | 1182 / 1200 |
| Clean misclassified | 18 / 1200 |

Com o filtro ainda não otimizado, o resultado de detecção foi:

| Métrica | Valor |
|---|---:|
| Attack success rate | 45.69% |
| TP | 125 |
| FN | 415 |
| FP | 17 |
| TN | 1183 |
| RTP | 104 |
| Recall | 23.15% |
| Precision | 88.03% |
| F1 | 36.66% |
| FPR | 1.42% |
| RTP% | 83.20% |

### Interpretação

A melhoria do classificador deixou a avaliação do detector mais confiável. O detector era preciso e tinha poucos falsos positivos, mas ainda detectava poucos adversariais.

---

## 4.3. Calibração inicial do filtro por entropia

### Hipótese

Os thresholds do artigo base foram definidos para imagens. Como espectrogramas possuem distribuição diferente de intensidade, testamos novos thresholds, quantizações e normalizações.

### Melhor configuração balanceada encontrada

```yaml
filter:
  normalization: global
  entropy_thresholds:
    low: 3.0
    high: 4.0
  quantization:
    low_entropy: 2
    mid_entropy: 4
    high_entropy: 6
  smoothing: mean3
```

### Resultado

| Métrica | Valor |
|---|---:|
| TP | 180 |
| FN | 360 |
| FP | 22 |
| TN | 1178 |
| RTP | 152 |
| Recall | 33.33% |
| Precision | 89.11% |
| F1 | 48.52% |
| FPR | 1.83% |
| RTP% | 84.44% |

### Ganho sobre o filtro anterior

| Métrica | Antes | Depois |
|---|---:|---:|
| Recall | 23.15% | 33.33% |
| Precision | 88.03% | 89.11% |
| F1 | 36.66% | 48.52% |
| FPR | 1.42% | 1.83% |
| RTP% | 83.20% | 84.44% |

### Interpretação

A normalização global em dB foi melhor do que normalização por amostra. A calibração aumentou recall e F1, mas o detector ainda era conservador.

---

## 4.4. Sweep de epsilon com o filtro calibrado

### Objetivo

Avaliar o comportamento do detector conforme a força do ataque FGSM aumenta.

### Resultados

| Epsilon | Attack success | Recall | Precision | F1 | FPR | RTP% |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.18% | 58.90% | 66.15% | 62.32% | 1.83% | 95.35% |
| 0.50 | 15.40% | 49.45% | 80.36% | 61.22% | 1.83% | 94.44% |
| 1.00 | 30.12% | 41.85% | 87.13% | 56.55% | 1.83% | 89.26% |
| 1.50 | 39.00% | 37.74% | 88.78% | 52.97% | 1.83% | 85.63% |
| 2.00 | 45.69% | 33.33% | 89.11% | 48.52% | 1.83% | 84.44% |
| 2.50 | 49.92% | 28.81% | 88.54% | 43.48% | 1.83% | 84.71% |
| 3.00 | 53.21% | 26.23% | 88.24% | 40.44% | 1.83% | 77.58% |
| 4.00 | 57.87% | 25.00% | 88.60% | 39.00% | 1.83% | 71.35% |

### Interpretação

O filtro calibrado funcionava melhor para ataques fracos e moderados. Para ataques mais fortes, a taxa de sucesso do ataque aumentava, mas o recall do detector caía. O FPR permanecia constante porque ele é calculado sobre os exemplos limpos, que não mudam com epsilon.

---

## 4.5. Filtro V2: quantização fixa sem entropia

### Hipótese

A entropia talvez não fosse uma boa regra de decisão para espectrogramas. Assim, testamos filtros fixos com normalização global e diferentes níveis de quantização.

### Melhor configuração V2

```yaml
filter:
  type: fixed_quantization
  normalization: global
  quantization: 5
  smoothing: cross5
  combine: replace
  noise_gate: none
```

### Resultado

| Métrica | Valor |
|---|---:|
| TP | 307 |
| FN | 233 |
| FP | 34 |
| TN | 1166 |
| Recall | 56.85% |
| Precision | 90.03% |
| F1 | 69.69% |
| FPR | 2.83% |
| RTP% | 89.25% |

### Comparação com o filtro calibrado por entropia

| Métrica | Entropia calibrada | V2 fixo |
|---|---:|---:|
| Recall | 33.33% | 56.85% |
| Precision | 89.11% | 90.03% |
| F1 | 48.52% | 69.69% |
| FPR | 1.83% | 2.83% |
| RTP% | 84.44% | 89.25% |

### Interpretação

O filtro fixo superou claramente a entropia calibrada inicial. Isso indicou que copiar diretamente a heurística adaptativa do artigo não era a melhor solução para áudio.

O modo `replace` foi melhor que o modo conservador. Isso sugere que escolher sempre a transformação menos distante do original pode preservar parte do padrão adversarial.

---

## 4.6. Filtro V3: geometria tempo-frequência

### Hipótese

Como espectrogramas têm eixos diferentes, suavizações anisotrópicas poderiam ser melhores do que filtros simétricos de imagem.

### Melhor filtro geral no V3

O melhor resultado continuou sendo o V2:

```yaml
filter:
  type: fixed_quantization
  normalization: global
  quantization: 5
  smoothing: cross5
  combine: replace
  noise_gate: none
```

Resultado:

| Métrica | Valor |
|---|---:|
| Recall | 56.85% |
| Precision | 90.03% |
| F1 | 69.69% |
| FPR | 2.83% |
| RTP% | 89.25% |

### Melhor filtro anisotrópico em frequência

```yaml
filter:
  type: fixed_quantization
  normalization: global
  quantization: 6
  smoothing: freq7
  combine: replace
  noise_gate: none
```

Resultado:

| Métrica | Valor |
|---|---:|
| TP | 304 |
| FN | 236 |
| FP | 30 |
| TN | 1170 |
| Recall | 56.30% |
| Precision | 91.02% |
| F1 | 69.57% |
| FPR | 2.50% |
| RTP% | 90.13% |

### Interpretação

O filtro `freq7` quase empatou com o melhor filtro geral, com menor FPR e maior precisão. Isso sugeriu que suavizar no eixo de frequência é uma direção promissora.

O noise gate não trouxe ganho. Thresholds `none`, `-75` e `-70 dB` geraram resultados iguais ou praticamente iguais, enquanto thresholds mais agressivos começaram a piorar as métricas.

---

## 4.7. Filtro V4: aprofundamento em frequência e entropia recalibrada

### Hipótese

Como o filtro anisotrópico em frequência foi competitivo, testamos:

- kernels frequenciais maiores (`freq9`, `freq11`);
- quantizações maiores (`q=7`, `q=8`);
- entropia recalibrada para a faixa real de entropia dos espectrogramas;
- combinações `replace` e `weighted`;
- noise gate.

### Melhor F1 absoluto

```yaml
filter_profile: v4_fixed_high_f1
normalization: global
quantization: 7
smoothing: freq11
combine: replace
noise_gate: none
```

Resultado:

| Métrica | Valor |
|---|---:|
| TP | 353 |
| FN | 187 |
| FP | 43 |
| TN | 1157 |
| Recall | 65.37% |
| Precision | 89.14% |
| F1 | 75.43% |
| FPR | 3.58% |
| RTP% | 89.24% |

### Melhor filtro balanceado com entropia recalibrada

```yaml
filter_profile: v4_entropy_balanced
normalization: global
entropy_thresholds:
  low: 6.1
  high: 6.5
quantization:
  low_entropy: 6
  mid_entropy: 7
  high_entropy: 8
smoothing:
  low_entropy: freq7
  mid_entropy: freq9
  high_entropy: freq11
combine: replace
noise_gate: none
```

Resultado:

| Métrica | Valor |
|---|---:|
| TP | 342 |
| FN | 198 |
| FP | 33 |
| TN | 1167 |
| Recall | 63.33% |
| Precision | 91.20% |
| F1 | 74.75% |
| FPR | 2.75% |
| RTP% | 88.89% |

### Comparação entre V4 fixo e V4 com entropia

| Métrica | V4 fixo `q7 + freq11` | V4 entropia balanceado |
|---|---:|---:|
| Recall | 65.37% | 63.33% |
| Precision | 89.14% | 91.20% |
| F1 | 75.43% | 74.75% |
| FPR | 3.58% | 2.75% |
| RTP% | 89.24% | 88.89% |
| TP | 353 | 342 |
| FP | 43 | 33 |

### Interpretação

O melhor F1 absoluto veio do filtro fixo `q7 + freq11 + replace`. Porém, o filtro com entropia recalibrada teve desempenho quase igual em F1, com maior precisão e menor falso positivo.

Isso mudou a conclusão sobre entropia:

```text
A entropia herdada diretamente do artigo base não transferiu bem para áudio.
Mas a entropia recalibrada para a distribuição dos espectrogramas voltou a ser útil.
```

---

## 5. Evolução dos principais resultados

| Etapa | Filtro | Recall | Precision | F1 | FPR | RTP% |
|---|---|---:|---:|---:|---:|---:|
| Inicial | CNN simples + filtro inicial | 17.81% | 67.06% | 28.15% | 4.67% | 77.19% |
| SpeechResCNN | filtro não otimizado | 23.15% | 88.03% | 36.66% | 1.42% | 83.20% |
| Calibração | entropia `3/4`, q `2/4/6`, `mean3` | 33.33% | 89.11% | 48.52% | 1.83% | 84.44% |
| V2 | fixo `q5 + cross5 + replace` | 56.85% | 90.03% | 69.69% | 2.83% | 89.25% |
| V3 | fixo `q6 + freq7 + replace` | 56.30% | 91.02% | 69.57% | 2.50% | 90.13% |
| V4 | fixo `q7 + freq11 + replace` | 65.37% | 89.14% | 75.43% | 3.58% | 89.24% |
| V4 | entropia `6.1/6.5`, q `6/7/8`, freq `7/9/11` | 63.33% | 91.20% | 74.75% | 2.75% | 88.89% |

---

## 6. Conclusões técnicas

### 6.1. O classificador deixou de ser o gargalo

Após a troca para `SpeechResCNN`, a acurácia limpa subiu para `98.50%`. Isso permitiu avaliar os filtros com mais confiança.

### 6.2. Normalização global é mais adequada para áudio

A normalização por faixa fixa `[-80, 0] dB -> [0, 255]` foi mais adequada do que normalização por amostra, pois preserva a escala absoluta de energia do espectrograma.

### 6.3. Entropia do artigo não transfere diretamente

Os thresholds de entropia do artigo base foram pensados para imagens. Em espectrogramas, os valores de entropia são mais altos e têm outra distribuição.

A versão inicial com entropia ajudou, mas foi superada por filtros fixos.

### 6.4. Entropia recalibrada pode ser útil

O V4 mostrou que entropia ainda pode funcionar, desde que recalibrada para áudio. O filtro com thresholds `6.1/6.5` teve F1 próximo do melhor resultado geral e reduziu falsos positivos.

### 6.5. Suavização em frequência é promissora

Os melhores resultados do V4 vieram de filtros `freq11` e de combinações adaptativas com `freq7/freq9/freq11`. Isso indica que suavizar no eixo de frequência pode ser mais adequado do que usar apenas filtros simétricos herdados de imagem.

### 6.6. Noise gate não trouxe benefício relevante

Nos experimentos, `noise_gate=none`, `-75 dB` e `-70 dB` produziram resultados iguais ou muito próximos. Thresholds mais agressivos começaram a degradar o desempenho.

### 6.7. `replace` superou combinações conservadoras

O modo `replace`, que aplica diretamente a versão suavizada, superou a lógica conservadora. Isso sugere que preservar demais o espectrograma original pode manter traços da perturbação adversarial.

---

## 7. Perfis finais recomendados

### 7.1. Perfil com maior F1

Use quando o objetivo for maximizar detecção e F1.

```yaml
filter_profile: v4_fixed_high_f1
normalization: global
quantization: 7
smoothing: freq11
combine: replace
noise_gate: none
```

Resultado:

```text
Recall:    65.37%
Precision: 89.14%
F1:        75.43%
FPR:        3.58%
RTP%:      89.24%
```

### 7.2. Perfil balanceado para IDS

Use quando o objetivo for manter FPR mais baixo e precisão mais alta.

```yaml
filter_profile: v4_entropy_balanced
normalization: global
entropy_thresholds:
  low: 6.1
  high: 6.5
quantization:
  low_entropy: 6
  mid_entropy: 7
  high_entropy: 8
smoothing:
  low_entropy: freq7
  mid_entropy: freq9
  high_entropy: freq11
combine: replace
noise_gate: none
```

Resultado:

```text
Recall:    63.33%
Precision: 91.20%
F1:        74.75%
FPR:        2.75%
RTP%:      88.89%
```

---

## 8. Próximos passos recomendados

Como muitos filtros foram calibrados sobre o mesmo subconjunto de `1200` exemplos, os resultados acima devem ser tratados como resultados de calibração.

Para consolidar o resultado final, recomenda-se:

1. congelar os dois perfis finais:
   - `v4_fixed_high_f1`;
   - `v4_entropy_balanced`;
2. rodar avaliação no conjunto de teste completo com `--max-test 0`;
3. rodar novo sweep de epsilon usando os perfis finais;
4. comparar os filtros finais com:
   - filtro inicial;
   - filtro calibrado por entropia `3/4`;
   - V2 `q5 + cross5 + replace`;
5. atualizar `audio_adversarial_poc.py` para aceitar perfis nomeados de filtro.

---

## 9. Comandos úteis

### 9.1. Rodar experimento principal

```bash
python scripts/dev/audio_adversarial_poc.py \
  --device cuda \
  --data-dir data/raw/speech_commands \
  --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
  --max-test 1200 \
  --batch-size 64 \
  --epsilon 2.0 \
  --metrics-out artifacts/audio_poc/metrics_audio.json
```

### 9.2. Rodar sweep de epsilon

```bash
python scripts/dev/audio_epsilon_sweep_poc.py \
  --device cuda \
  --data-dir data/raw/speech_commands \
  --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
  --max-test 1200 \
  --batch-size 64 \
  --epsilons "0.25,0.5,1.0,1.5,2.0,2.5,3.0,4.0" \
  --csv-out artifacts/audio_poc/epsilon_sweep.csv \
  --json-out artifacts/audio_poc/epsilon_sweep.json
```

### 9.3. Rodar V2

```bash
python scripts/dev/audio_filter_v2_fixed_quantization_sweep_poc.py \
  --device cuda \
  --data-dir data/raw/speech_commands \
  --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
  --max-test 1200 \
  --batch-size 64 \
  --epsilon 2.0 \
  --quantizations "2,3,4,5,6,8,12,16" \
  --smoothings "none,mean3,cross3,cross5" \
  --normalizations "global" \
  --smoothing-combines "conservative,replace" \
  --csv-out artifacts/audio_poc/filter_v2_fixed_quantization_sweep.csv \
  --json-out artifacts/audio_poc/filter_v2_fixed_quantization_sweep.json
```

### 9.4. Rodar V3

```bash
python scripts/dev/audio_filter_v3_geometry_sweep_poc.py \
  --device cuda \
  --data-dir data/raw/speech_commands \
  --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
  --max-test 1200 \
  --batch-size 64 \
  --epsilon 2.0 \
  --quantizations "4,5,6" \
  --smoothings "none,time3,time5,time7,freq3,freq5,freq7,cross5,mean3" \
  --combine-modes "replace,weighted" \
  --weighted-alphas "0.25,0.5,0.75" \
  --noise-gates "none,-75,-70,-65,-60" \
  --csv-out artifacts/audio_poc/filter_v3_geometry_sweep.csv \
  --json-out artifacts/audio_poc/filter_v3_geometry_sweep.json
```

### 9.5. Rodar V4

```bash
python scripts/dev/audio_filter_v4_freq_entropy_sweep_poc.py \
  --device cuda \
  --data-dir data/raw/speech_commands \
  --checkpoint artifacts/audio_poc/speech_rescnn_6cls_12k_e20.pt \
  --max-test 1200 \
  --batch-size 64 \
  --epsilon 2.0 \
  --csv-out artifacts/audio_poc/filter_v4_freq_entropy_sweep.csv \
  --json-out artifacts/audio_poc/filter_v4_freq_entropy_sweep.json
```

---

## 10. Resumo executivo

A extensão para áudio começou com uma adaptação direta do DeepDetector para espectrogramas, mas o desempenho inicial era limitado. Após melhorar o classificador com uma `SpeechResCNN`, o gargalo passou a ser o filtro.

A calibração inicial por entropia melhorou o resultado, mas foi superada por filtros fixos com quantização e smoothing mais agressivo. Em seguida, os experimentos V3 e V4 mostraram que suavizações no eixo de frequência são especialmente promissoras para espectrogramas.

O melhor resultado geral até agora é o filtro fixo `q7 + freq11 + replace`, com `F1=75.43%` e `Recall=65.37%`. Para um cenário mais balanceado de IDS, o filtro com entropia recalibrada `6.1/6.5 + q6/7/8 + freq7/freq9/freq11` é uma alternativa forte, com `F1=74.75%`, `Precision=91.20%` e `FPR=2.75%`.

A principal conclusão técnica é que a heurística do artigo base não deve ser transferida diretamente para áudio. O domínio de espectrogramas exige normalização global em dB, filtros que respeitem a estrutura tempo-frequência e thresholds de entropia recalibrados.
