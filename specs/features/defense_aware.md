# SPEC — Defense-Aware Adaptive CW-L2 Attack

## Objective

Implementar o experimento de **defense-aware attack** do artigo *Detecting Adversarial Image Examples in Deep Networks with Adaptive Noise Reduction*, avaliando a robustez do DeepDetector contra um ataque Carlini & Wagner L2 adaptativo.

A implementação deve comparar dois cenários:

1. **Defense-unaware attack**: o atacante gera exemplos adversariais contra o classificador original, sem considerar o detector.
2. **Defense-aware attack**: o atacante conhece o detector e tenta gerar exemplos adversariais que enganem o classificador e também não sejam detectados após a transformação adaptativa.

A implementação deve seguir a arquitetura atual do projeto, utilizando exclusivamente o runner centralizado:

```bash
python scripts/run_experiment.py --experiment defense_aware
````

---

## Scope

Este experimento deve:

* utilizar MNIST como dataset;
* utilizar o modelo MNIST M2;
* utilizar Carlini & Wagner L2 como ataque base;
* implementar uma versão adaptativa do CW-L2;
* comparar ataque defense-unaware e defense-aware;
* aplicar a transformação final adaptativa do DeepDetector durante a avaliação;
* produzir métricas agregadas de sucesso do ataque, taxa de detecção, taxa de evasão e distância L2;
* gerar somente os artefatos oficiais definidos pelo projeto.

Este experimento não deve:

* executar FGSM;
* executar DeepFool;
* executar ImageNet;
* executar Table 6;
* executar Table 7;
* executar Table 8;
* executar Table 9;
* alterar a lógica oficial da transformação adaptativa;
* gerar diagnósticos públicos;
* gerar relatórios em Markdown;
* criar experimentos auxiliares públicos.

---

## Background

No artigo, os autores avaliam dois cenários de ameaça.

No cenário **defense-unaware**, o atacante conhece apenas o classificador `C`. Assim, o exemplo adversarial é gerado para satisfazer:

```text
C(x_adv) != C(x)
```

Depois da geração, o detector verifica se a transformação adaptativa `T` muda a predição:

```text
C(x_adv) != C(T(x_adv))
```

Se a predição mudar, o exemplo é detectado.

No cenário **defense-aware**, o atacante conhece tanto o classificador `C` quanto a transformação adaptativa `T`. Assim, o ataque deve tentar gerar um exemplo adversarial que satisfaça simultaneamente:

```text
C(x_adv) != C(x)
C(x_adv) == C(T(x_adv))
```

Ou seja, o adversarial deve enganar o classificador e permanecer consistente após a transformação do DeepDetector.

No repositório original, essa lógica aparece principalmente em:

```text
Test/CW/adaptive_CWL2_MNIST.py
Test/CW/l2_adaptive_attack.py
```

A implementação deste projeto deve reproduzir esse comportamento de forma compatível com a arquitetura atual.

---

## Public Execution Interface

A única interface pública permitida é:

```bash
python scripts/run_experiment.py --experiment defense_aware
```

Não devem existir comandos adicionais como:

```bash
python scripts/run_experiment.py --experiment adaptive_cw
python scripts/run_experiment.py --experiment cw_l2_adaptive
python scripts/run_experiment.py --experiment blind_cw
python scripts/run_experiment.py --experiment defense_unaware
```

O ataque defense-unaware e o ataque defense-aware devem ser componentes internos do mesmo experimento público.

---

## Configuration

A configuração deve residir em:

```text
configs/experiments.yaml
```

Estrutura sugerida:

```yaml
defense_aware:
  description: Defense-aware CW-L2 adaptive attack against DeepDetector

  seed: 42

  dataset:
    name: mnist
    split: test
    start: 9000
    end: 10000
    value_range:
      min: 0.0
      max: 1.0

  model:
    name: mnist_m2

  attacks:
    defense_unaware:
      type: cw_l2_nn_robust
      nn_robust_attacks_root: nn_robust_attacks
      targeted: false
      confidence: 0
      max_iterations: 2000
      binary_search_steps: 5
      initial_const: 1.0
      learning_rate: 0.1

    defense_aware:
      type: adaptive_cw_l2
      nn_robust_attacks_root: nn_robust_attacks
      targeted: false
      confidence: 0
      max_iterations: 2000
      binary_search_steps: 5
      initial_const: 1.0
      learning_rate: 0.1

  detector:
    type: final_adaptive_detection_filter
    entropy_thresholds:
      low: 4.0
      medium: 5.0
    quantization:
      low_entropy_step: 128
      medium_entropy_step: 64
      high_entropy_step: 43
    spatial_filter:
      type: cross_mean
      radius: 3
```

Este experimento usa `attacks` no plural de forma intencional, pois ele compara dois ataques internos no mesmo experimento.

Não devem existir configurações independentes:

```yaml
defense_unaware:
adaptive_cw:
cw_l2_adaptive:
```

---

# Dataset

## MNIST

Utilizar o intervalo:

| Split | Intervalo   |
| ----- | ----------- |
| Test  | `9000-9999` |

A configuração deve usar `start` e `end`:

```yaml
start: 9000
end: 10000
```

Não utilizar `samples` como campo principal da configuração.

Amostras originalmente classificadas de forma incorreta devem ser ignoradas.

---

# Value Range

O experimento deve respeitar o intervalo de entrada esperado pelo modelo MNIST M2 atual do projeto.

Para este experimento:

```text
value_range = [0.0, 1.0]
```

Portanto, qualquer operação de clamp deve usar:

```python
x[x < 0.0] = 0.0
x[x > 1.0] = 1.0
```

Não utilizar clamp em `[-0.5, 0.5]` neste experimento, exceto se houver uma etapa explícita de conversão de escala antes da inferência.

---

# Attack Definitions

## Defense-Unaware CW-L2

O ataque defense-unaware deve utilizar o CW-L2 padrão.

O ataque conhece apenas o classificador `C`.

Critério de sucesso:

```python
C(x_adv) != y_original
```

Esse ataque não deve receber o detector como entrada durante a otimização.

---

## Defense-Aware Adaptive CW-L2

O ataque defense-aware deve utilizar uma versão adaptativa do CW-L2.

O ataque conhece:

* o classificador `C`;
* a transformação adaptativa `T`;
* a regra de detecção baseada na comparação entre `C(x_adv)` e `C(T(x_adv))`.

Critério de sucesso:

```python
C(x_adv) != y_original
C(x_adv) == C(T(x_adv))
```

A implementação deve rejeitar candidatos adversariais que enganem o classificador, mas sejam detectados pela transformação adaptativa.

---

# DeepDetector Transformation

A transformação `T` deve corresponder ao **filtro final adaptativo de detecção** usado pelo projeto.

Não descrever esta transformação genericamente como “lógica das Tables 7, 8 e 9”, pois essas tabelas podem usar variações específicas de smoothing, dataset ou etapa experimental.

A intenção deste experimento é aplicar a transformação final composta por:

1. cálculo de entropia;
2. quantização adaptativa;
3. filtro espacial apenas para imagens de alta entropia;
4. escolha entre imagem quantizada e imagem filtrada quando aplicável.

---

## Step 1 — Clamp

Garantir que os valores estejam no intervalo esperado pelo MNIST M2:

```python
x = clamp(x, min_value=0.0, max_value=1.0)
```

---

## Step 2 — Entropy Calculation

Calcular a entropia da imagem.

Para MNIST:

```python
entropy = entropy_1d(image)
```

---

## Step 3 — Adaptive Quantization Selection

Selecionar a transformação conforme a entropia.

| Condição        | Transformação                                         |
| --------------- | ----------------------------------------------------- |
| `H < 4.0`       | Scalar Quantization com step `128`                    |
| `4.0 ≤ H < 5.0` | Scalar Quantization com step `64`                     |
| `H ≥ 5.0`       | Scalar Quantization com step `43` + Cross Mean Filter |

---

## Step 4 — Low Entropy Transformation

Quando `H < 4.0`:

```python
input_final = scalar_quantization(input, step=128)
```

---

## Step 5 — Medium Entropy Transformation

Quando `4.0 ≤ H < 5.0`:

```python
input_final = scalar_quantization(input, step=64)
```

---

## Step 6 — High Entropy Transformation

Quando `H ≥ 5.0`, aplicar:

```python
input_after_q = scalar_quantization(input, step=43)
input_after_q_and_f = cross_mean_filter(input_after_q, radius=3)
input_final = choose_closer_filter(input, input_after_q, input_after_q_and_f)
```

A chamada do filtro espacial deve seguir a API atual do projeto:

```python
cross_mean_filter(image, radius=3)
```

Não utilizar a assinatura antiga do código original:

```python
cross_mean_filter(image, start=3, end=25, coefficient=13)
```

A equivalência com o código original deve ser tratada internamente pela implementação do filtro atual, não pela spec do experimento.

---

# Processing Flow

## Step 1 — Load Sample

Carregar cada amostra MNIST no intervalo configurado:

```python
for index in range(start, end):
    ...
```

---

## Step 2 — Clean Prediction

Executar inferência na imagem original.

Se a classificação original estiver incorreta:

```python
original_classified_wrong += 1
```

A amostra não participa da avaliação.

Caso contrário:

```python
total_valid += 1
```

---

## Step 3 — Generate Defense-Unaware Adversarial Example

Gerar `x_adv_blind` utilizando CW-L2 padrão.

Executar inferência:

```python
blind_pred = C(x_adv_blind)
```

Se o ataque não alterar a classificação:

```python
blind_failure += 1
```

A amostra não deve entrar no cálculo de distância L2 do ataque defense-unaware.

---

## Step 4 — Evaluate Detection on Defense-Unaware Example

Aplicar a transformação adaptativa:

```python
x_adv_blind_transformed = T(x_adv_blind)
```

Executar inferência:

```python
blind_transformed_pred = C(x_adv_blind_transformed)
```

Se a predição mudar após a transformação:

```python
blind_detected += 1
```

Caso contrário:

```python
blind_undetected += 1
```

---

## Step 5 — Generate Defense-Aware Adversarial Example

Gerar `x_adv_adaptive` utilizando Adaptive CW-L2.

Durante a busca pelo melhor adversarial, um candidato só pode ser aceito se satisfizer:

```python
C(x_adv_adaptive) != y_original
C(x_adv_adaptive) == C(T(x_adv_adaptive))
```

Se o ataque não encontrar uma amostra que satisfaça as duas condições:

```python
adaptive_failure += 1
```

---

## Step 6 — Evaluate Defense-Aware Example

Executar inferência na imagem adversarial adaptativa:

```python
adaptive_pred = C(x_adv_adaptive)
```

Aplicar a transformação:

```python
x_adv_adaptive_transformed = T(x_adv_adaptive)
```

Executar nova inferência:

```python
adaptive_transformed_pred = C(x_adv_adaptive_transformed)
```

A amostra é considerada sucesso defense-aware se:

```python
adaptive_pred != y_original
adaptive_pred == adaptive_transformed_pred
```

Nesse caso:

```python
adaptive_success += 1
adaptive_undetected += 1
```

Para manter o mesmo formato de métricas entre os dois ataques, `adaptive_detected` deve ser calculado como zero para amostras bem-sucedidas, pois o sucesso defense-aware já exige evasão do detector.

---

## Step 7 — L2 Distance

Para cada ataque bem-sucedido, calcular:

```python
l2_distance = norm(x_adv - x_original)
```

Manter listas separadas:

```python
blind_l2_distances
adaptive_l2_distances
```

---

# Counting Rules

## Total Valid Samples

Amostras originalmente classificadas corretamente:

```python
total_valid += 1
```

---

## Original Classified Wrong

Amostras limpas classificadas incorretamente:

```python
original_classified_wrong += 1
```

Essas amostras são excluídas da avaliação.

---

## Defense-Unaware Success

Ataque CW-L2 padrão altera a classificação:

```python
blind_success += 1
```

Condição:

```python
C(x_adv_blind) != y_original
```

---

## Defense-Unaware Failure

Ataque CW-L2 padrão não altera a classificação:

```python
blind_failure += 1
```

Condição:

```python
C(x_adv_blind) == y_original
```

---

## Defense-Unaware Detected

Ataque CW-L2 padrão é detectado pelo DeepDetector:

```python
blind_detected += 1
```

Condição:

```python
C(x_adv_blind) != C(T(x_adv_blind))
```

---

## Defense-Unaware Undetected

Ataque CW-L2 padrão não é detectado:

```python
blind_undetected += 1
```

Condição:

```python
C(x_adv_blind) == C(T(x_adv_blind))
```

---

## Defense-Aware Success

Ataque CW-L2 adaptativo altera a classificação e não é detectado:

```python
adaptive_success += 1
adaptive_undetected += 1
```

Condição:

```python
C(x_adv_adaptive) != y_original
C(x_adv_adaptive) == C(T(x_adv_adaptive))
```

---

## Defense-Aware Failure

Ataque CW-L2 adaptativo não encontra uma amostra válida:

```python
adaptive_failure += 1
```

---

## Defense-Aware Detected

Para o ataque defense-aware, candidatos detectados não devem ser contabilizados como sucesso.

A métrica `detected` para `defense_aware` deve representar apenas exemplos finais bem-sucedidos que ainda foram detectados na avaliação final.

Como o sucesso adaptativo exige evasão, espera-se:

```python
adaptive_detected = 0
```

Se a avaliação final detectar uma amostra retornada pelo ataque adaptativo, essa amostra deve ser tratada como falha de implementação ou falha do critério de aceitação do candidato.

---

# Metrics

## Attack Success Rate

```text
success / total_valid
```

---

## Detection Rate

```text
detected / success
```

Se `success == 0`, a métrica deve ser registrada como `0.0`.

---

## Evasion Rate

```text
undetected / success
```

Se `success == 0`, a métrica deve ser registrada como `0.0`.

---

## Failure Rate

```text
failures / total_valid
```

---

## Average L2 Distance

```text
mean(l2_distances)
```

Se não houver ataques bem-sucedidos, registrar:

```text
0.0
```

---

# Output Artifacts

A execução deve gerar apenas:

```text
results/
└── experiments/
    └── defense_aware/
        ├── metrics.csv
        └── metrics.json
```

---

## CSV Format

```csv
attack,total_valid,success,detected,undetected,failures,attack_success_rate_percent,detection_rate_percent,evasion_rate_percent,failure_rate_percent,mean_l2
```

Exemplo:

```csv
attack,total_valid,success,detected,undetected,failures,attack_success_rate_percent,detection_rate_percent,evasion_rate_percent,failure_rate_percent,mean_l2
defense_unaware,987,941,812,129,46,95.34,86.29,13.71,4.66,1.82
defense_aware,987,318,0,318,669,32.22,0.00,100.00,67.78,2.47
```

Os números acima são apenas ilustrativos.

---

## JSON Format

O arquivo `metrics.json` deve conter os mesmos valores semânticos do CSV.

Ele não deve incluir metadados extras.

```json
{
  "defense_unaware": {
    "total_valid": 0,
    "success": 0,
    "detected": 0,
    "undetected": 0,
    "failures": 0,
    "attack_success_rate_percent": 0.0,
    "detection_rate_percent": 0.0,
    "evasion_rate_percent": 0.0,
    "failure_rate_percent": 0.0,
    "mean_l2": 0.0
  },
  "defense_aware": {
    "total_valid": 0,
    "success": 0,
    "detected": 0,
    "undetected": 0,
    "failures": 0,
    "attack_success_rate_percent": 0.0,
    "detection_rate_percent": 0.0,
    "evasion_rate_percent": 0.0,
    "failure_rate_percent": 0.0,
    "mean_l2": 0.0
  }
}
```

---

# Reproducibility

A execução deve usar a semente definida em configuração:

```yaml
seed: 42
```

A seed deve ser aplicada sempre que o projeto já possuir suporte para controle determinístico.

Não adicionar metadados de reprodutibilidade ao `metrics.json`.

Se for necessário documentar limitações de reprodutibilidade, isso deve ser feito no código, em comentários internos, ou em documentação separada fora dos artefatos oficiais deste experimento.

---

# Architecture Constraints

O experimento deve seguir o padrão arquitetural do projeto.

Toda execução deve partir de:

```text
scripts/run_experiment.py
```

A lógica deve residir em:

```text
src/deepdetector/
```

Nenhuma lógica experimental deve ser implementada diretamente em scripts.

---

## Suggested Internal Modules

A implementação pode criar ou reutilizar os seguintes módulos internos:

```text
src/deepdetector/attacks/nn_robust.py
src/deepdetector/attacks/adaptive_cw_l2.py
src/deepdetector/filters/adaptive_noise_reduction.py
src/deepdetector/evaluation/defense_aware.py
src/deepdetector/experiments/defense_aware.py
```

O script público deve apenas resolver a configuração e chamar o experimento interno.

---

# Implementation Requirements

## Adaptive Attack Requirement

A implementação do ataque adaptativo deve aceitar uma função de transformação:

```python
transform_fn
```

ou um objeto detector:

```python
detector
```

O ataque adaptativo não deve duplicar manualmente toda a lógica do filtro dentro da classe de ataque caso já exista uma implementação reutilizável em:

```text
src/deepdetector/filters/
```

---

## Detector Consistency Requirement

A mesma transformação `T` deve ser usada em:

1. avaliação de exemplos defense-unaware;
2. critério de sucesso do ataque defense-aware;
3. métricas finais.

Não podem existir versões divergentes da transformação dentro do experimento.

---

## Value Range Requirement

O experimento deve preservar a escala esperada pelo MNIST M2:

```text
[0.0, 1.0]
```

Não converter para `[-0.5, 0.5]` sem uma justificativa explícita e sem converter de volta antes da inferência.

---

## Cross Mean Filter Requirement

A spec deve usar a API atual do projeto:

```python
cross_mean_filter(image, radius=3)
```

Não usar diretamente a assinatura do código original:

```python
cross_mean_filter(image, start=3, end=25, coefficient=13)
```

---

## Success Criterion Requirement

Para o ataque defense-aware, não basta verificar:

```python
C(x_adv) != y_original
```

Também é obrigatório verificar:

```python
C(x_adv) == C(T(x_adv))
```

A condição completa é:

```python
C(x_adv) != y_original and C(x_adv) == C(T(x_adv))
```

---

## Configuration Requirement

Este experimento pode usar `attacks` no plural, mesmo que outros experimentos usem `attack` no singular.

Essa exceção é permitida porque o experimento precisa comparar dois ataques internos sob uma única interface pública.

O runner central deve ter tratamento específico para o experimento:

```text
defense_aware
```

---

# Forbidden Artifacts

Não criar:

```text
results/experiments/defense_aware/debug/
```

Não criar:

```text
results/experiments/defense_aware/report.md
```

Não criar:

```text
results/experiments/defense_aware/diagnostic.json
```

Não criar:

```text
results/experiments/defense_aware/adversarial_examples/
```

---

# Acceptance Criteria

* O experimento é executado por:

```bash
python scripts/run_experiment.py --experiment defense_aware
```

* O experimento executa internamente os cenários defense-unaware e defense-aware.
* O defense-unaware usa CW-L2 padrão via `nn_robust_attacks.CarliniL2`.
* O defense-aware usa CW-L2 adaptativo com `nn_robust_attacks.CarliniL2` como
  ataque base.
* O ataque adaptativo conhece a transformação `T`.
* A transformação `T` corresponde ao filtro final adaptativo de detecção do projeto.
* A transformação `T` não é descrita nem implementada genericamente como união das Tables 7, 8 e 9.
* MNIST M2 usa escala `[0.0, 1.0]`.
* O clamp usa `[0.0, 1.0]`.
* O filtro espacial usa `cross_mean_filter(image, radius=3)`.
* A configuração usa `start` e `end`, não `samples`.
* A configuração usa `attacks` no plural apenas para este experimento.
* O ataque adaptativo só aceita candidatos que enganem o classificador e não sejam detectados.
* A condição abaixo é obrigatória no critério de sucesso adaptativo:

```python
C(x_adv) != y_original and C(x_adv) == C(T(x_adv))
```

* Amostras originalmente classificadas incorretamente são ignoradas.
* Distância L2 é calculada apenas para ataques bem-sucedidos.
* O CSV possui exatamente:

```csv
attack,total_valid,success,detected,undetected,failures,attack_success_rate_percent,detection_rate_percent,evasion_rate_percent,failure_rate_percent,mean_l2
```

* O JSON possui os mesmos valores semânticos do CSV.
* O JSON não contém metadados extras.
* Nenhum relatório adicional é produzido.
* Nenhum diagnóstico é produzido.
* Nenhum artefato morto é criado.
* Toda a lógica permanece compatível com o runner centralizado do projeto.
