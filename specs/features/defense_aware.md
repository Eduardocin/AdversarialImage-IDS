# SPEC — Defense-Aware Adaptive CW-L2 Attack

## Objective

Implementar o experimento de **defense-aware attack** do artigo *Detecting Adversarial Image Examples in Deep Networks with Adaptive Noise Reduction*, avaliando a robustez do DeepDetector contra um ataque Carlini & Wagner L2 adaptativo.

A implementação deve comparar dois cenários:

1. **Defense-unaware attack**: o atacante gera exemplos adversariais contra o classificador original, sem considerar o detector.
2. **Defense-aware attack**: o atacante conhece o detector e tenta gerar exemplos adversariais que enganem o classificador e também não sejam detectados após a transformação adaptativa.

A implementação deve seguir a arquitetura atual do projeto, utilizando exclusivamente o runner centralizado:

```bash
python scripts/run_experiment.py --experiment defense_aware
```

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

## Out Of Scope

Além dos itens excluídos no escopo, esta especificação não inclui:

* vendorização dos arquivos originais em `src/deepdetector`;
* uso dos arquivos em `temp/` como dependência de runtime;
* uso de `nn_robust_attacks/l2_adaptive_attack.py` como backend oficial do ataque adaptativo;
* criação de uma interface pública separada para o ataque adaptativo;
* alteração da lógica oficial do filtro final adaptativo fora do necessário para manter consistência com esta especificação.

---

## Context

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

A implementação deste projeto deve reproduzir o comportamento metodológico desses arquivos de forma compatível com a arquitetura atual, mas **sem usar os arquivos de `temp/` ou `Test/CW/` como parte do projeto**.

O ataque adaptativo oficial deve ser uma implementação nativa em `src/deepdetector`. A referência original serve apenas para entender a regra de busca: durante o loop de otimização do CW-L2, o melhor candidato defense-aware só é atualizado quando o candidato também evade a transformação do detector.

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
      type: native_adaptive_cw_l2
      targeted: false
      confidence: 0
      max_iterations: 2000
      binary_search_steps: 5
      initial_const: 1.0
      learning_rate: 0.1
      input_range:
        min: 0.0
        max: 1.0

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

O ataque defense-aware deve utilizar uma implementação nativa adaptativa do CW-L2 em `src/deepdetector`.

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

Essa rejeição deve acontecer durante a busca do CW-L2, não apenas depois que um CW-L2 padrão termina.

---

## Native Adaptive CW-L2 Search

A implementação oficial do ataque defense-aware deve ser nativa no pacote `deepdetector`.

Ela não deve importar, copiar em runtime, executar ou depender de:

```text
temp/adaptive_CWL2_MNIST.py
temp/l2_adaptive_attack.py
nn_robust_attacks/l2_adaptive_attack.py
```

Os arquivos em `temp/` podem ser usados apenas como referência de comportamento durante análise humana. Eles não devem ser tratados como código do projeto, fixture de teste, backend configurável ou dependência de execução.

O ataque nativo deve seguir a estrutura conceitual do CW-L2:

1. otimizar uma variável modificadora em espaço limitado por caixa;
2. usar a perda CW-L2 padrão para gerar candidatos adversariais;
3. executar `binary_search_steps` sobre a constante de trade-off;
4. executar até `max_iterations` passos de otimização por etapa de busca;
5. acompanhar o menor L2 para candidatos que enganam o classificador;
6. acompanhar separadamente o menor L2 para candidatos que também evadem a defesa.

Durante cada iteração de otimização, para cada candidato `x_candidate` produzido pelo estado atual do otimizador, a implementação deve avaliar:

```python
candidate_pred = C(x_candidate)
transformed_pred = C(T(x_candidate))
```

Um candidato só pode atualizar o melhor ataque defense-aware global quando todas as condições forem verdadeiras:

```python
candidate_pred != y_original
candidate_pred == transformed_pred
l2_distance(x_candidate, x_original) < best_defense_aware_l2
```

Para ataques untargeted, `candidate_pred != y_original` é o critério de adversarialidade. Se ataques targeted forem configurados futuramente, o critério de adversarialidade deve seguir a semântica CW-L2 targeted, mas a evasão continua exigindo:

```python
candidate_pred == transformed_pred
```

A atualização da constante de busca binária deve continuar baseada no sucesso adversarial do CW-L2 contra o classificador, como no ataque original. A atualização do melhor exemplo retornado pelo cenário defense-aware deve ser mais restrita e exigir a condição completa:

```python
C(x_candidate) != y_original and C(x_candidate) == C(T(x_candidate))
```

Se nenhum candidato defense-aware válido for encontrado em todas as iterações e etapas de busca, o ataque deve sinalizar falha de forma compatível com o avaliador atual. A forma permitida é retornar a imagem limpa original naquela posição, desde que a avaliação final conte a amostra como `adaptive_failure`.

É proibido implementar o ataque oficial como:

1. executar CW-L2 padrão até o final;
2. avaliar somente o adversarial final retornado pelo CW-L2;
3. aceitar ou rejeitar esse único resultado por pós-filtragem.

Essa estratégia de pós-filtragem pode existir apenas como implementação experimental interna, mas não pode ser usada pelo experimento oficial `defense_aware`.

---

## Business Rules

* Amostras limpas classificadas incorretamente pelo MNIST M2 são excluídas da avaliação.
* O ataque defense-unaware tem sucesso quando `C(x_adv) != y_original`.
* O ataque defense-unaware é detectado quando `C(x_adv) != C(T(x_adv))`.
* O ataque defense-aware tem sucesso somente quando `C(x_adv) != y_original` e `C(x_adv) == C(T(x_adv))`.
* Candidatos defense-aware detectados por `T` não podem ser contabilizados como sucesso.
* A distância L2 entra nas médias somente para ataques bem-sucedidos.
* A mesma transformação `T` deve ser usada durante a geração adaptativa, a avaliação defense-unaware e a avaliação final defense-aware.

---

## Functional Requirements

* Executar o experimento somente pelo runner central `scripts/run_experiment.py`.
* Carregar MNIST test no intervalo configurado por `start` e `end`.
* Carregar e avaliar o modelo MNIST M2.
* Gerar o ataque defense-unaware com CW-L2 padrão.
* Gerar o ataque defense-aware com CW-L2 adaptativo nativo.
* Avaliar candidatos intermediários do ataque adaptativo dentro do loop de otimização.
* Escrever somente `metrics.csv` e `metrics.json` no diretório oficial do experimento.

---

## Non-Functional Requirements

* Manter a implementação simples, legível e compatível com os padrões existentes em `deepdetector`.
* Evitar dependências novas quando as APIs atuais de TensorFlow/Keras e NumPy forem suficientes.
* Preservar a escala pública `[0.0, 1.0]` para imagens avaliadas pelo modelo, filtro e métricas.
* Não gerar datasets, pesos, adversariais persistidos, diagnósticos ou relatórios adicionais.
* Manter os arquivos em `temp/` fora do projeto e fora dos testes automatizados.

---

## Error Cases

* Se `dataset.name` não for `mnist`, a execução deve falhar com erro claro.
* Se o modelo MNIST M2 ou seu checkpoint não estiver disponível, a execução deve falhar com erro claro.
* Se o ataque adaptativo não receber `transform_fn` ou detector equivalente, a execução deve falhar com erro claro.
* Se o ataque adaptativo não receber uma função de predição compatível, a execução deve falhar com erro claro.
* Se `input_range` for inválido, a execução deve falhar com erro claro.
* Se nenhum candidato defense-aware válido for encontrado para uma amostra, a amostra deve ser contada como `adaptive_failure`.
* Se a avaliação final detectar uma amostra retornada como sucesso defense-aware, a amostra deve ser tratada como falha de implementação ou falha do critério de aceitação.

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

Gerar `x_adv_adaptive` utilizando o CW-L2 adaptativo nativo.

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

O ataque adaptativo deve avaliar candidatos intermediários dentro do loop de otimização do CW-L2. A função `transform_fn` deve participar do critério de aceitação desses candidatos intermediários.

O ataque adaptativo oficial não deve depender do backend externo `CarliniL2Adaptive`. O uso de `nn_robust_attacks.CarliniL2` continua permitido para o cenário defense-unaware e para comparação, mas não satisfaz por si só o requisito do cenário defense-aware se o detector for aplicado apenas ao resultado final.

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

O ataque nativo pode usar parametrização interna em tanh-space ou outro espaço de otimização limitado por caixa. Porém, todo candidato avaliado por `C`, por `T`, pelas métricas e pelo avaliador deve estar na escala `[0.0, 1.0]`.

Não converter para `[-0.5, 0.5]` antes da inferência do MNIST M2 ou da transformação `T`.

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
* O defense-aware usa uma implementação nativa de CW-L2 adaptativo em `src/deepdetector`.
* O defense-aware não depende de `temp/l2_adaptive_attack.py` nem de `nn_robust_attacks/l2_adaptive_attack.py`.
* O ataque adaptativo conhece a transformação `T`.
* O ataque adaptativo avalia candidatos intermediários durante o loop de otimização.
* O ataque adaptativo mantém o melhor candidato defense-aware encontrado ao longo de todas as iterações e etapas de busca binária.
* O ataque adaptativo oficial não é implementado como CW-L2 padrão seguido de pós-filtragem do resultado final.
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
