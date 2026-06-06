# Spec — Top-k Detector: extensão de regras

**Projeto:** DeepDetector — replicação  
**Branch de origem:** `feat/topk`  
**Escopo:** extensão do experimento `topk_detection` existente, sem criar novo experimento  
**Status:** proposta para implementação

---

## Glossário fixo

Os termos abaixo têm significado preciso neste documento. Ambiguidade entre eles foi a principal fonte de erros na spec anterior.

| Termo | Definição |
|---|---|
| `true_label` | Classe verdadeira da imagem antes de qualquer perturbação. Usa label space ImageNet/Caffe (cab=468, panda=388, zebra=340). |
| `top1_before_filter` | Classe com maior probabilidade na entrada do filtro. Para imagem limpa: geralmente igual a `true_label`, mas não garantido. Para imagem adversarial: geralmente diferente de `true_label`. |
| `top1_after_filter` | Classe com maior probabilidade na saída do filtro. |
| `topk_after_filter` | Conjunto das k classes com maiores probabilidades na saída do filtro. |
| Falso positivo (FP) | Imagem limpa detectada como adversarial. |
| Verdadeiro positivo (TP) | Imagem adversarial detectada como adversarial. |

Todas as regras de detecção operam sobre `top1_before_filter`, não sobre `true_label`. Isso é intencional: o detector não tem acesso ao rótulo verdadeiro em tempo de inferência.

---

## 1. Contexto e problema

O detector original usa:

```python
detected = argmax(f(x)) != argmax(f(T(x)))
```

Equivalente a comparar `top1_before_filter` com `top1_after_filter`. A branch `feat/topk` substituiu essa regra por interseção de conjuntos top-k:

```python
detected = len(set(topk_before_filter) & set(topk_after_filter)) == 0
```

Os resultados observados no conjunto de imagens ambíguas (selecionadas com margem top1-top2 < 0.15, confiança top-1 entre 0.20 e 0.60) foram:

| k | TP | FN | FP | TN | Recall | FPR  | F1   |
|---|----|----|----|----|--------|------|------|
| 1 | 59 | 41 | 72 | 28 | 59%    | 72%  | 51%  |
| 2 | 33 | 67 | 27 | 73 | 33%    | 27%  | 41%  |
| 3 | 22 | 78 | 16 | 84 | 22%    | 16%  | 32%  |
| 5 | 13 | 87 |  8 | 92 | 13%    |  8%  | 21%  |

Esses valores globais são a referência de regressão para a regra atual
`topk_overlap`. O projeto ainda não possui `tests/fixtures/` versionado para
este experimento; portanto a implementação não deve depender de fixtures
inexistentes. Se fixtures forem adicionadas futuramente, elas devem ser
criadas no mesmo PR da implementação e seus hashes devem ser documentados na
spec ou no teste correspondente.

O padrão FPR alto com k=1 e recall colapsando com k >= 3 indica que a regra de interseção vazia é conservadora demais para imagens ambíguas: basta uma classe permanecer no top-k (comum quando as classes são semanticamente próximas, como veículos) para o detector não acusar adversarial.

---

## 2. Stack e configuração: sem alterações

Esta spec **não altera** o modelo, o dataset, o espaço de labels, o filtro, nem o script de execução. Todos os experimentos novos rodam com a mesma configuração do `topk_detection` existente.

| Parâmetro | Valor atual — não alterar |
|---|---|
| Modelo | GoogLeNet/Caffe |
| `image_size` | 224 |
| `value_range` | [0.0, 1.0] |
| `class_indices` | cab=468, panda=388, zebra=340 |
| Filtro | `proposed_detection_filter` |
| Script de execução | `python scripts/run_experiment.py --experiment topk_detection` |

Qualquer resultado que cite a tabela de referência acima só é reproduzível com essa configuração. A spec anterior propunha Inception v3 com índices diferentes, o que tornava a reprodução impossível e os valores de referência sem sentido.

O FGSM deste experimento deve usar o caminho de gradiente de ataque do wrapper
GoogLeNet/Caffe (`model.gradient`), que usa `attack_deploy_proto` /
`deploy_removeSoftmax`. Não usar `model.prediction_gradient` neste experimento:
na configuração atual esse caminho pode retornar gradiente zero para a rede
original com softmax, gerando uma imagem adversarial idêntica à limpa e
invalidando `test_number`, `TP` e `FN`. Esta regra é específica de
`topk_detection`; não alterar a preferência global de
`generate_fgsm_caffe_image()`, pois outros experimentos dependem dela.

---

## 3. Estrutura do experimento

A spec original define `topk_detection` como o único experimento público em `configs/experiments.yaml`. Esta spec adiciona um campo `rules` dentro de `experiments.topk_detection.topk_detection`, sem criar um experimento paralelo.

```yaml
# configs/experiments.yaml — seção existente, com adição de rules

experiments:
  topk_detection:
    topk_detection:
      # ... configurações existentes mantidas integralmente ...

      baseline_k: 1                   # campo legado preservado
      candidate_k_values: [2, 3, 5]   # campo legado preservado
      default_k: 3                    # campo legado preservado

      rules:
        - name: topk_overlap          # regra atual da branch; deve reproduzir a tabela de referência
        - name: top1_in_topk          # usa baseline_k + candidate_k_values
        - name: rank_displacement
          thresholds: [1, 2, 3, 5]    # zero-based: threshold=k equivale a top1_in_topk(k)
        - name: confidence_drop
          ks: [2, 3]
          deltas: [0.05, 0.10, 0.20]
```

Regras que dependem de `k` e não declaram `ks` explicitamente usam a lista
`[baseline_k] + candidate_k_values`, preservando o comportamento atual para
`topk_overlap`. Quando `rules` não estiver configurado, o experimento deve se
comportar como hoje: avaliar apenas `topk_overlap` para `baseline_k` e
`candidate_k_values`, gerando os mesmos artefatos e o mesmo formato legado.

O campo `name: topk_overlap` com `k ∈ {1,2,3,5}` deve reproduzir os números da
tabela de referência. Se não reproduzir, há uma regressão, não uma feature
nova.

---

## 4. Módulo

O código novo vai em `topk_detection.py` existente — não em um arquivo separado. As funções atuais (`is_detected_topk`, `top_k_indices`) são mantidas sem alteração para não quebrar testes existentes. As novas funções são adicionadas ao mesmo módulo.

```python
# Adições a src/deepdetector/evaluation/topk_detection.py

import numpy as np


# ── já existente, preservar comportamento público ───────────────────────────

def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Função existente. Manter validações e ordenação estável já implementadas."""


def is_detected_topk(probs_before: np.ndarray, probs_after: np.ndarray, k: int) -> bool:
    """Regra atual da branch. Manter sem alteração semântica."""


# ── novas funções ────────────────────────────────────────────────────────────

def detect_top1_change(before: np.ndarray, after: np.ndarray) -> bool:
    """
    Detector original (baseline). Equivalente a is_detected_topk com k=1.
    Não usa k como parâmetro para deixar explícito que é uma regra distinta.
    """
    return int(np.argmax(before)) != int(np.argmax(after))


def detect_top1_in_topk(before: np.ndarray, after: np.ndarray, k: int) -> bool:
    """
    Detecta se top1_before_filter saiu do top-k após o filtro.

    Regra assimétrica: só verifica se a classe que era top-1 na entrada
    ainda aparece entre as top-k classes na saída. Não exige que os dois
    conjuntos sejam disjuntos.

    Motivação: em imagens limpas ambíguas, o filtro pode reordenar o top-1
    mas tende a manter a classe original no top-k. Em adversariais fortes,
    a perturbação empurra a classe original para além do top-k.
    """
    top1_before = int(np.argmax(before))
    return top1_before not in set(top_k_indices(after, k))


def detect_rank_displacement(before: np.ndarray, after: np.ndarray, threshold: int) -> bool:
    """
    Detecta se top1_before_filter caiu para uma posição >= threshold no ranking filtrado.

    Rank é zero-based: posição 0 = top-1, posição 1 = segundo lugar, etc.

    Equivalência com top1_in_topk (verificada em testes):
        detect_rank_displacement(before, after, threshold=k)
        ==
        detect_top1_in_topk(before, after, k)

    Útil para análise de sensibilidade: permite observar em que posição os
    adversariais empurram a classe original, não apenas se ela saiu do top-k.
    """
    top1_before = int(np.argmax(before))
    ranking_after = list(np.argsort(after)[::-1])
    rank = ranking_after.index(top1_before) if top1_before in ranking_after else len(ranking_after)
    return rank >= threshold


def detect_confidence_drop(
    before: np.ndarray,
    after: np.ndarray,
    k: int,
    delta: float,
) -> bool:
    """
    Detecta se top1_before_filter saiu do top-k OU se sua confiança caiu >= delta.

    Dois caminhos independentes de detecção:
    - Posição: top1_before_filter não está mais em topk_after_filter.
    - Confiança: probabilidade de top1_before_filter caiu mais que delta,
      mesmo que a classe ainda apareça no top-k.

    Motivação: adversariais fracos podem manter a classe original no top-k
    mas reduzir sua confiança de forma anormal. is_detected_topk e
    detect_top1_in_topk não capturam esse caso.

    Risco: o filtro de defesa também pode reduzir confiança em imagens
    limpas, aumentando FPR. O valor de delta precisa de calibração empírica.
    """
    top1_before = int(np.argmax(before))
    position_changed = top1_before not in set(top_k_indices(after, k))
    confidence_dropped = (float(before[top1_before]) - float(after[top1_before])) >= delta
    return position_changed or confidence_dropped


_RULE_DISPATCH = {
    "top1_change":       lambda before, after, k, threshold, delta: detect_top1_change(before, after),
    "topk_overlap":      lambda before, after, k, threshold, delta: is_detected_topk(before, after, k),
    "top1_in_topk":      lambda before, after, k, threshold, delta: detect_top1_in_topk(before, after, k),
    "rank_displacement": lambda before, after, k, threshold, delta: detect_rank_displacement(before, after, threshold),
    "confidence_drop":   lambda before, after, k, threshold, delta: detect_confidence_drop(before, after, k, delta),
}


def detect_by_rule(
    before: np.ndarray,
    after: np.ndarray,
    rule: str,
    k: int = 1,
    threshold: int = 1,
    delta: float = 0.10,
) -> bool:
    """
    Ponto de entrada único. Parâmetros não relevantes para a regra são ignorados.

    Args:
        before:    scores softmax antes do filtro (shape: [num_classes])
        after:     scores softmax após o filtro   (shape: [num_classes])
        rule:      nome da regra (ver _RULE_DISPATCH)
        k:         tamanho do conjunto top-k (usado por topk_overlap, top1_in_topk, confidence_drop)
        threshold: posição mínima para rank_displacement (zero-based)
        delta:     queda mínima de confiança para confidence_drop
    """
    if rule not in _RULE_DISPATCH:
        raise ValueError(f"Regra desconhecida: '{rule}'. Disponíveis: {list(_RULE_DISPATCH)}")
    return _RULE_DISPATCH[rule](before, after, k, threshold, delta)
```

---

## 5. Saída esperada

### 5.1 Métricas por (classe, regra, parâmetro)

```json
{
  "class": "cab",
  "rule": "top1_in_topk",
  "k": 2,
  "threshold": null,
  "delta": null,
  "selected": 50,
  "TP": 0,
  "FN": 0,
  "FP": 0,
  "TN": 0,
  "recall_percent": 0.0,
  "precision_percent": 0.0,
  "f1_percent": 0.0,
  "false_positive_rate_percent": 0.0,
  "false_positive_reduction_percent": 0.0,
  "attack_success_rate_percent": 0.0
}
```

`false_positive_reduction_percent` é sempre calculado em relação a `top1_change`
(o detector original), independente de qual regra está sendo medida. O nome do
campo preserva o padrão já usado em `topk_detection.py`; a mudança é apenas a
base de comparação quando múltiplas regras estiverem configuradas.

### 5.2 Auditoria por amostra

```json
{
  "sample_id": "cab/n02930766_123.JPEG",
  "class_name": "cab",
  "true_label": 468,
  "is_adversarial": false,
  "top1_before_filter": 468,
  "top1_before_filter_confidence": 0.312,
  "top1_after_filter": 817,
  "top1_after_filter_confidence": 0.289,
  "top5_before_filter": [468, 817, 511, 654, 555],
  "top5_after_filter":  [817, 468, 511, 555, 654],
  "confidence_of_top1_before_in_after": 0.289,
  "rank_of_top1_before_in_after": 1,
  "detections": {
    "top1_change":           true,
    "topk_overlap__k1":      true,
    "topk_overlap__k2":      false,
    "top1_in_topk__k2":      false,
    "top1_in_topk__k3":      false,
    "rank_displacement__t1": true,
    "rank_displacement__t2": false,
    "confidence_drop__k2__d010": false
  }
}
```

Campos obrigatórios para auditabilidade interna:

- `rank_of_top1_before_in_after`: posição zero-based de `top1_before_filter` no ranking pós-filtro. Permite verificar manualmente por que cada regra decidiu como decidiu.
- `confidence_of_top1_before_in_after`: probabilidade da classe `top1_before_filter` após o filtro. Necessário para auditar `confidence_drop`.
- `true_label` usa label space ImageNet/Caffe consistente com o resto do projeto.

Essa estrutura descreve o registro lógico usado para auditoria e testes. Ela
não implica a criação de um arquivo público próprio; se persistida, deve ficar
dentro de `metrics.json`, conforme a seção de arquivos de saída.

### 5.3 Arquivos de saída

```
results/experiments/topk_detection/
├── ambiguous_images/           # já adotado pela spec original
├── selection.csv               # formato legado preservado
├── selection.json              # formato legado preservado
├── metrics.csv                 # métricas das regras configuradas
└── metrics.json                # métricas das regras configuradas
```

Não criar `metrics_topk_overlap.json`, `metrics_all_rules.json`,
`metrics_all_rules.csv` ou `per_sample.jsonl`. O padrão já adotado no projeto é
manter artefatos oficiais com nomes estáveis (`metrics.csv`, `metrics.json`,
`selection.csv`, `selection.json`) e evitar arquivos públicos paralelos para o
mesmo experimento.

Quando `rules` não estiver configurado, `metrics.csv` e `metrics.json` mantêm o
formato legado atual (`class × k`) para que os testes existentes continuem
válidos. Quando `rules` estiver configurado, os mesmos arquivos oficiais devem
incluir as colunas/campos `rule`, `k`, `threshold` e `delta`, usando `null` para
parâmetros que não se aplicam.

Os campos de auditabilidade por amostra (`rank_of_top1_before_in_after` e
`confidence_of_top1_before_in_after`) devem ser calculados internamente para
explicar as decisões das regras. Se a auditoria por amostra for persistida, ela
deve ficar dentro de `metrics.json` sob uma chave própria, sem criar arquivo
standalone.

---

## 6. Testes unitários

Arquivo: `tests/test_topk_detection_rules.py`

Os testes que já existem em `tests/test_topk_detection.py` não são alterados.

```python
import numpy as np
import pytest
from deepdetector.evaluation.topk_detection import (
    detect_top1_change,
    detect_top1_in_topk,
    detect_rank_displacement,
    detect_confidence_drop,
    is_detected_topk,
    detect_by_rule,
)


# ── top1_change ──────────────────────────────────────────────────────────────

def test_top1_change_detects_when_argmax_changes():
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.1, 0.8, 0.1])
    assert detect_top1_change(before, after) is True


def test_top1_change_no_detection_when_argmax_stable():
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.7, 0.2, 0.1])
    assert detect_top1_change(before, after) is False


# ── top1_in_topk ─────────────────────────────────────────────────────────────

def test_top1_in_topk_detects_when_original_class_expelled():
    # top1_before = 0; top-2 after = {1, 2}; 0 not in {1, 2}
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.1, 0.5, 0.4])
    assert detect_top1_in_topk(before, after, k=2) is True


def test_top1_in_topk_no_detection_when_original_class_remains():
    # top1_before = 0; top-2 after = {1, 0}; 0 in {1, 0}
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.4, 0.5, 0.1])
    assert detect_top1_in_topk(before, after, k=2) is False


def test_top1_in_topk_k1_equivalent_to_top1_change():
    """top1_in_topk com k=1 deve ser idêntico a top1_change."""
    rng = np.random.default_rng(99)
    for _ in range(500):
        before = rng.dirichlet(np.ones(10))
        after  = rng.dirichlet(np.ones(10))
        assert detect_top1_in_topk(before, after, k=1) == detect_top1_change(before, after)


# ── rank_displacement ────────────────────────────────────────────────────────

def test_rank_displacement_at_boundary():
    # top1_before = 0; ranking_after = [2, 1, 0]; rank_of_0 = 2 (zero-based)
    before = np.array([0.7, 0.2, 0.1])
    after  = np.array([0.1, 0.3, 0.6])
    assert detect_rank_displacement(before, after, threshold=2) is True
    assert detect_rank_displacement(before, after, threshold=3) is False


def test_rank_displacement_equivalence_with_top1_in_topk():
    """
    rank_displacement(threshold=k) deve ser idêntico a top1_in_topk(k).
    Rank é zero-based: threshold=k significa 'caiu para além das primeiras k posições'.
    """
    rng = np.random.default_rng(42)
    for _ in range(2000):
        before = rng.dirichlet(np.ones(10))
        after  = rng.dirichlet(np.ones(10))
        for k in [1, 2, 3, 5]:
            a = detect_top1_in_topk(before, after, k=k)
            b = detect_rank_displacement(before, after, threshold=k)
            assert a == b, f"Mismatch at k={k}: top1_in_topk={a}, rank_displacement={b}"


# ── confidence_drop ──────────────────────────────────────────────────────────

def test_confidence_drop_detects_via_position():
    # top1_before = 0; 0 not in top-2 after → detecta independente de delta
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.1, 0.5, 0.4])
    assert detect_confidence_drop(before, after, k=2, delta=0.99) is True


def test_confidence_drop_detects_via_delta_only():
    # top1_before = 0; 0 ainda no top-2; mas confiança caiu 0.20
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.6, 0.3, 0.1])
    assert detect_confidence_drop(before, after, k=2, delta=0.10) is True
    assert detect_confidence_drop(before, after, k=2, delta=0.30) is False


def test_confidence_drop_no_detection_below_threshold():
    # top1_before = 0; 0 ainda no top-2; confiança quase estável
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.78, 0.12, 0.1])
    assert detect_confidence_drop(before, after, k=2, delta=0.10) is False


# ── topk_overlap (regressão) ──────────────────────────────────────────────────

def test_topk_overlap_k1_equivalent_to_top1_change():
    """Regressão: is_detected_topk com k=1 deve ser idêntico a detect_top1_change."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        before = rng.dirichlet(np.ones(1000))
        after  = rng.dirichlet(np.ones(1000))
        assert is_detected_topk(before, after, k=1) == detect_top1_change(before, after)


# ── dispatch ─────────────────────────────────────────────────────────────────

def test_detect_by_rule_unknown_raises():
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.1, 0.8, 0.1])
    with pytest.raises(ValueError, match="Regra desconhecida"):
        detect_by_rule(before, after, rule="regra_inventada")


def test_detect_by_rule_dispatches_correctly():
    before = np.array([0.8, 0.1, 0.1])
    after  = np.array([0.1, 0.5, 0.4])
    assert detect_by_rule(before, after, rule="top1_change") is True
    assert detect_by_rule(before, after, rule="top1_in_topk", k=2) is True
    assert detect_by_rule(before, after, rule="topk_overlap", k=2) is False
```

---

## 7. Hipóteses a validar

As hipóteses abaixo devem ser confirmadas ou refutadas pelos resultados do experimento. Não são garantias de implementação.

**H1 — `top1_in_topk` terá melhor F1 que `topk_overlap` para k=2.**  
Fundamento: a regra assimétrica é menos restritiva que interseção vazia; deve perder menos TPs sem ganhar muitos FPs.  
Risco: se os adversariais de `cab` mantiverem `top1_before_filter` no top-2 pós-filtro, o recall de `cab` pode ser semelhante ao de `topk_overlap`.

**H2 — `cab` continuará com recall inferior a `panda` para qualquer regra com k >= 2.**  
Fundamento: os dados da branch mostram que adversariais em `cab` não removem a classe original do top-k (TP=0 com k=5 para `topk_overlap`), sugerindo que o problema é inerente à ambiguidade semântica entre classes de veículos, não à escolha de regra.  
Implicação: se H2 for confirmada, o relatório deve concluir que o recall baixo em `cab` exige um filtro adversarial mais forte, não apenas uma regra top-k diferente.

**H3 — `confidence_drop` com delta=0.10 terá recall maior que `top1_in_topk` para o mesmo k.**  
Fundamento: o segundo caminho de detecção (queda de confiança sem expulsão do top-k) deve capturar adversariais fracos que `top1_in_topk` perde.  
Risco: o filtro de defesa pode reduzir a confiança de imagens limpas ambíguas na mesma proporção, aumentando FPR proporcionalmente.

**H4 — `rank_displacement` com threshold=k produzirá resultados idênticos a `top1_in_topk` com k (verificado em testes).**  
Esta hipótese é matemática, não experimental. O teste `test_rank_displacement_equivalence_with_top1_in_topk` deve passar com 0 discrepâncias. Se falhar, é bug.

---

## 8. Critérios de aceitação

A implementação é correta quando todos os itens abaixo passam:

1. `topk_overlap` para k∈{1,2,3,5} reproduz os números globais da tabela de referência desta spec com diferença máxima de 2 amostras em TP, FP, TN e FN quando executado com a configuração oficial `topk_detection`.

2. O teste `test_rank_displacement_equivalence_with_top1_in_topk` passa para 2000 amostras aleatórias sem nenhuma discrepância.

3. O teste `test_topk_overlap_k1_equivalent_to_top1_change` passa para 500 amostras aleatórias sem nenhuma discrepância.

4. A avaliação calcula `rank_of_top1_before_in_after` e `confidence_of_top1_before_in_after` para todas as amostras avaliadas, e esses campos aparecem na auditoria por amostra caso ela seja persistida em `metrics.json`.

5. `metrics.csv` e `metrics.json` cobrem todas as combinações de regra × parâmetro definidas em `experiments.topk_detection.topk_detection.rules`.

6. O comportamento legado continua disponível quando `rules` não está configurado: `metrics.csv`, `metrics.json`, `selection.csv`, `selection.json` e `ambiguous_images/` são gerados no formato atual, e os testes preexistentes em `tests/test_topk_detection.py` continuam passando.

7. As Tables 7, 8, 9 e 10 do relatório não são afetadas.

---

## 9. O que esta spec não cobre

- **Alteração do filtro de defesa.** Se o filtro mudar, todos os resultados ficam incomparáveis com a tabela de referência.
- **Configuração diferente de k por classe.** Usar k distintos para `cab` e `panda` é uma extensão futura.
- **Ataque adaptativo.** O adversarial não foi otimizado contra as novas regras. Se for, o recall observado será menor.
- **Outros modelos ou datasets.** A spec é específica para GoogLeNet/Caffe + ImageNet com as três classes definidas.
