# Spec de Refatoração — Reprodução de `Test_CWL2_ImageNet.py`

> **Branch:** `fix/inceptionv3-ampere`  
> **Prioridade:** Alta — bloqueia a validade dos resultados da Tabela 10  
> **Componentes afetados:**
> - `configs/table_10_inception_v3.yaml`
> - `src/deepdetector/models/imagenet_wrappers.py`
> - `loaders/imagenet_loader.py`

---

## Contexto

O objetivo da branch é reproduzir o script original `Test_CWL2_ImageNet.py`, que executa o ataque CW-L2 não-direcionado contra o Inception v3 no dataset ImageNet e mede TP, FP, FN e TTP do filtro DeepDetector.

Foram identificadas **5 falhas** que impedem a reprodução fiel dos resultados.

| ID  | Falha                                              | Severidade  |
|-----|----------------------------------------------------|-------------|
| F-1 | Subconjunto do dataset não respeita 40/40/20       | BLOQUEANTE  |
| F-2 | Ordem de classes não reproduz o script base        | MODERADA    |
| F-3 | Hiperparâmetros do CW-L2 precisam de validação     | MODERADA    |
| F-4 | Eager execution não desabilitado no TF2            | BLOQUEANTE  |
| F-5 | `input_map_name` do grafo Inception não validado   | CRÍTICA     |

---

## F-1 — Subconjunto do Dataset (BLOQUEANTE)

### Problema

O script original define manualmente **40 imagens de Zebra, 40 de Panda e 20 de Cab** (total = 100).

```python
# Test_CWL2_ImageNet.py
for i in range(40):
    inputs.append(data.zebraData[i])
    targets.append(data.zebraLabel[i])

for i in range(40):
    inputs.append(data.pandaData[i])
    targets.append(data.pandaLabel[i])

for i in range(20):
    inputs.append(data.cabData[i])
    targets.append(data.cabLabel[i])
```

A configuração atual usa `n_samples: 100` com `shuffle: true`, que seleciona 100 imagens aleatórias das três pastas sem garantir a proporção correta. Qualquer distribuição diferente de 40/40/20 torna os resultados incomparáveis com a Tabela 10.

### Mudança — `configs/table_10_inception_v3.yaml`

```yaml
# ANTES (incorreto)
dataset:
  n_samples: 100
  shuffle: true

# DEPOIS (correto)
dataset:
  name: imagenet
  split: test
  images_dir: data/inceptionV3
  image_size: 299
  value_range: [-0.5, 0.5]
  shuffle: false
  class_order: [zebra, panda, cab]
  class_indices:
    zebra: 80
    panda: 169
    cab: 267
  class_quotas:
    zebra: 40
    panda: 40
    cab: 20
```

### Mudança — script de materialização do subconjunto

Criar um script que separa o subconjunto por quotas antes do experimento e
materializa as imagens em `data/inceptionV3`. O experimento deve ler
`dataset.images_dir: data/inceptionV3`, não a pasta completa
`data/imagenet/test`.

O script deve copiar deterministicamente:

- as primeiras 40 imagens ordenadas de `data/imagenet/test/zebra`
- as primeiras 40 imagens ordenadas de `data/imagenet/test/panda`
- as primeiras 20 imagens ordenadas de `data/imagenet/test/cab`

O layout gerado deve ser:

```text
data/inceptionV3/
  zebra/
  panda/
  cab/
```

```python
# ANTES — corte global (quebra a proporção)
if n_samples is not None:
    rows = rows[:n_samples]

# DEPOIS — corte por classe
if class_quotas:
    per_class = []
    for cls in class_order:
        quota = class_quotas[cls]
        cls_rows = [r for r in rows if r['class'] == cls][:quota]
        per_class.extend(cls_rows)
    rows = per_class
```

### Critério de Aceitação

- Exatamente 40 amostras da classe Zebra (índice 80)
- Exatamente 40 amostras da classe Panda (índice 169)
- Exatamente 20 amostras da classe Cab (índice 267)
- O experimento lê as imagens já separadas em `data/inceptionV3`
- `shuffle: false` — ordem determinística

---

## F-2 — Ordem das Classes (MODERADA)

### Problema

O script original sempre processa Zebra → Panda → Cab nessa ordem. Não impacta as métricas finais, mas dificulta a comparação de índices de imagens entre a reprodução e o script base durante a depuração.

### Mudança — `configs/table_10_inception_v3.yaml`

```yaml
dataset:
  class_order: [zebra, panda, cab]
```

### Critério de Aceitação

- Índices 0–39 do batch correspondem a zebras
- Índices 40–79 correspondem a pandas
- Índices 80–99 correspondem a cabs

---

## F-3 — Hiperparâmetros do CW-L2 (MODERADA)

### Contexto

O original usa `CarliniL2` do repositório `nn_robust_attacks`. A reprodução usa CW-L2 via CleverHans. O ataque é **equivalente em objetivo** (métricas agregadas TP/FP/FN/TTP), mas não idêntico pixel a pixel. Para minimizar divergências, todos os hiperparâmetros devem bater exatamente.

### Tabela de Parâmetros

| Parâmetro              | Original    | Atual  | Status    |
|------------------------|-------------|--------|-----------|
| `targeted`             | `False`     | `false` | ✅ OK    |
| `confidence` / `kappa` | `0`         | `0.0`  | ✅ OK     |
| `max_iterations`       | `1000`      | `1000` | ✅ OK     |
| `batch_size`           | `1`         | `1`    | ✅ OK     |
| `learning_rate`        | `0.01`      | `0.01` | ✅ OK     |
| `binary_search_steps`  | `9`         | `9`    | ✅ OK     |
| `initial_const`        | `1e-3`      | `1e-3` | ✅ OK     |
| `clip_min`             | `-0.5`      | `-0.5` | ✅ OK     |
| `clip_max`             | `0.5`       | `0.5`  | ✅ OK     |
| `abort_early`          | implícito   | `true` | ⚠️ VERIFICAR |

### Configuração YAML Completa do Ataque

```yaml
attack:
  name: cw_l2
  kappa: 0.0
  batch_size: 1
  max_iterations: 1000
  learning_rate: 0.01
  binary_search_steps: 9
  initial_const: 0.001
  abort_early: true
  targeted: false
  clip_min: -0.5
  clip_max: 0.5
```

### Critério de Aceitação

- Todos os 10 hiperparâmetros listados acima estão presentes no YAML
- Nenhum parâmetro usa valor default silencioso diferente do original

---

## F-4 — Eager Execution no TensorFlow 2 (BLOQUEANTE)

### Problema

O TensorFlow 2 ativa **eager execution por padrão**. O wrapper `InceptionV3TensorFlowWrapper` usa `tf.compat.v1.placeholder`, `tf.compat.v1.Session` e `sess.run` — API de grafo estático do TF1. Sem desabilitar eager, essas chamadas falham silenciosamente ou lançam erros em tempo de execução.

> ⚠️ O script de validação básico pode passar parcialmente sem essa correção, mas o experimento real falha ao criar `placeholder`, importar o grafo ou executar `sess.run`.

### Arquivo Afetado

`src/deepdetector/models/imagenet_wrappers.py` — método `InceptionV3TensorFlowWrapper.__init__`

### Mudança

```python
# ANTES
def __init__(self, ...):
    import tensorflow as tf
    self.tf = tf
    self.image_size = 299
    # ... resto do __init__

# DEPOIS — adicionar disable_eager logo após atribuir self.tf
def __init__(self, ...):
    import tensorflow as tf
    self.tf = tf
    self.tf.compat.v1.disable_eager_execution()   # <-- LINHA ADICIONADA
    self.image_size = 299
    # ... resto do __init__
```

### Por que Preserva a Semântica

O código original de Inception e CW-L2 sempre operou em **modo grafo/sessão do TF1**, usando `tf.Session`, `tf.placeholder` e `tf.import_graph_def`. Desabilitar eager mantém exatamente esse comportamento no ambiente TF2.

### Critério de Aceitação

- `tf.compat.v1.disable_eager_execution()` é chamado antes de qualquer operação de grafo
- Criação de `placeholder` e execução de `sess.run` não levantam exceção
- O wrapper carrega o grafo congelado sem erros em ambiente TF2

---

## F-5 — Validação do `input_map_name` (CRÍTICA)

### Problema

O script original mapeia a imagem escalada para `Cast:0` ao importar o grafo Inception. A branch atual usa `ResizeBilinear:0`. Dependendo do arquivo `.pb` usado, um dos dois pode estar errado. Uma entrada mapeada incorretamente resulta em **predições completamente erradas** mesmo que o resto do pipeline esteja correto.

### Comportamento de Escala por `input_map_name`

| `input_map_name`   | Conversão aplicada            | Status             |
|--------------------|-------------------------------|--------------------|
| `Cast:0`           | `(tensor + 0.5) × 255`        | Original           |
| `ResizeBilinear:0` | `(tensor + 0.5) × 255`        | Atual — verificar  |
| `Sub:0`            | `(tensor + 0.5) × 255 − 128`  | Alternativo        |
| `Mul:0`            | `tensor × 2`                  | Alternativo        |

### Script de Validação

Executar antes de rodar qualquer experimento completo:

```python
import numpy as np

# 1. Carregar uma imagem de classe conhecida (ex: zebra, label = 80)
image = load_image('test_zebra.jpg')  # resultado em [-0.5, 0.5]

# 2. Verificar range antes da predição
assert image.min() >= -0.5, f'min={image.min()}'
assert image.max() <= 0.5,  f'max={image.max()}'

# 3. Predição
pred_class = wrapper.predict(image)

# 4. Comparar com expected
assert pred_class == 80, f'Esperado 80 (zebra), obtido {pred_class}'
print('input_map_name OK')
```

Se a predição não bater, alterar no YAML:

```yaml
input_map_name: "Cast:0"
```

### Critério de Aceitação

- Script de validação acima passa sem assertions
- Classe predita para imagens conhecidas bate com `modified_setup_inception.py`

---

## Checklist de Implementação

Executar nesta ordem:

- [ ] **1. F-4** — Adicionar `disable_eager_execution()` em `imagenet_wrappers.py` → verificar: `sess.run` sem exceção
- [ ] **2. F-5** — Rodar script de validação do `input_map_name` com zebra conhecida → verificar: `pred == 80`
- [ ] **3. F-1** — Adicionar `class_quotas` no YAML e cortar por classe no loader → verificar: `len(zebra)==40, panda==40, cab==20`
- [ ] **4. F-2** — Adicionar `class_order: [zebra, panda, cab]` no YAML → verificar: índices 0–39 são zebras
- [ ] **5. F-3** — Conferir todos os 10 hiperparâmetros do CW-L2 no YAML → verificar: tabela de parâmetros OK
- [ ] **6. —** Smoke test: rodar experimento completo para 5 imagens → verificar: TP+FN+FP reportados sem erros
- [ ] **7. —** Rodar experimento completo para 100 imagens → verificar: métricas próximas às da Tabela 10

---

## Decisões de Design e Limitações

**CleverHans vs. `nn_robust_attacks`:** Aceitável para reprodução de métricas agregadas (TP, FP, FN, TTP). Não é aceitável se o objetivo for comparar adversariais pixel a pixel.

**TensorFlow 2 com API TF1:** Usar `tf.compat.v1` com eager desabilitado mantém compatibilidade com o grafo congelado `.pb` original. Migrar para TF2 nativo exigiria reescrever o carregamento do grafo com `tf.saved_model` ou converter o `.pb` para SavedModel — fora do escopo desta refatoração.

**Predição por logits vs. softmax:** `argmax(logits)` e `argmax(softmax(logits))` retornam exatamente a mesma classe top-1. A diferença só importa se for necessário reportar probabilidade, o que não é exigido pelos contadores TP/FP/FN/TTP.
