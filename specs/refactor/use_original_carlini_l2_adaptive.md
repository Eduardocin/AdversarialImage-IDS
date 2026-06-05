# Plano de mudanças para usar o `CarliniL2Adaptive` original

## Contexto

O experimento `defense_aware` da branch `fix/inceptionv3-ampere` já está integrado ao `configs/experiments.yaml` e ao runner central. Ele compara dois cenários:

- `defense_unaware`: CW-L2 normal usando `nn_robust_attacks`.
- `defense_aware`: ataque adaptativo contra o filtro final do DeepDetector.

A implementação atual de `adaptive_cw_l2` é conceitualmente correta, mas não é uma reprodução exata do repositório base. Hoje o fluxo gera candidatos com CW-L2 normal e depois rejeita/aceita os exemplos conforme passam ou não pela defesa.

No código base original, a lógica adaptativa fica dentro do próprio processo de otimização do Carlini-Wagner L2. O melhor exemplo adversarial só é atualizado quando a imagem adversarial também passa pela transformação/filtro sem alterar a predição.

Condição original essencial:

```python
if self.mnistPredicate(ii) == self.transformed(ii):
    o_bestattack[e] = ii
```

Portanto, o objetivo deste plano é substituir a aproximação atual por uma integração direta com o `CarliniL2Adaptive` original.

### Verificação local

O checkout local do backend original já existe em:

```text
nn_robust_attacks/
```

Esse diretório é um checkout Git de `https://github.com/carlini/nn_robust_attacks.git` e já contém os backends usados hoje pelo projeto:

```text
nn_robust_attacks/l2_attack.py
nn_robust_attacks/li_attack.py
nn_robust_attacks/l0_attack.py
nn_robust_attacks/l2_adaptive_attack.py
```

Portanto, a implementação não deve criar uma nova cópia vendorizada em `third_party/` nem em `src/deepdetector/external/`. O caminho local `nn_robust_attacks` deve continuar sendo a raiz configurável do backend externo, como já ocorre em `configs/experiments.yaml`.

Verificação adicional: o checkout local atual contém `nn_robust_attacks/l2_adaptive_attack.py` e expõe a classe `CarliniL2Adaptive`. Mesmo assim, o loader deve falhar com erro claro se esse arquivo ou classe não estiver disponível em outro ambiente.

---

## Objetivo

Usar o `CarliniL2Adaptive` original do repositório base para o experimento `defense_aware`, mantendo a organização atual do projeto:

- configuração via `configs/experiments.yaml`;
- execução via runner central;
- saída oficial apenas em `metrics.csv` e `metrics.json`;
- compatibilidade com o modelo MNIST M2 atual;
- comparação entre ataque `defense_unaware` e ataque `defense_aware`.

---

## Estado atual da implementação

### Arquivos principais existentes

- `configs/experiments.yaml`
- `src/deepdetector/experiments/runner.py`
- `src/deepdetector/experiments/defense_aware.py`
- `src/deepdetector/evaluation/defense_aware.py`
- `src/deepdetector/attacks/adaptive_cw_l2.py`
- `src/deepdetector/attacks/nn_robust.py`
- `src/deepdetector/filters/adaptive_noise_reduction.py`
- `tests/test_defense_aware.py`

### Problema principal

O arquivo `src/deepdetector/attacks/adaptive_cw_l2.py` atualmente implementa uma estratégia de pós-validação:

1. gera adversariais com CW-L2 normal;
2. aplica o filtro final;
3. aceita apenas os candidatos em que:
   - `C(x_adv) != y_true`;
   - `C(x_adv) == C(T(x_adv))`.

Isso mede evasão da defesa, mas não força o otimizador do CW-L2 a procurar exemplos já adaptados à defesa durante a busca.

---

## Mudança 1 — Usar o checkout local `nn_robust_attacks`

### Proposta

Usar o backend externo local já existente:

```text
nn_robust_attacks/
```

O arquivo adaptativo original esperado deve ficar em:

```text
nn_robust_attacks/l2_adaptive_attack.py
```

### Recomendação

Manter `nn_robust_attacks` fora do pacote `src/deepdetector`, como checkout externo local. O projeto já usa esse padrão para `CarliniL2` e `CarliniLi`, carregando os arquivos diretamente a partir de `nn_robust_attacks_root`.

### Observações

O arquivo `l2_adaptive_attack.py`, quando adicionado ao checkout local, deve preservar o comportamento original, mas pode receber mudanças mínimas para compatibilidade com TensorFlow 1.x via `tf.compat.v1`, caso necessário.

Evitar reescrever a lógica do ataque neste primeiro momento. A prioridade é fidelidade ao base.

### Erro esperado

Se `nn_robust_attacks/l2_adaptive_attack.py` não existir, ou se o arquivo não expor `CarliniL2Adaptive`, a execução deve falhar com uma mensagem explícita, por exemplo:

```text
Missing nn_robust_attacks l2_adaptive_attack.py: nn_robust_attacks/l2_adaptive_attack.py
```

---

## Mudança 2 — Criar loader para `CarliniL2Adaptive`

### Arquivo a alterar

```text
src/deepdetector/attacks/nn_robust.py
```

### Adicionar função interna

```python
def _load_nn_robust_carlini_l2_adaptive(root: str) -> Any:
    return _load_nn_robust_class(root, "l2_adaptive_attack.py", "CarliniL2Adaptive")
```

### Justificativa

O arquivo `nn_robust.py` já possui loaders para:

- `CarliniL2`, vindo de `l2_attack.py`;
- `CarliniLi`, vindo de `li_attack.py`.

Devemos seguir o mesmo padrão para `CarliniL2Adaptive`.

---

## Mudança 3 — Criar wrapper real para o ataque adaptativo original

### Arquivo recomendado

Substituir ou complementar:

```text
src/deepdetector/attacks/adaptive_cw_l2.py
```

com uma função nova:

```python
def generate_original_adaptive_cw_l2_attack(...):
    ...
```

### Comportamento esperado

A função deve:

1. receber imagens em `[0, 1]`, padrão interno do projeto;
2. converter para `[-0.5, 0.5]` antes do ataque;
3. instanciar `CarliniL2Adaptive` original;
4. executar `attack.attack(centered_images, one_hot_labels)`;
5. converter a saída de volta para `[0, 1]`;
6. retornar `np.float32` com shape igual ao batch de entrada.

### Pseudocódigo

```python
def generate_original_adaptive_cw_l2_attack(
    model,
    images,
    labels,
    *,
    nn_robust_attacks_root,
    confidence=0.0,
    batch_size=1,
    max_iterations=2000,
    learning_rate=0.1,
    binary_search_steps=5,
    initial_const=1.0,
    abort_early=True,
    targeted=False,
    clip_min=0.0,
    clip_max=1.0,
    **kwargs,
):
    image_array = np.asarray(images, dtype=np.float32)
    centered_images = image_array - 0.5

    adapter = _adapter_for_images(model, centered_images)
    one_hot_labels = _one_hot(np.asarray(labels), adapter.num_labels)
    session = _model_session(model)
    CarliniL2Adaptive = _load_nn_robust_carlini_l2_adaptive(nn_robust_attacks_root)

    with _session_graph_context(session):
        attack = CarliniL2Adaptive(
            session,
            adapter,
            batch_size=batch_size,
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            binary_search_steps=binary_search_steps,
            max_iterations=max_iterations,
            abort_early=abort_early,
            initial_const=initial_const,
            boxmin=-0.5,
            boxmax=0.5,
        )
        centered_adv = attack.attack(centered_images, one_hot_labels)

    return np.clip(centered_adv + 0.5, clip_min, clip_max).astype(np.float32)
```

---

## Mudança 4 — Garantir que o adapter do M2 é compatível com entrada centralizada

### Arquivo a revisar

```text
src/deepdetector/attacks/nn_robust.py
```

### Problema

O `CarliniL2Adaptive` original espera imagens em `[-0.5, 0.5]`, mas o modelo M2 do projeto provavelmente foi treinado/avaliado em `[0, 1]`.

### Solução

O adapter usado pelo ataque precisa somar `+0.5` antes de chamar o modelo, como já era feito no script antigo de Table 10 M2:

```python
class M2NnRobustAdapter(object):
    def predict(self, data):
        return self.model(data + 0.5)
```

### Ação recomendada

Criar um parâmetro no adapter genérico:

```python
input_shift: float = 0.0
```

Exemplo:

```python
class NnRobustModelAdapter(object):
    def __init__(..., input_shift: float = 0.0):
        self.input_shift = float(input_shift)

    def predict(self, data):
        shifted = data + self.input_shift
        ...
```

Para MNIST M2 com CW-L2 original:

```python
adapter = NnRobustModelAdapter(
    model,
    image_shape=(28, 28, 1),
    num_labels=10,
    input_shift=0.5,
)
```

---

## Mudança 5 — Ajustar o YAML para diferenciar faixa externa e faixa interna do ataque

### Arquivo

```text
configs/experiments.yaml
```

### Estado atual

```yaml
clip_min: 0.0
clip_max: 1.0
```

### Problema

Esses limites estão corretos para os dados externos do projeto, mas o `CarliniL2Adaptive` original opera internamente em `[-0.5, 0.5]`.

### Proposta

Adicionar campos explícitos:

```yaml
attacks:
  defense_aware:
    type: original_adaptive_cw_l2
    nn_robust_attacks_root: nn_robust_attacks
    targeted: false
    confidence: 0
    max_iterations: 2000
    binary_search_steps: 5
    initial_const: 1.0
    learning_rate: 0.1
    abort_early: true
    input_range:
      min: 0.0
      max: 1.0
    attack_box:
      min: -0.5
      max: 0.5
    model_input_shift: 0.5
```

### Justificativa

Isso evita ambiguidade entre:

- faixa dos dados carregados e salvos: `[0, 1]`;
- faixa usada pelo ataque Carlini original: `[-0.5, 0.5]`;
- compensação aplicada antes de chamar o modelo: `+0.5`.

---

## Mudança 6 — Alterar o evaluator para usar o novo wrapper

### Arquivo

```text
src/deepdetector/evaluation/defense_aware.py
```

### Estado atual

A função `_adaptive_attack_fn` chama:

```python
generate_adaptive_cw_l2_attack(...)
```

### Proposta

Trocar para:

```python
generate_original_adaptive_cw_l2_attack(...)
```

quando o YAML declarar:

```yaml
type: original_adaptive_cw_l2
```

### Estratégia recomendada

Não remover a implementação atual imediatamente. Manter as duas opções:

- `adaptive_cw_l2`: implementação atual por pós-validação;
- `original_adaptive_cw_l2`: reprodução fiel com `CarliniL2Adaptive` original.

Isso permite comparar os dois comportamentos.

---

## Mudança 7 — Registrar o novo ataque no registry

### Arquivo provável

```text
src/deepdetector/attacks/registry.py
```

### Adicionar

```python
"original_adaptive_cw_l2": generate_original_adaptive_cw_l2_attack
```

### Justificativa

O teste atual já verifica se `adaptive_cw_l2` está registrado. Devemos adicionar um teste equivalente para o novo wrapper original.

---

## Mudança 8 — Ajustar o filtro para empate fiel ao código base

### Arquivo

```text
src/deepdetector/filters/adaptive_noise_reduction.py
```

### Estado atual

```python
use_quantized = np.abs(quantized - image_array) <= np.abs(smoothed - image_array)
```

### Código base

O `chooseCloserFilter` original escolhe o primeiro filtro apenas quando `a < b`. Em caso de empate, escolhe o segundo filtro.

### Proposta

Trocar `<=` por `<`:

```python
use_quantized = np.abs(quantized - image_array) < np.abs(smoothed - image_array)
```

### Impacto

Pequeno, mas melhora a fidelidade ao repositório base.

---

## Mudança 9 — Revisar o filtro `cross_mean`

### Arquivos

```text
src/deepdetector/filters/adaptive_noise_reduction.py
src/deepdetector/filters/mean_filters.py
```

### Ponto de atenção

No código base, o filtro usado no MNIST percorre pixels internos com:

```python
crossMeanFilterOperations(inputAfterQ, 3, 25, 13)
```

Isso significa:

- raio efetivo: `3`;
- região processada: linhas/colunas `3..24`;
- coeficiente: `13`, isto é, centro + 4 direções * 3 passos.

### Ação

Confirmar que `cross_mean_filter(..., radius=3)` preserva exatamente esse comportamento:

- não deve suavizar bordas fora da região válida;
- deve usar o centro mais vizinhos verticais/horizontais até distância 3;
- deve dividir por 13.

Se a implementação atual tratar bordas com padding/reflexão/convolução genérica, ela pode divergir do base.

---

## Mudança 10 — Testes necessários

### Testes unitários

Adicionar ou ajustar em:

```text
tests/test_defense_aware.py
```

#### 1. Teste de config

Validar que o YAML aceita:

```yaml
type: original_adaptive_cw_l2
attack_box:
  min: -0.5
  max: 0.5
model_input_shift: 0.5
```

#### 2. Teste de registry

```python
def test_original_adaptive_cw_l2_is_registered():
    assert "original_adaptive_cw_l2" in ATTACK_REGISTRY
```

#### 3. Teste de conversão de faixa

Verificar que:

- entrada `[0, 1]` vira `[-0.5, 0.5]` antes do ataque;
- saída `[-0.5, 0.5]` volta para `[0, 1]`.

#### 4. Teste de adapter

Garantir que o modelo recebe `data + 0.5` quando o ataque trabalha centralizado.

#### 5. Teste do empate no filtro

Criar caso sintético em que `abs(quantized - original) == abs(smoothed - original)` e validar que o filtro escolhe `smoothed`, como no base.

#### 6. Teste de dispatch

Garantir que `_adaptive_attack_fn` chama o wrapper original quando `type == original_adaptive_cw_l2`.

---

## Mudança 11 — Saídas e rastreabilidade

### Saídas oficiais

Manter apenas:

```text
results/experiments/defense_aware/metrics.csv
results/experiments/defense_aware/metrics.json
```

### Metadados recomendados no JSON

Atualmente o `metrics.json` contém apenas métricas. Para reprodução, considerar adicionar um arquivo separado opcional:

```text
results/experiments/defense_aware/manifest.json
```

com:

- commit hash;
- tipo de ataque usado;
- caminho do `l2_adaptive_attack.py`;
- parâmetros do ataque;
- intervalo do dataset;
- checkpoint do M2;
- data/hora de execução.

Se o requisito for manter somente CSV e JSON, o `manifest` pode ser incorporado em `metrics.json` sob uma chave `_metadata`. Porém, os testes atuais indicam que não deve haver metadata. Nesse caso, manter o manifesto fora do experimento oficial ou documentar os parâmetros no YAML.

---

## Mudança 12 — Documentação de execução

Adicionar no README ou em `specs/features` um comando padrão:

```bash
python -m deepdetector.cli.run_experiment defense_aware --config configs/experiments.yaml
```

ou o comando real usado pelo projeto.

Também documentar a necessidade de ter o diretório:

```text
nn_robust_attacks/l2_adaptive_attack.py
```

ou configurar:

```yaml
nn_robust_attacks_root: /caminho/para/nn_robust_attacks
```

---

## Plano de implementação sugerido

### Etapa 1 — Preparar o código externo

- [ ] Validar que o checkout local `nn_robust_attacks/` existe.
- [ ] Adicionar/restaurar `nn_robust_attacks/l2_adaptive_attack.py` no checkout local, se ele ainda não existir.
- [ ] Não criar cópia vendorizada em `third_party/` ou `src/deepdetector/external/`.
- [ ] Validar imports TensorFlow 1.x com `patch_tensorflow_v1_symbols()`.
- [ ] Garantir que a classe `CarliniL2Adaptive` é carregável.

### Etapa 2 — Criar wrapper original

- [ ] Adicionar loader `_load_nn_robust_carlini_l2_adaptive`.
- [ ] Criar `generate_original_adaptive_cw_l2_attack`.
- [ ] Garantir conversão `[0, 1] -> [-0.5, 0.5] -> [0, 1]`.
- [ ] Adicionar `model_input_shift=0.5` no adapter.

### Etapa 3 — Integrar ao experimento

- [ ] Atualizar `configs/experiments.yaml`.
- [ ] Atualizar `_adaptive_attack_fn` em `evaluation/defense_aware.py`.
- [ ] Registrar `original_adaptive_cw_l2` no registry.

### Etapa 4 — Ajustar fidelidade do filtro

- [ ] Trocar `<=` por `<` no critério de escolha entre quantizado e suavizado.
- [ ] Validar comportamento do `cross_mean_filter` com `radius=3` e coeficiente 13.

### Etapa 5 — Testar

- [ ] Atualizar testes unitários.
- [ ] Rodar testes rápidos do defense-aware.
- [ ] Rodar uma amostra pequena, por exemplo `start=9000`, `end=9010`.
- [ ] Rodar experimento completo `9000:10000`.

---

## Critérios de aceite

A mudança será considerada concluída quando:

- [ ] o experimento `defense_aware` usar `CarliniL2Adaptive` original durante a otimização;
- [ ] o backend adaptativo ser carregado a partir de `nn_robust_attacks/l2_adaptive_attack.py`;
- [ ] a execução falhar com erro claro se `l2_adaptive_attack.py` ou `CarliniL2Adaptive` não estiver disponível no checkout local;
- [ ] a configuração do YAML explicitar entrada externa `[0,1]` e ataque interno `[-0.5,0.5]`;
- [ ] o adapter aplicar `+0.5` antes de chamar o M2;
- [ ] o filtro final reproduzir a regra de empate do base;
- [ ] os testes unitários passarem;
- [ ] o experimento gerar apenas `metrics.csv` e `metrics.json` como saídas oficiais;
- [ ] os resultados permitirem comparar `defense_unaware` vs `defense_aware` no mesmo intervalo `MNIST test[9000:10000]`.

---

## Risco principal

O maior risco é incompatibilidade entre o `CarliniL2Adaptive` original e a abstração atual do modelo M2. O original espera um objeto com:

- `image_size`;
- `num_channels`;
- `num_labels`;
- método `predict(newimg)` retornando logits;
- em alguns pontos, acesso a `model.model.predict(...)` dentro de `mnistPredicate`.

Portanto, talvez seja necessário criar um adapter específico para o `CarliniL2Adaptive`, não apenas reutilizar o `NnRobustModelAdapter` genérico.

### Adapter específico recomendado

```python
class M2CarliniAdaptiveAdapter(object):
    image_size = 28
    num_channels = 1
    num_labels = 10

    def __init__(self, keras_model, sess, input_tensor, logits_tensor):
        self.model = keras_model
        self.sess = sess
        self.input_tensor = input_tensor
        self.logits_tensor = logits_tensor

    def predict(self, centered_data):
        return self.model(centered_data + 0.5)
```

Se o método original `mnistPredicate` exigir `self.tempmodel.model.predict`, o adapter precisa expor `.model.predict(...)` de forma compatível ou o `l2_adaptive_attack.py` precisará de uma adaptação mínima para usar `self.tempmodel.predict(...)` também na predição auxiliar.

---

## Decisão recomendada

Implementar como novo tipo de ataque:

```yaml
type: original_adaptive_cw_l2
```

em vez de substituir imediatamente:

```yaml
type: adaptive_cw_l2
```

Assim, preservamos a implementação atual para comparação e adicionamos uma trilha fiel ao repositório base.
