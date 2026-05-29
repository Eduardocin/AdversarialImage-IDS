````markdown
# Spec — Implementar DeepFool para Table 10 / ImageNet / GoogLeNet

## 1. Objetivo

Implementar o suporte ao ataque **DeepFool** necessário para a linha 7 da **Table 10**:

| No. | Attack/Model | Dataset |
|---:|---|---|
| 7 | `DeepFool/GoogLeNet` | ImageNet |

A implementação deve se alinhar ao fluxo atual já criado para a Table 10, isto é:

```text
table_10_googlenet
├── No. 5  FGSM (ε=1/255)/GoogLeNet
├── No. 6  FGSM (ε=2/255)/GoogLeNet
└── No. 7  DeepFool/GoogLeNet
````

A tarefa deve **remover a condição de bloqueio da linha 7** quando o DeepFool estiver funcional, sem criar uma nova estrutura paralela dentro de `evaluation/`.


O código base utiliza o deepfool definido em  https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py
verifique se podemos adpatar a nossa implementação .

---

## 2. Princípio da implementação

A implementação do DeepFool deve entrar como um **ataque reutilizável**, não como uma lógica específica da Table 10.

Portanto:

```text
DeepFool pertence ao módulo de ataques.
Table 10 apenas consome o ataque configurado.
```

O runner da Table 10 não deve saber detalhes internos do algoritmo DeepFool. Ele deve apenas ler:

```yaml
attack:
  name: deepfool
```

e despachar para o mecanismo de ataques já existente.

---

## 3. Escopo

Esta spec cobre:

* criar ou integrar o ataque `deepfool`;
* registrar o ataque no fluxo atual de execução;
* permitir que a linha 7 de `table_10_googlenet` deixe de ser `blocked`;
* manter o output da Table 10 no padrão atual;
* reaproveitar o runner `table_10_group`;
* reaproveitar funções já existentes de métricas, outputs, filtros, dataset e modelo;
* não criar diretórios novos dentro de `evaluation/`;
* não criar runner específico para DeepFool;
* não criar runner específico para GoogLeNet.

---

## 4. Fora do escopo

Não implementar nesta tarefa:

* CaffeNet;
* Inception v3;
* linhas 8, 14–18 e 20;
* agregado geral da Table 10;
* novo diretório em `evaluation/`;
* novo script em `scripts/`;
* `manifest.json`;
* relatórios `.md`;
* diagnóstico extra;
* comparação automática com números do artigo;
* uso de resultados antigos.

---

## 5. Estrutura esperada

A implementação deve se encaixar na estrutura atual do projeto.

Se já existe um módulo de ataques, usar ele.

Estrutura esperada conceitualmente:

```text
src/deepdetector/
├── attacks/
│   ├── fgsm.py
│   ├── deepfool.py
│   └── registry.py        # se já existir padrão de registry
│
├── evaluation/
│   ├── metrics.py
│   ├── outputs.py
│   └── ...                # manter padrão atual, sem criar pasta nova
│
└── ...
```

Não criar:

```text
src/deepdetector/evaluation/tables/deepfool.py
src/deepdetector/evaluation/deepfool/
src/deepdetector/evaluation/table_10_deepfool.py
scripts/deepfool_googlenet.py
scripts/table_10_deepfool.py
```

---

## 6. Configuração atual da Table 10

O grupo `table_10_googlenet` deve continuar sendo o ponto de entrada da linha 7.

Configuração esperada antes da implementação completa:

```yaml
table_10_googlenet:
  kind: table_10_group
  output_dir: results/experiments/table_10/imagenet/googlenet
  dataset:
    name: imagenet
  model:
    name: googlenet
  model_group: googlenet
  dataset_label: ImageNet
  rows:
    - "no": 5
      attack_model: "FGSM (ε=1/255)/GoogLeNet"
      status: planned
      attack:
        name: fgsm
        epsilon: 0.00392156862745098

    - "no": 6
      attack_model: "FGSM (ε=2/255)/GoogLeNet"
      status: planned
      attack:
        name: fgsm
        epsilon: 0.00784313725490196

    - "no": 7
      attack_model: "DeepFool/GoogLeNet"
      status: blocked
      blocked_reason: "DeepFool ainda não está implementado no projeto."
      attack:
        name: deepfool
```

Após a implementação do DeepFool, a linha 7 deve ser alterada para:

```yaml
    - "no": 7
      attack_model: "DeepFool/GoogLeNet"
      status: implemented
      attack:
        name: deepfool
        max_iter: 50
        overshoot: 0.02
        clip_min: 0.0
        clip_max: 1.0
```

Se o projeto preferir ainda não considerar o resultado final como reprodutível, usar:

```yaml
status: planned
```

mas **não manter `blocked`** depois que o ataque estiver integrado.

---

## 7. Contrato do ataque DeepFool

O ataque deve expor uma interface compatível com o mecanismo atual de ataques.

Caso ainda não exista padrão formal, usar uma função com contrato semelhante:

```python
def generate_deepfool(
    model,
    images,
    labels=None,
    *,
    max_iter: int = 50,
    overshoot: float = 0.02,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
):
    """
    Gera exemplos adversariais com DeepFool.

    Args:
        model: modelo diferenciável usado para classificação.
        images: batch de imagens limpas.
        labels: labels verdadeiros ou labels preditos, conforme padrão do projeto.
        max_iter: número máximo de iterações por imagem.
        overshoot: fator de extrapolação da perturbação.
        clip_min: limite inferior dos pixels.
        clip_max: limite superior dos pixels.

    Returns:
        batch de imagens adversariais no mesmo formato esperado pelo pipeline.
    """
```

A assinatura final deve respeitar o padrão já existente para `fgsm`, se houver.

Exemplo: se o FGSM atual usa uma classe, DeepFool também deve usar classe. Se usa função, DeepFool deve usar função.

---

## 8. Regras de implementação do DeepFool

A implementação deve:

* funcionar com o modelo GoogLeNet já usado no projeto;
* preservar o formato de entrada e saída usado pelos ataques existentes;
* respeitar `clip_min` e `clip_max`;
* retornar imagens adversariais no mesmo domínio esperado pelo detector;
* permitir execução por batch, ou pelo menos por imagem com wrapper de batch;
* não alterar o modelo em modo treino;
* não salvar imagens intermediárias como output oficial;
* não escrever arquivos diretamente;
* não calcular métricas dentro do ataque;
* não aplicar filtro de detecção dentro do ataque.

O ataque deve apenas gerar `x_adv`.

---

## 9. Integração com registry ou dispatcher de ataques

Se o projeto já possui algum dispatcher de ataques, por exemplo:

```python
generate_attack(config, model, images, labels)
```

ou registry equivalente, adicionar `deepfool` ali.

Exemplo conceitual:

```python
if attack_name == "fgsm":
    return generate_fgsm(...)

if attack_name == "deepfool":
    return generate_deepfool(...)
```

Preferencialmente, usar padrão de registry:

```python
ATTACK_REGISTRY = {
    "fgsm": generate_fgsm,
    "deepfool": generate_deepfool,
}
```

A Table 10 não deve chamar `generate_deepfool` diretamente se já existir dispatcher comum.

---

## 10. Integração com `table_10_group`

O runner do grupo `table_10_googlenet` deve continuar genérico.

Fluxo esperado:

```text
table_10_googlenet
    ↓
kind: table_10_group
    ↓
runner atual da Table 10
    ↓
linha 7: attack.name = deepfool
    ↓
dispatcher de ataques
    ↓
generate_deepfool(...)
    ↓
detector/filtro já existente
    ↓
métricas já existentes
    ↓
metrics.csv e metrics.json
```

Não criar fluxo alternativo:

```text
table_10_googlenet_deepfool.py
```

---

## 11. Métricas

A linha 7 deve preencher os mesmos campos oficiais da Table 10:

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
```

O cálculo das métricas deve reutilizar o módulo comum já usado pelas demais tabelas.

Não recalcular manualmente dentro do runner de DeepFool.

Campos esperados:

| Campo          | Descrição                                                 |
| -------------- | --------------------------------------------------------- |
| `num_failures` | número de falhas do ataque, correspondente a `#F`         |
| `tp`           | adversariais detectados                                   |
| `fn`           | adversariais não detectados                               |
| `fp`           | benignos classificados como adversariais                  |
| `rtp`          | adversariais detectados e recuperados para classe correta |
| `rtp_percent`  | `RTP / TP`                                                |
| `recall`       | `TP / (TP + FN)`                                          |
| `precision`    | `TP / (TP + FP)`                                          |
| `f1`           | F1-score                                                  |

---

## 12. Definição de falha do ataque

Para a Table 10, `#F` deve representar casos em que o ataque falhou em alterar a decisão do modelo.

Para cada imagem válida:

```text
clean_pred = C(x)
adv_pred = C(x_adv)

se adv_pred == clean_pred:
    num_failures += 1
```

Essas amostras não entram como `TP` ou `FN`, pois não formaram adversarial efetivo.

O mesmo critério deve ser usado para DeepFool e FGSM, se já for o padrão do projeto.

---

## 13. Detector após DeepFool

Depois de gerar `x_adv` com DeepFool, o fluxo deve usar o detector já existente:

```text
detected = C(T(x_adv)) != C(x_adv)
```

Para `RTP`:

```text
recovered = C(T(x_adv)) == y_true
```

Para `FP`:

```text
false_positive = C(T(x_clean)) != C(x_clean)
```

Não duplicar essa lógica se ela já existir em `filters/detector.py`, `evaluation/metrics.py` ou equivalente.

---

## 14. Saída esperada

O comando continua sendo:

```bash
python scripts/run_experiment.py --experiment table_10_googlenet
```

A saída continua sendo:

```text
results/experiments/table_10/imagenet/googlenet/
├── metrics.csv
└── metrics.json
```

Sem:

```text
manifest.json
```

---

## 15. Comportamento antes e depois

### Antes da implementação do DeepFool

A linha 7 aparece sem métricas:

```csv
7,DeepFool/GoogLeNet,ImageNet,,,,,,,,,
```

### Depois da implementação do DeepFool

A linha 7 deve preencher métricas computadas:

```csv
7,DeepFool/GoogLeNet,ImageNet,<#F>,<TP>,<FN>,<FP>,<RTP>,<RTP%>,<Recall>,<Precision>,<F1>
```

As linhas 5 e 6 podem continuar `planned` se o fluxo FGSM ImageNet ainda não estiver conectado.

---

## 16. Tratamento de dependências

Se DeepFool depender de biblioteca externa, a dependência deve ser explícita.

Opções aceitáveis:

1. implementação própria com PyTorch/autograd;
2. integração com biblioteca já usada no projeto;
3. wrapper controlado para biblioteca externa.

Não aceitar:

* import implícito sem declarar dependência;
* download automático de código externo em tempo de execução;
* referência quebrada como `about:blank`;
* script copiado sem adaptação ao padrão do projeto.

Se a referência do artigo apontar para uma URL quebrada ou aparecer como `about:blank`, isso deve ser tratado como **informação insuficiente para integração direta**. A implementação deve seguir uma fonte clara ou uma implementação própria documentada.

---

## 17. Tratamento de modelos não diferenciáveis

DeepFool precisa de gradiente.

Antes de executar linha 7, validar se o wrapper do GoogLeNet expõe:

* forward diferenciável;
* logits ou scores antes de softmax;
* gradiente em relação à imagem de entrada.

Se o GoogLeNet atual for wrapper Caffe não diferenciável no ambiente atual, a linha 7 deve permanecer bloqueada com motivo explícito:

```yaml
status: blocked
blocked_reason: "DeepFool requer gradiente, mas o wrapper atual do GoogLeNet não expõe backward diferenciável."
```

Se houver wrapper PyTorch/TensorFlow diferenciável, usar esse wrapper.

---

## 18. Configuração de DeepFool

Adicionar parâmetros explícitos na linha 7:

```yaml
attack:
  name: deepfool
  max_iter: 50
  overshoot: 0.02
  clip_min: 0.0
  clip_max: 1.0
```

Esses parâmetros devem vir da config, não hardcoded dentro do runner.

---

## 19. Testes unitários

### 19.1. Teste de registry

Verificar que o ataque está registrado:

```python
def test_deepfool_attack_is_registered():
    assert "deepfool" in ATTACK_REGISTRY
```

### 19.2. Teste de contrato de saída

Com modelo dummy diferenciável:

```python
def test_deepfool_returns_same_shape_as_input():
    x_adv = generate_deepfool(
        model=dummy_model,
        images=images,
        labels=labels,
        max_iter=3,
        overshoot=0.02,
        clip_min=0.0,
        clip_max=1.0,
    )

    assert x_adv.shape == images.shape
```

### 19.3. Teste de limites

```python
def test_deepfool_respects_clip_bounds():
    x_adv = generate_deepfool(...)

    assert x_adv.min() >= 0.0
    assert x_adv.max() <= 1.0
```

### 19.4. Teste de integração com config

```python
def test_table_10_googlenet_deepfool_row_has_config():
    row = get_table_10_row("table_10_googlenet", no=7)

    assert row["attack"]["name"] == "deepfool"
    assert "max_iter" in row["attack"]
    assert "overshoot" in row["attack"]
```

### 19.5. Teste de bloqueio se modelo não expõe gradiente

```python
def test_deepfool_requires_differentiable_model():
    with pytest.raises(NotImplementedError, match="requires gradient"):
        generate_deepfool(model=non_differentiable_model, images=images)
```

---

## 20. Testes de integração

### 20.1. Execução controlada com amostra pequena

Adicionar opção de avaliação pequena na config ou fixture de teste:

```yaml
evaluation:
  n_samples: 2
```

Teste esperado:

```python
def test_table_10_googlenet_deepfool_smoke_run():
    result = run_experiment("table_10_googlenet", override={"evaluation": {"n_samples": 2}})

    assert "metrics.csv" in result.outputs
    assert "metrics.json" in result.outputs
```

Esse teste não precisa validar os valores finais do artigo. Ele valida apenas que o fluxo roda.

---

## 21. Atualização do status da linha 7

Quando DeepFool estiver implementado e integrado, alterar:

```yaml
status: blocked
blocked_reason: "DeepFool ainda não está implementado no projeto."
```

para:

```yaml
status: implemented
```

ou, se ainda estiver experimental:

```yaml
status: planned
```

Critério recomendado:

* `implemented`: gera métrica real pelo CLI;
* `planned`: ataque existe, mas ainda falta validação completa da linha;
* `blocked`: ataque ou pré-condição técnica ausente.

---

## 22. Critérios de aceite

* [ ] DeepFool foi implementado ou integrado no módulo de ataques.
* [ ] DeepFool é acessível pelo dispatcher/registry atual de ataques.
* [ ] Nenhum novo diretório foi criado dentro de `evaluation/`.
* [ ] Nenhum runner específico para DeepFool foi criado.
* [ ] Nenhum script novo foi criado em `scripts/`.
* [ ] A linha 7 de `table_10_googlenet` possui configuração explícita de DeepFool.
* [ ] A linha 7 não fica `blocked` se o ataque estiver funcional.
* [ ] Se o modelo GoogLeNet não expuser gradiente, a linha 7 continua bloqueada com motivo técnico claro.
* [ ] O comando `python scripts/run_experiment.py --experiment table_10_googlenet` continua sendo o ponto de entrada.
* [ ] O output continua em `results/experiments/table_10/imagenet/googlenet/`.
* [ ] O output contém apenas `metrics.csv` e `metrics.json`.
* [ ] A implementação reaproveita métricas e outputs existentes.
* [ ] A implementação não referencia resultados antigos.
* [ ] A implementação não usa `about:blank` como dependência ou referência válida.

---

## 23. Definition of Done

A implementação estará concluída quando:

1. `deepfool` estiver disponível no mecanismo comum de ataques;
2. a linha 7 de `table_10_googlenet` estiver configurada com parâmetros explícitos;
3. o runner atual da Table 10 conseguir processar a linha 7 sem fluxo especial;
4. `metrics.csv` e `metrics.json` forem gerados no diretório:

```text
results/experiments/table_10/imagenet/googlenet/
```

5. nenhum `manifest.json` for gerado;
6. nenhum novo diretório for criado dentro de `evaluation/`;
7. os testes unitários de DeepFool passarem;
8. o smoke test da Table 10 / GoogLeNet / DeepFool passar com amostra pequena.

```
```
