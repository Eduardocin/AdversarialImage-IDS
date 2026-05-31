# Spec: Refatoração Mínima InceptionV3 para GPUs Ampere

## Objetivo

Fazer `InceptionV3TensorFlowWrapper` funcionar com TensorFlow 2.11 no lugar de
TensorFlow 1.15.5, sem alterar nenhuma lógica de negócio. O código trava hoje no
primeiro `sess.run` em GPUs Ampere (RTX 3090, A100) porque o binário
`tensorflow-gpu==1.15.5` foi compilado para CUDA 10 e não tem kernels para sm_86/sm_80.
A solução é trocar o stack de TF1+CUDA10 para TF2.11+CUDA11, com as mínimas
adaptações de API necessárias.

## Repositório e branch

- Repo: `Eduardocin/AdversarialImage-IDS`
- Branch de trabalho: `deepfool`
- Criar branch nova a partir de `deepfool`: `fix/inceptionv3-ampere`

---

## Tarefa 1 — Criar arquivo de ambiente conda

**Criar o arquivo:** `envs/inceptionv3-tf2.yml`

Conteúdo exato:

```yaml
name: adversarialimage-inceptionv3-tf2

channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.8.18
  - pip=23.3.1
  - setuptools
  - wheel

  - pip:
      - tensorflow==2.11.0
      - numpy==1.23.5
      - pillow==9.5.0
      - scipy==1.10.1
      - matplotlib==3.7.2
      - pandas==2.0.3
      - tqdm==4.65.0
      - pyyaml==6.0.1
      - h5py==3.9.0
      - protobuf==3.20.3
      - pytest==7.4.0
      - cleverhans==3.1.0
```

**Não modificar** o arquivo `envs/adversarialimage-ids-gpu.yml` existente.

---

## Tarefa 2 — Editar `imagenet_wrappers.py`

**Arquivo:** `src/deepdetector/models/imagenet_wrappers.py`  
**Classe:** `InceptionV3TensorFlowWrapper` (única classe a ser modificada)

### Mudança 2a — `__init__`: adicionar `disable_eager_execution`

Localizar o bloco abaixo dentro de `__init__`:

```python
        self.tf = tf
        self.image_size = 299
```

Substituir por:

```python
        self.tf = tf
        self.tf.compat.v1.disable_eager_execution()
        self.image_size = 299
```

### Mudança 2b — `_import_logits`: corrigir chamada removida no TF2

Localizar o bloco abaixo dentro de `_import_logits`:

```python
        elements = self.tf.import_graph_def(
            self.graph_def,
            name=name,
            input_map={self.input_map_name: self._scaled_input(tensor)},
            return_elements=[self.output_tensor_name],
        )
```

Substituir por:

```python
        elements = self.tf.compat.v1.graph_util.import_graph_def(
            self.graph_def,
            name=name,
            input_map={self.input_map_name: self._scaled_input(tensor)},
            return_elements=[self.output_tensor_name],
        )
```

### O que NÃO deve ser alterado

- Qualquer outro método de `InceptionV3TensorFlowWrapper`
- As classes `GoogLeNetCaffeWrapper`, `CaffeNetCaffeWrapper`, `ImageNetModelWrapper`
- Qualquer outro arquivo do repositório

---

## Tarefa 3 — Criar script de validação

**Criar o arquivo:** `scripts/validate_inception_env.py`

Conteúdo exato:

```python
"""
Validacao do ambiente adversarialimage-inceptionv3-tf2.

Execucao:
    conda activate adversarialimage-inceptionv3-tf2
    python scripts/validate_inception_env.py
"""
import sys
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print(f"Python : {sys.version}")
print(f"TF     : {tf.__version__}")
print(f"GPUs   : {tf.config.list_physical_devices('GPU')}")

# Teste 1 — sess.run basico
g = tf.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 4))
    y = x * 2.0
sess = tf.compat.v1.Session(graph=g)
out = sess.run(y, feed_dict={x: np.ones((2, 4), dtype=np.float32)})
assert out.shape == (2, 4)
np.testing.assert_allclose(out, 2.0, rtol=1e-5)
print("Teste 1 - sess.run basico: OK")

# Teste 2 — import_graph_def
ig = tf.Graph()
with ig.as_default():
    a = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3), name="inp")
    _ = a + 0.5
gdef = ig.as_graph_def()

og = tf.Graph()
with og.as_default():
    p = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
    elems = tf.compat.v1.graph_util.import_graph_def(
        gdef, name="im", input_map={"inp:0": p}, return_elements=["add:0"]
    )
sess2 = tf.compat.v1.Session(graph=og)
out2 = sess2.run(elems[0], feed_dict={p: np.zeros((1, 299, 299, 3), dtype=np.float32)})
assert out2.shape == (1, 299, 299, 3)
np.testing.assert_allclose(out2, 0.5, rtol=1e-5)
print("Teste 2 - import_graph_def + sess.run: OK")

# Teste 3 — cleverhans importavel
import cleverhans  # noqa: F401
print("Teste 3 - cleverhans importavel: OK")

print("\nAmbiente OK.")
```

---

## Tarefa 4 — Abrir Pull Request

- Base: `deepfool`
- Head: `fix/inceptionv3-ampere`
- Título: `fix: InceptionV3TensorFlowWrapper compatível com Ampere (TF2 + CUDA11)`
- Corpo:

```
## Problema
TF1.15 + CUDA10 trava no primeiro sess.run em GPUs Ampere (RTX 3090, A100).
O ambiente legado funciona localmente na GTX 1650 (Turing/sm_75).

## Solução
- Novo ambiente conda `envs/inceptionv3-tf2.yml` com TF 2.11 + CUDA 11.
- Duas mudanças mínimas em `InceptionV3TensorFlowWrapper`:
  1. `disable_eager_execution()` para compatibilidade com `sess.run`.
  2. `tf.import_graph_def` → `tf.compat.v1.graph_util.import_graph_def` (API removida no TF2).
- Ambiente Caffe legado (`adversarialimage-ids-gpu`) inalterado.
- Nenhuma lógica de negócio alterada.

## Validação
`python scripts/validate_inception_env.py` após ativar o novo ambiente.
```

---

## Restrições

- Não alterar nenhum arquivo fora do descrito nas Tarefas 1–3.
- Não alterar a lógica de nenhum método, apenas as duas substituições de texto especificadas.
- O diff final deve ter exatamente **2 linhas alteradas** em `imagenet_wrappers.py`
  (uma adição em `__init__`, uma substituição em `_import_logits`).
- Não instalar nem executar nada — apenas criar/editar os arquivos e abrir o PR.
