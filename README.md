# AdversarialImage-IDS

Replicacao MNIST do fluxo do DeepDetector / Liang et al.,
"Detecting Adversarial Image Examples in Deep Neural Networks with Adaptive
Noise Reduction".

O foco atual e MNIST. O codigo principal em `src/mnist` reescreve o fluxo no
estilo do projeto, preservando a pilha e os parametros da base original.

## Fluxo MNIST

```text
MNIST
 -> modelo classificador M1/M2
 -> exemplos adversariais
      M1: FGSM via CleverHans FastGradientMethod
      M2: C&W L2/Linf via Carlini nn_robust_attacks
 -> reducao adaptativa de ruido
      entropia < 4: quantizacao 128
      entropia < 5: quantizacao 64
      caso contrario: quantizacao 43 + cross mean filter
 -> comparacao da predicao antes/depois da reducao
 -> contadores TP, FN, FP, TTP
 -> precision e recall
```

DeepFool esta adiado porque ainda nao ha uma implementacao MNIST propria nesta
etapa.

## Ambiente

O ambiente principal e legado de proposito, para preservar a reproducao:

- Caffe
- TensorFlow 1.x
- Keras 2.x
- CleverHans 2.x
- Carlini `nn_robust_attacks` em `third_party/`

Criar o ambiente:

```bash
conda env create -f environment.yml
```

Ativar:

```bash
conda activate adversarialimage-ids-legacy
```

Validar imports da pilha MNIST:

```bash
python scripts/mnist/check_environment.py
```

Em automacao sem ativar o shell:

```bash
conda run -n adversarialimage-ids-legacy python scripts/mnist/check_environment.py
```

## Como Rodar o M1

M1 e o classificador MNIST usado no fluxo FGSM/CleverHans. Ele salva checkpoint
em:

```text
outputs/mnist/m1/mnist.ckpt
```

Smoke test de 1 epoca:

```bash
python scripts/mnist/run_m1.py --smoke --no-load-model
```

Treino completo com os defaults da base:

```bash
python scripts/mnist/run_m1.py --no-load-model
```

Carregar checkpoint existente e avaliar:

```bash
python scripts/mnist/run_m1.py --load-model
```

Ultimo resultado local obtido:

```text
epochs=6
batch_size=128
learning_rate=0.001
clean_accuracy=0.9899
checkpoint=outputs/mnist/m1/mnist.ckpt
```

## Como Rodar o M2

M2 e o classificador MNIST usado no fluxo C&W/Carlini. Os pesos esperados ficam
em:

```text
third_party/nn_robust_attacks/models/mnist
```

Verificar se o M2 ja existe e carrega:

```bash
python scripts/mnist/check_m2.py
```

Treinar/recriar o M2 com a arquitetura e hiperparametros Carlini:

```bash
python scripts/mnist/train_m2.py
```

Smoke de treino curto:

```bash
python scripts/mnist/train_m2.py --epochs 1
```

O treino completo usa por default:

```text
epochs=50
batch_size=128
optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
output=third_party/nn_robust_attacks/models/mnist
```

## Estrutura

```text
src/mnist/
  config.py           # defaults e caminhos do fluxo MNIST
  data.py             # MNIST IDX, normalizacao M1/M2 e splits
  models.py           # modelos M1/M2 em Keras/CleverHans/Carlini
  m1.py               # treino/carregamento/avaliacao do M1
  m2.py               # checagem/carregamento do M2
  attacks.py          # FGSM e C&W usando as bibliotecas originais
  noise_reduction.py  # quantizacao, entropia e filtro adaptativo
  detection.py        # regra de deteccao por mudanca de predicao
  evaluation.py       # TP, FN, FP, TTP, precision, recall
  experiments.py      # orquestracao ataque -> reducao -> avaliacao
```

## Testes

Testes rapidos das partes que nao exigem o runtime legado completo:

```bash
pytest tests
```

Se o `pytest` estiver em outro ambiente, use:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path; pytest tests
```
