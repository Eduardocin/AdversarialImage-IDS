# DeepDetector Reimplementation

Base organizada para reproduzir o metodo de Liang et al.,
"Detecting Adversarial Image Examples in Deep Networks with Adaptive Noise
Reduction", usando o repositorio `OwenSec/DeepDetector` como referencia
metodologica.

O Marco 0 cria apenas a estrutura modular e documenta o ambiente legado. A
implementacao experimental deve preservar primeiro o comportamento original
antes de qualquer extensao.

## Ambiente Legado

Este projeto usa bibliotecas antigas de proposito:

- TensorFlow 1.x
- Keras 2.2.x
- CleverHans 2.x
- Caffe para a trilha ImageNet, se a reproducao fiel for mantida

Ambiente conda usado localmente:

```bash
# Se for a primeira vez, crie o ambiente:
conda create -n adversarialimage-ids-legacy python=3.7

# Ative o ambiente e instale as dependencias:
conda activate adversarialimage-ids-legacy
pip install -r requirements.txt
python scripts/smoke_test.py
```

Em automacao sem ativar o shell:

```bash
conda run -n adversarialimage-ids-legacy python scripts/smoke_test.py
```

Detalhes de versao e decisoes estao em
`reproduction_notes/environment_setup.md`.

## Estrutura

```text
configs/
  mnist_fgsm.yaml
  mnist_entropy.yaml
src/deepdetector/
  data/
  models/
  attacks/
  filters/
  detection/
  evaluation/
  utils/
scripts/
  smoke_test.py
tests/
results/
  mnist/
  imagenet/
reproduction_notes/
  environment_setup.md
  caffe_setup.md
```

## Configuracoes Iniciais

- `configs/mnist_fgsm.yaml`: esqueleto do experimento MNIST com FGSM.
- `configs/mnist_entropy.yaml`: parametros iniciais da reducao adaptativa por
  entropia descrita na reproducao.

Os arquivos de configuracao ainda não executam experimentos completos, eles servem como contrato inicial para os proximos marcos.

## Validacao Rapida

O smoke test valida somente imports da pilha mínima:

```bash
python scripts/smoke_test.py
```

Quando esse comando falha, o problema esperado esta no ambiente legado, não na estrutura modular do projeto.

## MNIST Clean Baseline

O primeiro pipeline executável treina ou carrega o classificador MNIST limpo com TensorFlow 1.x, Keras e CleverHans:

```bash
python scripts/train_mnist.py --epochs 1 --batch-size 256 --learning-rate 0.001
```

Para restaurar o checkpoint salvo:

```bash
python scripts/train_mnist.py --load-model
```

O script salva checkpoint em `results/mnist/clean_baseline/checkpoints/` e
registra a acuracia limpa em:

```text
results/mnist/clean_baseline/summary.md
```

## MNIST FGSM

Gerar exemplos adversariais FGSM com CleverHans a partir do baseline limpo:

```bash
python scripts/generate_mnist_fgsm.py --load-model --epsilons 0.2 --samples 4500
```

Para comparar epsilons:

```bash
python scripts/generate_mnist_fgsm.py --load-model --epsilons 0.1,0.2
```

As imagens adversariais ficam em `results/mnist/fgsm/eps_<valor>/` e as
metricas agregadas em:

```text
results/mnist/fgsm/summary.csv
results/mnist/fgsm/summary.md
```
