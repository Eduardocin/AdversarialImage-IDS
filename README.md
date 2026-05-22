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
# Se for a primeira vez, crie o ambiente completo:
conda env create -f environment.yml

# Ative o ambiente e valide as dependencias:
conda activate adversarialimage-ids-legacy
```

Se o ambiente ja existir e voce quiser sincronizar os pacotes com este
repositorio:

```bash
conda env update -n adversarialimage-ids-legacy -f environment.yml
```

Em automacao sem ativar o shell:

```bash
conda run -n adversarialimage-ids-legacy python scripts/smoke_test.py
```

Detalhes de versao e decisoes estao em
`reproduction_notes/environment_setup.md`.

## Estrutura

```text
.
├── configs/
│   ├── mnist_fgsm.yaml        # parametros do ataque FGSM em MNIST
│   └── mnist_entropy.yaml     # parametros da reducao adaptativa por entropia
├── scripts/
│   ├── smoke_test.py          # valida imports do ambiente legado
│   ├── train_mnist.py         # treina/carrega baseline limpo MNIST
│   ├── generate_mnist_fgsm.py # gera adversariais FGSM com CleverHans
│   └── evaluate_mnist_detector.py # avalia detector por mudanca de predicao
├── src/deepdetector/
│   ├── data/                  # carregamento e preparacao de dados
│   │   └── mnist.py
│   ├── models/                # modelos TensorFlow/Keras
│   │   └── mnist_cnn.py
│   ├── training/              # treino e restore de checkpoints
│   │   └── train_mnist.py
│   ├── attacks/               # ataques adversariais
│   │   └── fgsm.py
│   ├── filters/               # filtros NumPy independentes do modelo
│   │   └── quantization.py
│   ├── detection/             # regra DeepDetector-style
│   │   └── prediction_change.py
│   ├── evaluation/            # metricas e relatorios
│   │   ├── adversarial.py
│   │   └── detector_metrics.py
│   └── utils/
├── tests/
│   └── test_quantization_numpy.py
├── results/
│   ├── mnist/
│   │   ├── .gitkeep
│   │   └── quantization/filter_notes.md
│   └── imagenet/
│       └── .gitkeep
├── reproduction_notes/
│   ├── environment_setup.md
│   └── caffe_setup.md
├── environment.yml            # criacao automatizada do conda legado
├── requirements.txt           # pins pip do ambiente legado
└── README.md
```

`results/mnist/clean_baseline/`, `results/mnist/fgsm/` e
`results/mnist/detector/` sao saidas locais de execucao e ficam ignoradas pelo
Git. A excecao versionada em `results/` e a nota
`results/mnist/quantization/filter_notes.md`, porque ela documenta a reproducao
dos filtros.

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
