# DeepDetector Reimplementation

Reimplementacao organizada do metodo de Liang et al., "Detecting Adversarial
Image Examples in Deep Networks with Adaptive Noise Reduction", usando o
repositorio `OwenSec/DeepDetector` como referencia metodologica.

O objetivo principal e reproduzir primeiro o comportamento experimental do
artigo e do codigo original antes de propor extensoes. A arquitetura atual
separa codigo reutilizavel em `src/deepdetector`, configuracoes em `configs/`,
scripts operacionais em `scripts/`, notas de reproducao em `reproduction_notes/`
e saidas experimentais em `results/` e `artifacts/`.

## Estado Atual

- Pacote Python instalavel em modo editavel: `deepdetector`.
- Trilhas MNIST:
  - M1 + FGSM.
  - M2 + Carlini-Wagner L2/Linf.
- Trilha ImageNet com wrappers e utilitarios para Caffe/GoogLeNet.
- Filtros NumPy independentes do modelo: quantizacao, media, entropia e
  quantizacao adaptativa.
- Avaliacoes para detectar mudanca de predicao apos reducao de ruido.
- Scripts de reproducao para tabelas do artigo.
- Testes unitarios focados em filtros, dados, ataques, wrappers e comparacoes
  de reproducao.

## Ambiente

Este projeto preserva uma pilha legada de proposito:

- TensorFlow 1.x
- Keras 2.2.x
- CleverHans 2.x
- Caffe para a trilha ImageNet, quando a reproducao fiel for necessaria

Crie e valide o ambiente Conda:

```bash
conda env create -f environment.yml
conda activate adversarialimage-ids-legacy
pip install -e .
python scripts/dev/smoke_test.py
```

Se o ambiente ja existir:

```bash
conda env update -n adversarialimage-ids-legacy -f environment.yml
conda activate adversarialimage-ids-legacy
pip install -e .
python scripts/dev/smoke_test.py
```

Em automacao sem ativar o shell:

```bash
conda run -n adversarialimage-ids-legacy python scripts/dev/smoke_test.py
```

Detalhes de versao e decisoes ficam em
`reproduction_notes/environment_setup.md`.

## Arquitetura

```text
.
|-- configs/                         # contratos YAML por responsabilidade
|   |-- article_reproduction/         # reproducoes das tabelas do artigo
|   |-- attacks/                      # contratos isolados dos ataques
|   |-- experiments/                  # pipelines operacionais/exploratorios
|   |-- filters/                      # filtros e detector
|   `-- training/                     # treino/restauracao de modelos
|-- scripts/
|   |-- article_reproduction/         # scripts para tabelas do artigo
|   |-- dev/                          # validacoes rapidas de ambiente
|   |-- imagenet/                     # utilitarios da trilha ImageNet
|   `-- mnist/
|       |-- m1_fgsm/                  # fluxo MNIST M1 + FGSM
|       `-- m2_cw/                    # fluxo MNIST M2 + CW L2/Linf
|-- src/deepdetector/
|   |-- attacks/                      # FGSM e CW
|   |-- data/                         # loaders MNIST e ImageNet
|   |-- detection/                    # regra por mudanca de predicao
|   |-- evaluation/                   # metricas e relatorios
|   |-- filters/                      # filtros de reducao de ruido
|   |-- models/                       # modelos MNIST e wrappers ImageNet
|   |-- training/                     # treino/restauracao de baselines
|   |-- paths.py                      # convencoes de caminhos do projeto
|   `-- utils/
|-- tests/                            # cobertura automatizada
|-- reproduction_notes/               # notas, planos e decisoes de reproducao
|-- results/                          # relatorios e metricas versionaveis
|-- artifacts/                        # modelos, datasets processados e ataques
|-- environment.yml
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

### Modulos principais

| Modulo | Responsabilidade |
| --- | --- |
| `deepdetector.data` | Carregamento e preparacao de MNIST/ImageNet. |
| `deepdetector.models` | Definicoes de modelos MNIST e wrappers ImageNet/Caffe. |
| `deepdetector.training` | Treino e restore dos classificadores baseline. |
| `deepdetector.attacks` | Geracao de exemplos adversariais FGSM e CW. |
| `deepdetector.filters` | Reducoes de ruido usadas pelo detector. |
| `deepdetector.detection` | Detector DeepDetector-style por mudanca de predicao. |
| `deepdetector.evaluation` | Metricas, agregacoes e comparacoes com o artigo. |

## Configuracoes

Os arquivos em `configs/` documentam parametros experimentais e ajudam a manter
os fluxos rastreaveis:

- `configs/article_reproduction/`: contratos das tabelas do artigo.
- `configs/attacks/`: contratos isolados de FGSM, DeepFool e CW.
- `configs/experiments/`: pipelines operacionais ou exploratorios.
- `configs/filters/`: filtros/detector reutilizaveis.
- `configs/training/`: treino e restauracao dos modelos baseline.

## Fluxos Operacionais

### Validacao rapida

```bash
python scripts/dev/smoke_test.py
pytest
```

### MNIST M1 + FGSM

```bash
python scripts/mnist/m1_fgsm/train.py --load-model
python scripts/mnist/m1_fgsm/generate_attack.py --load-model
python scripts/mnist/m1_fgsm/run_comparison.py
```

Scripts relacionados:

- `scripts/mnist/m1_fgsm/train.py`
- `scripts/mnist/m1_fgsm/generate_attack.py`
- `scripts/mnist/m1_fgsm/evaluate_detector.py`
- `scripts/mnist/m1_fgsm/evaluate_entropy.py`
- `scripts/mnist/m1_fgsm/run_comparison.py`

Saidas principais:

- `artifacts/models/mnist/m1/`
- `artifacts/adversarial_examples/mnist/m1/fgsm/`
- `results/mnist/clean_baseline/`
- `results/mnist/fgsm/`
- `results/mnist/detector/`
- `results/mnist/final_mnist_results.csv`

### MNIST M2 + CW

```bash
python scripts/mnist/m2_cw/run_experiments.py --kappas 0.0,0.5,1.0,2.0,4.0 --samples 1000 --start-index 9000
```

Scripts relacionados:

- `scripts/mnist/m2_cw/train.py`
- `scripts/mnist/m2_cw/generate_attack_l2.py`
- `scripts/mnist/m2_cw/generate_attack_linf.py`
- `scripts/mnist/m2_cw/evaluate_detector.py`
- `scripts/mnist/m2_cw/run_experiments.py`

Saidas principais:

- `artifacts/models/mnist/m2/`
- `artifacts/adversarial_examples/mnist/m2/cw_l2/`
- `artifacts/adversarial_examples/mnist/m2/cw_linf/`
- `results/mnist/m2_cw/`

### ImageNet / Caffe

Baixe os ativos Caffe necessarios:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet
```

Liste modelos suportados e prepare outros ativos:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --list-models
python scripts/imagenet/download_caffe_imagenet_assets.py --model alexnet
```

Outros scripts:

- `scripts/imagenet/process_imagenet_subset.py`
- `scripts/imagenet/googlenet_fgsm.py`

Detalhes de fontes, espelhos e limitacoes ficam em
`reproduction_notes/caffe_model_downloads.md` e
`reproduction_notes/caffe_setup.md`.

### Reproducao das tabelas do artigo

```bash
python scripts/article_reproduction/table_3.py
python scripts/article_reproduction/table_4_mnist.py
python scripts/article_reproduction/table_6.py
python scripts/article_reproduction/table_10.py
python scripts/article_reproduction/table_10_m2.py
```

ImageNet:

```bash
python scripts/article_reproduction/table_4_imagenet.py
python scripts/article_reproduction/table_7_imagenet.py --config configs/article_reproduction/imagenet_table_7.yaml
```

As comparacoes ficam em `results/mnist/article_reproduction/`,
`results/mnist/m2_cw/article_comparison/` e
`results/imagenet/article_reproduction/`.

## Dados, Artefatos e Resultados

- `data/` e ignorado pelo Git e deve conter datasets locais.
- `artifacts/` armazena modelos, checkpoints, assets Caffe e adversariais
  gerados. Modelos grandes e binarios nao devem ser versionados.
- `results/` guarda relatorios, CSVs e notas de execucao que documentam a
  reproducao. A politica atual permite versionar resultados MNIST selecionados
  e mantem a trilha ImageNet restrita a arquivos explicitamente liberados.

## Notas de Reproducao

Use `reproduction_notes/` para decisoes tecnicas que precisam ser versionadas:

- `environment_setup.md`
- `caffe_setup.md`
- `caffe_model_downloads.md`
- `mnist_reproduction_notes.md`
- `experiment_plans/`

Documentos locais de trabalho que nao devem entrar no Git, como o guia de
Spec-Driven Development, ficam ignorados pelo `.gitignore`.

## Testes

Execute toda a suite:

```bash
pytest
```

Execute um teste especifico:

```bash
pytest tests/test_quantization_numpy.py
```

Os testes atuais cobrem filtros NumPy, registro de filtros, dados ImageNet,
wrappers ImageNet, FGSM ImageNet e comparacoes de reproducao do artigo.
