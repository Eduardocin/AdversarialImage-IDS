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
|   `-- experiments.yaml              # inventario de experimentos
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
|   |-- evaluation/                   # metricas, tabelas e comparacoes do artigo
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
| `deepdetector.evaluation` | Metricas, agregacoes e avaliacao das tabelas do artigo, incluindo materializadores oficiais. |

## Configuracoes

O fluxo oficial usa `configs/experiments.yaml`. Cada experimento declara
explicitamente dataset, modelo, ataque, amostras/splits e filtros. As
configuracoes de reproducoes especificas ficam em
`configs/article_reproduction/`.

## Fluxos Operacionais

### Validacao rapida

```bash
python scripts/dev/smoke_test.py
pytest
```

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

Detalhes de fontes, espelhos e limitacoes ficam em
`reproduction_notes/caffe_model_downloads.md` e
`reproduction_notes/caffe_setup.md`.

### Reproducao das tabelas do artigo

```bash
python scripts/run_experiment.py --experiment table_3
python scripts/run_experiment.py --experiment table_4
python scripts/run_experiment.py --experiment table_6
python scripts/run_experiment.py --experiment table_7
python scripts/run_experiment.py --experiment table_8
python scripts/run_experiment.py --experiment table_9
python scripts/run_experiment.py --experiment table_10_m1
python scripts/run_experiment.py --experiment table_10_googlenet
python scripts/run_experiment.py --experiment table_10_caffenet
python scripts/run_experiment.py --experiment table_10_m2
python scripts/run_experiment.py --experiment table_10_inception_v3
```

As Tabelas 3-9 usam `configs/experiments.yaml`. A Table 4 e composta: por
padrao `table_4` executa `table_4_mnist` e `table_4_imagenet` em sequencia.
As Tables 6 e 9 tambem combinam MNIST e ImageNet internamente e gravam metricas
agregadas por split. Tambem e possivel executar apenas um lado da Table 4:

```bash
python scripts/run_experiment.py --experiment table_4_mnist
python scripts/run_experiment.py --experiment table_4_imagenet
```

Os resultados da Table 4 ficam sob `results/experiments/table_4/`, separados em
`mnist/` e `imagenet/` porque os CSVs tem schemas diferentes. Table 5 nao
aparece nesse arquivo porque nao ha fluxo correspondente no inventario atual.

Table 10 e executada por grupo de modelo. Cada grupo escreve
`metrics.csv`, `metrics.json` e `manifest.json` em
`results/experiments/table_10/<grupo>/`.

## Dados, Artefatos e Resultados

- `data/` e ignorado pelo Git e deve conter datasets locais.
- `artifacts/` armazena modelos, checkpoints, assets Caffe e adversariais
  gerados. Modelos grandes e binarios nao devem ser versionados.
- `results/` nao e fonte de verdade. Outputs oficiais sao regenerados em
  `results/experiments/<experiment_id>/metrics.csv` e `metrics.json`.

## Notas de Reproducao

Use `reproduction_notes/` para decisoes tecnicas que precisam ser versionadas:

- `environment_setup.md`
- `caffe_setup.md`
- `caffe_model_downloads.md`

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
