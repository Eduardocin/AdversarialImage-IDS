# AdversarialImage-IDS / DeepDetector

Reimplementacao orientada por especificacoes do metodo de Liang et al.,
"Detecting Adversarial Image Examples in Deep Networks with Adaptive Noise
Reduction". O projeto usa `OwenSec/DeepDetector` como referencia metodologica,
mas o codigo atual esta organizado como um pacote Python testavel, com
experimentos declarados em YAML e execucao centralizada.

O foco do repositorio e reproduzir e estender, de forma controlada, fluxos de
deteccao de exemplos adversariais para datasets MNIST-like e ImageNet:

- reproducoes das tabelas 3, 4, 6, 7, 8, 9 e 10 do artigo;
- avaliacao de ataques FGSM, DeepFool, Carlini-Wagner L2/Linf e variantes
  defense-aware;
- filtros de reducao de ruido por quantizacao, suavizacao espacial, entropia e
  filtro final proposto;
- experimentos novos com Fashion-MNIST, novas classes ImageNet e regras top-k.

## Como o projeto e desenvolvido

Este repositorio segue Spec-Driven Development (SDD). Antes de alterar
comportamento ou implementar uma feature, leia a especificacao correspondente em
`specs/`. As tarefas derivadas ficam em `specs/tasks.md`.

Fluxo esperado:

1. especificacao em `specs/features/` ou `specs/refactor/`;
2. item de implementacao em `specs/tasks.md`;
3. codigo em `src/deepdetector`;
4. testes em `tests`;
5. validacao por pytest e revisao contra os criterios de aceite.

Nao use o README como fonte de verdade para requisitos de experimento. A fonte
de verdade e `specs/`, seguida de `configs/experiments.yaml`.

## Estrutura

```text
.
|-- src/deepdetector/          # pacote Python principal
|   |-- attacks/               # FGSM, DeepFool, CW e ataques adaptativos
|   |-- data/                  # loaders MNIST, Fashion-MNIST e ImageNet
|   |-- detection/             # regras de deteccao por mudanca de predicao
|   |-- evaluation/            # metricas e logica das tabelas
|   |-- experiments/           # runners configuraveis
|   |-- filters/               # filtros de reducao de ruido
|   |-- io/                    # YAML, caminhos e escrita de resultados
|   |-- models/                # modelos MNIST-like e wrappers ImageNet
|   `-- training/              # treino/restauracao de checkpoints
|-- configs/
|   |-- experiments.yaml       # inventario oficial de experimentos
|   `-- article_reproduction/  # configs auxiliares preservadas
|-- scripts/
|   |-- run_experiment.py      # ponto de entrada oficial
|   |-- train_fashion_mnist_checkpoint.py
|   |-- validate_inception_env.py
|   |-- article_reproduction/  # auxiliares especificos, principalmente M2/CW
|   |-- dev/                   # smoke tests locais
|   `-- imagenet/              # materializacao e ativos ImageNet/Caffe
|-- specs/                     # especificacoes e tarefas SDD
|-- tests/                     # suite pytest
|-- envs/                      # ambientes Conda versionados
|-- reproduction_notes/        # notas de reproducao e decisoes tecnicas
|-- data/                      # datasets locais, ignorados pelo Git
|-- artifacts/                 # modelos/cache/ataques, ignorados pelo Git
`-- results/                   # resultados regeneraveis
```

## Ambientes

O ambiente principal preserva a pilha legada usada pela maior parte da
reproducao:

- TensorFlow 1.x;
- Keras 2.2.x;
- CleverHans 2.x;
- Caffe para os fluxos ImageNet com GoogLeNet/CaffeNet.

Crie ou atualize o ambiente principal:

```bash
conda env create -f envs/environment.yml
conda activate adversarialimage-ids-legacy
pip install -e .
python scripts/dev/smoke_test.py
```

Se o ambiente ja existir:

```bash
conda env update -n adversarialimage-ids-legacy -f envs/environment.yml
conda activate adversarialimage-ids-legacy
pip install -e .
python scripts/dev/smoke_test.py
```

Para automacao local neste repositorio, prefira executar pelo WSL com o ambiente
`adversarialimage-ids-legacy`.

```bash
conda run -n adversarialimage-ids-legacy python scripts/dev/smoke_test.py
```

### Inception v3 / TensorFlow 2

O fluxo Inception v3 usa um ambiente separado, definido em
`envs/inceptionv3-tf2.yml`, com TensorFlow 2.11 em modo `compat.v1`.

```bash
conda env create -f envs/inceptionv3-tf2.yml
conda activate adversarialimage-inceptionv3-tf2
pip install -e .
python scripts/validate_inception_env.py
```

## Experimentos

O ponto de entrada oficial e sempre:

```bash
python scripts/run_experiment.py --experiment <experiment_id>
```

Os experimentos publicos sao declarados em `configs/experiments.yaml`. Exemplos:

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
python scripts/run_experiment.py --experiment defense_aware
python scripts/run_experiment.py --experiment topk_detection
python scripts/run_experiment.py --experiment fashion_mnist_fgsm_m3
python scripts/run_experiment.py --experiment fashion_mnist_cw_l2_m2
python scripts/run_experiment.py --experiment imagenet_new_classes_fgsm_googlenet
python scripts/run_experiment.py --experiment imagenet_new_classes_deepfool_caffenet
python scripts/run_experiment.py --experiment imagenet_new_classes_cw_l2_inception_v3
```

`table_4` e composta e executa `table_4_mnist` e `table_4_imagenet`. Tambem e
possivel rodar apenas um componente:

```bash
python scripts/run_experiment.py --experiment table_4_mnist
python scripts/run_experiment.py --experiment table_4_imagenet
```

Table 5 nao tem fluxo oficial no inventario atual.

## Dados e artefatos

Datasets, checkpoints, modelos Caffe, grafos TensorFlow e exemplos
adversariais gerados nao devem ser versionados.

Principais caminhos locais:

| Caminho | Papel |
| --- | --- |
| `data/` | datasets locais, como MNIST/Fashion-MNIST CSV e pastas ImageNet |
| `artifacts/models/` | checkpoints e modelos externos |
| `artifacts/adversarial_examples/` | cache de exemplos adversariais |
| `results/` | metricas e relatorios regeneraveis |

Para ativos Caffe:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --list-models
python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet
python scripts/imagenet/download_caffe_imagenet_assets.py --model alexnet
```

Ativos que nao estao no downloader, como alguns artefatos CaffeNet usados nos
experimentos, devem seguir as notas em `reproduction_notes/caffe_setup.md`.

Para materializar o subset local usado por Inception v3:

```bash
python scripts/imagenet/materialize_inceptionv3_subset.py
```

As notas de ambiente e ativos ficam em:

- `reproduction_notes/environment_setup.md`
- `reproduction_notes/caffe_setup.md`
- `reproduction_notes/caffe_model_downloads.md`
- `reproduction_notes/table10_inceptionv3_cw_l2_investigation.md`

## Fashion-MNIST

Antes dos experimentos Fashion-MNIST, prepare os checkpoints configurados:

```bash
python scripts/train_fashion_mnist_checkpoint.py --model m3
python scripts/train_fashion_mnist_checkpoint.py --model m2
```

Depois rode:

```bash
python scripts/run_experiment.py --experiment fashion_mnist_fgsm_m3
python scripts/run_experiment.py --experiment fashion_mnist_cw_l2_m2
```

## Table 10 M2 / CW

O grupo M2/CW depende dos backends originais de `nn_robust_attacks`. Para
regenerar adversariais:

```bash
python scripts/article_reproduction/mnist_table_10_m2_cw.py \
  --generate-attacks \
  --overwrite-attacks \
  --nn-robust-attacks-root nn_robust_attacks
```

Sem `--generate-attacks`, o script avalia exemplos `.npy` ja existentes.

## Saidas

A maioria dos experimentos grava `metrics.csv` e `metrics.json` no diretorio
configurado. Alguns fluxos tem contratos especificos:

| Experimento | Saida principal |
| --- | --- |
| `table_4` | `results/table_4/mnist/`, `results/table_4/imagenet/` e manifesto |
| `table_7` | pivot `table_7_imagnet.csv` e `table_7_status.json` |
| `table_8` | pivot `table_8_imagenet.csv` e `table_8_status.json` |
| `table_10_*` | `metrics.csv`, `metrics.json` e, quando aplicavel, `manifest.json` |
| `topk_detection` | selecao, metricas e imagens ambiguas selecionadas |

## Testes

Execute primeiro a validacao mais estreita relacionada a mudanca. Para a suite
completa:

```bash
pytest
```

Para um arquivo especifico:

```bash
pytest tests/test_quantization_numpy.py
```

Os testes cobrem filtros NumPy, loaders, wrappers ImageNet, ataques, runners,
contratos de output e comparacoes de reproducao.
