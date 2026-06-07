# Scripts

Indice operacional dos scripts do projeto. A pasta `scripts/` e uma interface
de execucao; a logica experimental comum fica em `src/deepdetector`.

Use `envs/README.md` para a matriz completa experimento -> ambiente. O resumo
abaixo apenas ajuda a escolher rapidamente o ambiente antes de chamar o script.

## Runner Oficial

Os experimentos oficiais usam um unico ponto de entrada e a configuracao
consolidada em `configs/experiments.yaml`:

```bash
python scripts/run_experiment.py --experiment <experiment_id>
```

| Grupo | Experimentos | Ambiente recomendado |
| --- | --- | --- |
| MNIST/local | `table_3`, `table_4_mnist`, `table_10_m1`, `table_10_m2`, `defense_aware` | `adversarialimage-ids-legacy` |
| ImageNet/Caffe | `table_4_imagenet`, `table_7`, `table_8`, `table_10_googlenet`, `table_10_caffenet` | `adversarialimage-ids-gpu` |
| Compostos MNIST + ImageNet | `table_4`, `table_6`, `table_9` | `adversarialimage-ids-legacy` or `adversarialimage-ids-gpu`, depending on the component |
| InceptionV3 | `table_10_inception_v3`, `imagenet_new_classes_cw_l2_inception_v3` | `adversarialimage-inceptionv3-tf2` |
| New datasets | `fashion_mnist_fgsm_m3`, `fashion_mnist_cw_l2_m2`, selected ImageNet new-class experiments | See `envs/README.md` |
| Top-k | `topk_detection` | Depends on the configured dataset |

Common commands:

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

`table_4` executes both table components in sequence. To run only one side:

```bash
python scripts/run_experiment.py --experiment table_4_mnist
python scripts/run_experiment.py --experiment table_4_imagenet
```

Table 5 is not part of the operational path because the current code inventory
has no official script, config, or result contract for it.

## Outputs

Most simple runs write:

```text
results/<experiment_id>/metrics.csv
results/<experiment_id>/metrics.json
```

Tables 7 and 8 ImageNet are format exceptions: they write pivot CSV files
`table_7_imagenet.csv` and `table_8_imagenet.csv`, with their respective
`table_*_status.json` files.

Table 4 is composite: its results are written under `results/table_4/mnist/`
and `results/table_4/imagenet/`, with `manifest.json` at the table root. Table 6
and Table 9 also run MNIST and ImageNet internally, but write only the official
aggregates under `results/table_6/` and `results/table_9/`.

Table 10 is separated by model group. ImageNet groups write `metrics.csv`,
`metrics.json`, and `manifest.json` under `results/table_10/<group>/`. M2 CW is
split between `results/table_10/M2_cw_l2/` and
`results/table_10/M2_cw_Linf/`.

## Auxiliary Scripts

| Script | Role | Environment |
| --- | --- | --- |
| `dev/smoke_test.py` | Fast import/dependency validation. | `adversarialimage-ids-legacy` or `adversarialimage-ids-gpu` |
| `imagenet/download_caffe_imagenet_assets.py` | Download Caffe assets for the ImageNet path. | `adversarialimage-ids-gpu` |
| `imagenet/materialize_inceptionv3_subset.py` | Materialize the local InceptionV3 subset. | `adversarialimage-inceptionv3-tf2` |
| `train_fashion_mnist_checkpoint.py` | Prepare Fashion-MNIST M2/M3 checkpoints. | `adversarialimage-ids-legacy` |

For Fashion-MNIST checkpoints:

```bash
python scripts/train_fashion_mnist_checkpoint.py --model m3
python scripts/train_fashion_mnist_checkpoint.py --model m2
```

For M2 CW adversarial regeneration:

```bash
python scripts/article_reproduction/mnist_table_10_m2_cw.py \
  --generate-attacks \
  --overwrite-attacks \
  --nn-robust-attacks-root nn_robust_attacks
```

This M2 flow uses `nn_robust_attacks.CarliniL2` for CW-L2 and
`nn_robust_attacks.CarliniLi` for CW-Linf.

Historical article reproduction scripts can remain for compatibility, but the
official path for public experiments is `scripts/run_experiment.py`.
