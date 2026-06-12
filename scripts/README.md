# Scripts

Operational index for the project scripts. The `scripts/` directory is an execution interface; shared experiment logic should live under `src/deepdetector`.

Official public experiments should be executed through the centralized runner and the consolidated configuration file:

```bash
python scripts/run_experiment.py --experiment <experiment_id>
```

The experiment registry is defined in:

```text
configs/experiments.yaml
```

For environment details, use `envs/README.md`. In the current `main` branch, the default environment for most workflows is `adversarialimage-ids-gpu`. InceptionV3 workflows use `adversarialimage-inceptionv3-tf2`.

## Official Runner

```bash
python scripts/run_experiment.py --experiment <experiment_id>
```

## Experiment Groups

| Group | Experiments | Recommended environment |
| --- | --- | --- |
| MNIST/local | `table_3`, `table_4_mnist`, `table_10_m1`, `table_10_m2`, `defense_aware` | `adversarialimage-ids-gpu` |
| ImageNet/Caffe | `table_4_imagenet`, `table_7`, `table_8`, `table_10_googlenet`, `table_10_caffenet` | `adversarialimage-ids-gpu` |
| Composite MNIST + ImageNet | `table_4`, `table_6`, `table_9` | `adversarialimage-ids-gpu` |
| InceptionV3 | `table_10_inception_v3`, `imagenet_new_classes_cw_l2_inception_v3` | `adversarialimage-inceptionv3-tf2` |
| New datasets/classes | `fashion_mnist_cw_l2_m2`, selected ImageNet new-class experiments | See `envs/README.md` |
| Top-k investigation | `topk_detection` | Depends on the configured dataset |

## Common Commands

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

Table 5 is not part of the operational path because the current code inventory has no official script, config, or result contract for it.

## Outputs

Most simple runs write:

```text
results/<experiment_id>/metrics.csv
results/<experiment_id>/metrics.json
```

Tables 7 and 8 ImageNet are format exceptions. They write pivot CSV files, respectively:

```text
table_7_imagenet.csv
table_8_imagenet.csv
```

and their corresponding status files:

```text
table_7_status.json
table_8_status.json
```

Table 4 is composite. Its results are written under:

```text
results/table_4/mnist/
results/table_4/imagenet/
```

with `manifest.json` at the table root.

Table 6 and Table 9 also run MNIST and ImageNet internally, but write only the official aggregates under:

```text
results/table_6/
results/table_9/
```

Table 10 is separated by model group. ImageNet groups write `metrics.csv`, `metrics.json`, and `manifest.json` under their configured result directories. The MNIST M2 CW group (`table_10_m2`) follows the same contract and writes `metrics.csv`, `metrics.json`, and `manifest.json` under `results/table_10/M2_cw/`.

## Auxiliary Scripts

| Script | Role | Environment |
| --- | --- | --- |
| `dev/smoke_test.py` | Fast import and dependency validation. | `adversarialimage-ids-gpu` |
| `imagenet/download_caffe_imagenet_assets.py` | Download Caffe assets for the ImageNet path. | `adversarialimage-ids-gpu` |
| `imagenet/materialize_inceptionv3_subset.py` | Materialize the local InceptionV3 subset. | `adversarialimage-inceptionv3-tf2` |
| `train_fashion_mnist_checkpoint.py` | Prepare Fashion-MNIST M2 checkpoints. | `adversarialimage-ids-gpu` |

For Fashion-MNIST checkpoints:

```bash
python scripts/train_fashion_mnist_checkpoint.py --model m2
```

The official M2 CW path is the central runner, which evaluates the configured
`.npy` adversarial examples by default:

```bash
python scripts/run_experiment.py --experiment table_10_m2
```

For M2 CW adversarial regeneration, enable generation through overrides:

```bash
python scripts/run_experiment.py \
  --experiment table_10_m2 \
  --override generation.enabled=true \
  --override generation.overwrite=true \
  --override evaluation.nn_robust_attacks_root=nn_robust_attacks
```

This M2 flow uses `nn_robust_attacks.CarliniL2` for CW-L2. The legacy
`scripts/article_reproduction/mnist_table_10_m2_cw.py` (which also handled CW-Linf
via `nn_robust_attacks.CarliniLi`) is kept only for compatibility.

## Validation

Use the smoke test for a fast check after creating or updating an environment:

```bash
python scripts/dev/smoke_test.py
```

For experiment-specific validation, run the smallest related experiment first and inspect the generated outputs.

## Notes

- Keep public script documentation in English.
- Keep experiment IDs synchronized with `configs/experiments.yaml`.
- Keep environment recommendations synchronized with `envs/README.md`.
- Historical article reproduction scripts can remain for compatibility, but the official path for public experiments is `scripts/run_experiment.py`.
