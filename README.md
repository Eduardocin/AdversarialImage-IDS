# AdversarialImage-IDS / DeepDetector

Reimplementation and experimental extension of the adversarial image detection method proposed by Liang et al. in *Detecting Adversarial Image Examples in Deep Neural Networks with Adaptive Noise Reduction*.

The project uses `OwenSec/DeepDetector` as a methodological reference, but the current codebase is organized as a Python package with YAML-driven experiments, centralized execution, reproducible outputs, and additional experimental workflows.

## Project Scope

This repository focuses on reproducing and extending adversarial example detection experiments for MNIST-like and ImageNet datasets.

The current scope includes:

- reproduction of selected tables from the reference paper;
- evaluation of FGSM, DeepFool, Carlini-Wagner L2/Linf, and defense-aware attacks;
- adaptive noise-reduction filters based on entropy, scalar quantization, and spatial smoothing;
- centralized experiment configuration through YAML files;
- structured outputs for metrics, manifests, and reproduction checks;
- experimental extensions with Fashion-MNIST, new ImageNet classes, and top-k based detection rules.

The top-k workflow is currently treated as an experimental investigation, especially for ambiguous ImageNet samples. It should not yet be described as a finalized improvement over the reference method.

## Method Summary

The original DeepDetector method treats adversarial perturbations as a form of image noise. A detection filter is applied to the input image before classification. The system then compares the classifier prediction on the original sample with the prediction on the filtered sample:

```text
x -> C(x)
x -> T(x) -> C(T(x))

if C(x) == C(T(x)):
    sample is considered benign
else:
    sample is considered adversarial
```

The filter `T` combines:

- image entropy estimation;
- scalar quantization;
- spatial smoothing;
- adaptive filter selection according to image complexity.

The goal of this project is to reproduce this behavior as faithfully as possible while making the code easier to run, inspect, and extend.

## Operational Sources

The public operational sources for this repository are:

1. `README.md` — project overview and main execution guide.
2. `envs/README.md` — supported Conda environments and experiment-to-environment matrix.
3. `configs/experiments.yaml` — official experiment registry.
4. `scripts/README.md` — script-level execution notes.
5. `reproduction_notes/` — technical notes for reproduction-specific cases.

Official experiments should be declared in `configs/experiments.yaml` and executed through:

```bash
python scripts/run_experiment.py --experiment <experiment_id>
```

## Repository Layout

```text
.
|-- src/deepdetector/          # Main Python package
|   |-- attacks/               # FGSM, DeepFool, CW, and adaptive attacks
|   |-- data/                  # MNIST, Fashion-MNIST, and ImageNet loaders
|   |-- detection/             # Detection rules based on prediction changes
|   |-- evaluation/            # Metrics and table-level evaluation logic
|   |-- experiments/           # Configurable experiment runners
|   |-- filters/               # Noise-reduction filters
|   |-- io/                    # YAML loading, paths, and output writing
|   |-- models/                # MNIST-like models and ImageNet wrappers
|   `-- training/              # Training and checkpoint restoration helpers
|-- configs/
|   |-- experiments.yaml       # Official experiment inventory
|   `-- article_reproduction/  # Preserved auxiliary configs
|-- scripts/
|   |-- run_experiment.py      # Official experiment entry point
|   |-- train_fashion_mnist_checkpoint.py
|   |-- article_reproduction/  # Reproduction-specific helpers, mainly M2/CW
|   |-- dev/                   # Smoke tests and local validation scripts
|   `-- imagenet/              # ImageNet/Caffe asset preparation
|-- envs/                      # Versioned Conda environments
|-- reproduction_notes/        # Reproduction notes and technical decisions
|-- data/                      # Local datasets, not versioned
|-- artifacts/                 # Models, checkpoints, attack caches; not versioned
`-- results/                   # Regenerable experiment outputs
```

## Environment Strategy

The recommended default environment for the current main branch is the GPU environment:

```bash
conda env create -f envs/environment-gpu.yml
conda activate adversarialimage-ids-gpu
pip install -e .
python scripts/dev/smoke_test.py
```

Use this environment for MNIST-like experiments, ImageNet/Caffe workflows, GoogLeNet, CaffeNet, FGSM, DeepFool, and the M2/CW reproduction workflows unless a specific experiment says otherwise.

InceptionV3 experiments use a separate TensorFlow 2 environment in `compat.v1` mode:

```bash
conda env create -f envs/inceptionv3-tf2.yml
conda activate adversarialimage-inceptionv3-tf2
pip install -e .
python - <<'PY'
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

For the complete environment matrix, use `envs/README.md`.

## Running Experiments

The official entry point is:

```bash
python scripts/run_experiment.py --experiment <experiment_id>
```

Examples:

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
python scripts/run_experiment.py --experiment fashion_mnist_cw_l2_m2
python scripts/run_experiment.py --experiment imagenet_new_classes_fgsm_googlenet
python scripts/run_experiment.py --experiment imagenet_new_classes_deepfool_caffenet
python scripts/run_experiment.py --experiment imagenet_new_classes_cw_l2_inception_v3
```

`table_4` is a composite experiment that executes both:

```bash
python scripts/run_experiment.py --experiment table_4_mnist
python scripts/run_experiment.py --experiment table_4_imagenet
```

Table 5 is not part of the official operational path because the current repository inventory does not define an official script, config, or output contract for it.

## Main Experiment Groups

| Group | Experiments | Recommended environment |
| --- | --- | --- |
| MNIST/local | `table_3`, `table_4_mnist`, `table_10_m1`, `table_10_m2`, `defense_aware` | `adversarialimage-ids-gpu` |
| ImageNet/Caffe | `table_4_imagenet`, `table_7`, `table_8`, `table_10_googlenet`, `table_10_caffenet` | `adversarialimage-ids-gpu` |
| Composite MNIST + ImageNet | `table_4`, `table_6`, `table_9` | `adversarialimage-ids-gpu` |
| InceptionV3 | `table_10_inception_v3`, `imagenet_new_classes_cw_l2_inception_v3` | `adversarialimage-inceptionv3-tf2` |
| New datasets/classes | `fashion_mnist_cw_l2_m2`, selected ImageNet new-class experiments | See `envs/README.md` |
| Top-k investigation | `topk_detection` | Depends on the configured dataset |

Some rows inside an experiment group may be marked as `planned` in `configs/experiments.yaml`. Always check the YAML status before claiming full reproduction coverage for a table row.

## Data and Artifacts

Datasets, checkpoints, Caffe models, TensorFlow graphs, and generated adversarial examples should not be committed to Git.

Main local paths:

| Path | Purpose |
| --- | --- |
| `data/` | Local datasets such as MNIST/Fashion-MNIST CSV files and ImageNet folders |
| `artifacts/models/` | Model checkpoints and external model assets |
| `artifacts/adversarial_examples/` | Cached adversarial examples |
| `results/` | Regenerable metrics, manifests, and reports |

### Caffe Assets

To list and download supported Caffe assets:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --list-models
python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet
python scripts/imagenet/download_caffe_imagenet_assets.py --model alexnet
```

Assets not handled by the downloader, such as some CaffeNet files required by reproduction workflows, should follow the instructions in:

```text
reproduction_notes/caffe_setup.md
```

### InceptionV3 Subset

To materialize the local subset used by InceptionV3 workflows:

```bash
python scripts/imagenet/materialize_inceptionv3_subset.py
```

## Fashion-MNIST Workflow

Before running the Fashion-MNIST CW-L2 experiment, prepare the configured M2 checkpoint:

```bash
python scripts/train_fashion_mnist_checkpoint.py --model m2
```

Then run:

```bash
python scripts/run_experiment.py --experiment fashion_mnist_cw_l2_m2
```

## Table 10 M2 / CW Workflow

The official M2/CW path is the central runner:

```bash
python scripts/run_experiment.py --experiment table_10_m2
```

By default this command evaluates the configured `.npy` adversarial examples
(MNIST slice `start=5500`, `samples=1000`, matching the artifacts' `manifest.json`)
and writes `metrics.csv`, `metrics.json`, and `manifest.json` under
`results/table_10/M2_cw/`.

To regenerate the adversarial examples (depends on the original
`nn_robust_attacks` backend), enable generation through overrides:

```bash
python scripts/run_experiment.py \
  --experiment table_10_m2 \
  --override generation.enabled=true \
  --override generation.overwrite=true \
  --override evaluation.nn_robust_attacks_root=nn_robust_attacks
```

The legacy `scripts/article_reproduction/mnist_table_10_m2_cw.py` is superseded by
this central path and is kept only for compatibility.

## Outputs

Most experiments write:

```text
results/<experiment_id>/metrics.csv
results/<experiment_id>/metrics.json
```

Some workflows have specific output contracts:

| Experiment | Main output |
| --- | --- |
| `table_4` | `results/table_4/mnist/`, `results/table_4/imagenet/`, and a root manifest |
| `table_7` | `table_7_imagenet.csv` and `table_7_status.json` |
| `table_8` | `table_8_imagenet.csv` and `table_8_status.json` |
| `table_10_*` | `metrics.csv`, `metrics.json`, and, when applicable, `manifest.json` |
| `topk_detection` | selected ambiguous samples, top-k metrics, and aggregate summaries |
| `fashion_mnist_cw_l2_m2` | metrics and manifest under `results/experiments/fashion_mnist/` |
| `imagenet_new_classes_*` | metrics and manifest under `results/experiments/imagenet_new_classes/` |

## Validation

Use the smoke test for a fast import and dependency check:

```bash
python scripts/dev/smoke_test.py
```

For experiment-specific validation, prefer running the smallest relevant experiment first and checking the generated `metrics.csv`, `metrics.json`, and `manifest.json` files when available.

## Development Notes

- Keep public documentation in English.
- Keep experiment IDs synchronized with `configs/experiments.yaml`.
- Keep environment recommendations synchronized with `envs/README.md`.
- Avoid committing generated datasets, checkpoints, adversarial examples, or result files unless explicitly required.
- Treat top-k detection as an experimental investigation until results and implementation are finalized.
- Prefer `scripts/run_experiment.py` for public experiment execution.
- Historical scripts can remain for compatibility, but they should not replace the official YAML-driven runner.

## Reference

This project is based on the method introduced in:

Bin Liang, Hongcheng Li, Miaoqiang Su, Xirong Li, Wenchang Shi, and Xiaofeng Wang.  
*Detecting Adversarial Image Examples in Deep Neural Networks with Adaptive Noise Reduction.*
