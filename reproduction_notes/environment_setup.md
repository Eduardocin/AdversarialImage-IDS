# Legacy Environment Setup

## Purpose

This note records legacy environment decisions and compatibility constraints.
It is not the primary operational environment matrix. Use `envs/README.md` for
the official experiment -> environment mapping and setup commands.

The dependency stack is intentionally legacy-oriented to keep the reproduction
close to `OwenSec/DeepDetector` and the Liang et al. adaptive noise reduction
method.

## Legacy Local Conda Environment

The expected local environment name is:

```bash
adversarialimage-ids-legacy
```

Create it from scratch with:

```bash
conda env create -f envs/environment.yml
conda activate adversarialimage-ids-legacy
pip install -e .
python scripts/dev/smoke_test.py
```

Synchronize an existing copy with:

```bash
conda env update -n adversarialimage-ids-legacy -f envs/environment.yml
conda activate adversarialimage-ids-legacy
pip install -e .
python scripts/dev/smoke_test.py
```

The same pinned Python packages are also listed in `requirements.txt` for
pip-only inspection or emergency repair:

```bash
pip install -r requirements.txt
```

## GPU and InceptionV3 Environments

ImageNet/Caffe experiments that require GPU execution should use
`envs/environment-gpu.yml` (`adversarialimage-ids-gpu`).

InceptionV3 experiments use `envs/inceptionv3-tf2.yml`
(`adversarialimage-inceptionv3-tf2`) with TensorFlow 2.11 in `compat.v1` mode.
This is a targeted compatibility environment, not a broad migration of the
project to TensorFlow 2.

## Version Decisions

The baseline legacy dependency pins are:

```text
tensorflow==1.15.5
keras<2.0
cleverhans==3.1.0
numpy==1.18.5
scipy==1.5.4
matplotlib==3.3.4
pillow==8.4.0
pyyaml==5.4.1
tqdm==4.64.1
pandas==1.1.5
h5py==2.10.0
protobuf==3.19.6
caffe==1.0
```

These versions preserve compatibility with the TensorFlow 1.x execution model,
legacy Keras APIs, CleverHans flows, and the original Caffe-based ImageNet
paths.

## Compatibility Notes

- Prefer Python 3.7 for TensorFlow 1.15.x compatibility.
- Keep graph/session style code paths available when implementing models and
  attacks.
- Avoid silently changing preprocessing, model architecture, attack parameters,
  or metrics for convenience.
- If a platform-specific package build forces a version adjustment, document
  the exact package, build source, and reason here and in `envs/README.md`
  before running experiments.

## Seeds

```text
Seed TensorFlow : tf.set_random_seed(1234)
Seed NumPy      : np.random.RandomState([2017, 8, 30])
Source          : original reference repository code
```

## Current Validation Commands

```bash
conda run -n adversarialimage-ids-legacy python scripts/dev/smoke_test.py
conda run -n adversarialimage-ids-legacy pytest tests/test_experiment_runner.py
```

When the TF2/InceptionV3 environment is available:

```bash
conda run -n adversarialimage-inceptionv3-tf2 python - <<'PY'
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
```
