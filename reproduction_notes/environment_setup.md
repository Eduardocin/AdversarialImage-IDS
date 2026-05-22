# Legacy Environment Setup

## Purpose

The environment is intentionally legacy-oriented to keep the reproduction close
to `OwenSec/DeepDetector` and the Liang et al. adaptive noise reduction method.

## Local Conda Environment

The expected local environment name is:

```bash
adversarialimage-ids-legacy
```

Activate it with:

```bash
conda activate adversarialimage-ids-legacy
```

Create it from scratch with:

```bash
conda env create -f environment.yml
```

Synchronize an existing copy with:

```bash
conda env update -n adversarialimage-ids-legacy -f environment.yml
```

The same pinned Python packages are also listed in `requirements.txt` for
pip-only inspection or emergency repair:

```bash
pip install -r requirements.txt
```

Validate the import surface with:

```bash
python scripts/dev/smoke_test.py
```

## Version Decisions

The baseline dependency pins are:

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

These versions were verified in the local `adversarialimage-ids-legacy`
environment. They preserve compatibility with the TensorFlow 1.x execution
model and the CleverHans APIs used by the original MNIST attack flow.

## Compatibility Notes

- Prefer Python 3.7 for TensorFlow 1.15.x compatibility.
- Keep graph/session style code paths available when implementing models and
  attacks.
- Avoid silently changing preprocessing, model architecture, attack parameters,
  or metrics for convenience.
- If a platform-specific package build forces a version adjustment, document
  the exact package, build source, and reason in this file before running
  experiments.

## Seeds

```text
Seed TensorFlow : tf.set_random_seed(1234)
Seed NumPy      : np.random.RandomState([2017, 8, 30])
Fonte           : código original do repositório de referência
```

## Current Validation Command

```bash
conda run -n adversarialimage-ids-legacy python scripts/dev/smoke_test.py
```
