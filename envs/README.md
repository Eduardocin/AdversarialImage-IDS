# Conda Environments

The recommended default environment for the current `main` branch is the GPU environment.

Most active workflows, including MNIST-like experiments, ImageNet/Caffe experiments, GoogLeNet, CaffeNet, FGSM, DeepFool, and M2/CW reproduction flows, should run with:

```text
adversarialimage-ids-gpu
```

InceptionV3 workflows use a separate TensorFlow 2 environment in `compat.v1` mode:

```text
adversarialimage-inceptionv3-tf2
```

Do not change pinned dependency versions without updating this file and the related reproduction notes. Environment changes can affect model loading, preprocessing, attack behavior, Caffe compatibility, TensorFlow compatibility, or adversarial attack behavior.

## Supported Environments

| File | Conda env | Main stack | When to use |
| --- | --- | --- | --- |
| `envs/environment-gpu.yml` | `adversarialimage-ids-gpu` | Python 3.7, TensorFlow GPU 1.15, CUDA 10, cuDNN 7, Caffe GPU | Default environment for current MNIST-like and ImageNet/Caffe workflows |
| `envs/inceptionv3-tf2.yml` | `adversarialimage-inceptionv3-tf2` | Python 3.8, TensorFlow 2.11 in `compat.v1` mode | InceptionV3, CW ImageNet, and GPUs where TF1.15 is not compatible |

## Default GPU Environment

Create the default GPU environment:

```bash
conda env create -f envs/environment-gpu.yml
conda activate adversarialimage-ids-gpu
pip install -e .
python scripts/dev/smoke_test.py
```

Use this environment unless the selected experiment explicitly requires the InceptionV3 TensorFlow 2 environment.

## InceptionV3 TensorFlow 2 Environment

Create the InceptionV3 TensorFlow 2 environment:

```bash
conda env create -f envs/inceptionv3-tf2.yml
conda activate adversarialimage-inceptionv3-tf2
pip install -e .
```

Validate the TF2/InceptionV3 environment with this minimal check:

```bash
python - <<'PY'
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

## Experiment Matrix

| Experiment | Recommended environment | Notes |
| --- | --- | --- |
| `table_3` | `adversarialimage-ids-gpu` | MNIST/local quantization. |
| `table_4` | `adversarialimage-ids-gpu` | Composite MNIST + ImageNet. |
| `table_4_mnist` | `adversarialimage-ids-gpu` | MNIST. |
| `table_4_imagenet` | `adversarialimage-ids-gpu` | ImageNet/Caffe. |
| `table_6` | `adversarialimage-ids-gpu` | Composite MNIST + ImageNet. |
| `table_7` | `adversarialimage-ids-gpu` | ImageNet/GoogLeNet/Caffe. |
| `table_8` | `adversarialimage-ids-gpu` | ImageNet/GoogLeNet/Caffe. |
| `table_9` | `adversarialimage-ids-gpu` | Composite MNIST + ImageNet. |
| `table_10_m1` | `adversarialimage-ids-gpu` | MNIST/M1/FGSM. |
| `table_10_m2` | `adversarialimage-ids-gpu` | MNIST/M2/CW. |
| `table_10_googlenet` | `adversarialimage-ids-gpu` | ImageNet/GoogLeNet/Caffe. |
| `table_10_caffenet` | `adversarialimage-ids-gpu` | ImageNet/CaffeNet/Caffe. |
| `table_10_inception_v3` | `adversarialimage-inceptionv3-tf2` | InceptionV3/TF2 `compat.v1`. |
| `defense_aware` | `adversarialimage-ids-gpu` | MNIST/CW defense-aware unless a specific config says otherwise. |
| `topk_detection` | `adversarialimage-ids-gpu` | ImageNet/GoogLeNet top-k investigation. |
| `fashion_mnist_cw_l2_m2` | `adversarialimage-ids-gpu` | Fashion-MNIST/M2/CW. |
| `imagenet_new_classes_fgsm_googlenet` | `adversarialimage-ids-gpu` | ImageNet/GoogLeNet/Caffe. |
| `imagenet_new_classes_deepfool_caffenet` | `adversarialimage-ids-gpu` | ImageNet/CaffeNet/Caffe. |
| `imagenet_new_classes_cw_l2_inception_v3` | `adversarialimage-inceptionv3-tf2` | InceptionV3/TF2 `compat.v1`. |

## Operational Sources

The public operational sources for running this project are:

1. `README.md`
2. `envs/README.md`
3. `configs/experiments.yaml`
4. `scripts/README.md`
5. `reproduction_notes/` when applicable
