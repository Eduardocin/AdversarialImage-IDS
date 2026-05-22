# Caffe Setup Notes

## When Caffe Is Needed

Caffe is required for the ImageNet reproduction path if the team chooses to
preserve the original DeepDetector setup based on CaffeNet and GoogLeNet.

The MNIST path can start with TensorFlow 1.x, Keras, and CleverHans. The
ImageNet path should not be treated as equivalent unless the Caffe-based model
loading, preprocessing, labels, and evaluation flow are reproduced or the
deviation is explicitly documented.

## Expected Role In The Reproduction

- Load CaffeNet and GoogLeNet-compatible model definitions and weights.
- Preserve ImageNet preprocessing used by the original implementation.
- Generate and evaluate adversarial examples against the same model family used
  in the reference project.
- Compare detector behavior before and after adaptive noise reduction.

## Conda Setup

The ImageNet track expects the `caffe` Python module from the conda-forge Caffe
package. It is declared in `environment.yml` as a Conda dependency because
`pycaffe` is not installed through `pip`.

Create or update the local environment with:

```bash
conda env create -f environment.yml
conda activate adversarialimage-ids-legacy
python scripts/dev/smoke_test.py
```

For an existing environment:

```bash
conda env update -n adversarialimage-ids-legacy -f environment.yml
conda activate adversarialimage-ids-legacy
python scripts/dev/smoke_test.py
```

The expected package source is `conda-forge::caffe`. The current reproduction is
configured for CPU execution by default (`use_gpu: false` in the ImageNet YAMLs).
If a GPU build is used later, record the CUDA/cuDNN versions and any package
channel changes here before running experiments.

Model assets are still local artifacts and must exist at the configured paths:

- `artifacts/models/imagenet/googlenet/deploy.prototxt`
- `artifacts/models/imagenet/googlenet/bvlc_googlenet.caffemodel`
- `artifacts/models/imagenet/googlenet/ilsvrc_2012_mean.npy`

Download them with:

```bash
python scripts/imagenet/download_googlenet_assets.py
```

The model definition comes from `BVLC/caffe` at
`models/bvlc_googlenet/deploy.prototxt`. The pretrained weights come from the
model URL documented in that directory's `readme.md`, with SHA1
`405fc5acd08a3bb12de8ee5e23a96bec22f08204`.
