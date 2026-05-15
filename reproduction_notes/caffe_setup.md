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

## Setup Placeholder

The exact Caffe build steps are platform-dependent and should be finalized when
the ImageNet track begins. At that point, record:

- Caffe source or package channel.
- CPU or GPU build.
- CUDA/cuDNN versions, if applicable.
- Model prototxt and weight sources.
- Any preprocessing differences from the reference implementation.
