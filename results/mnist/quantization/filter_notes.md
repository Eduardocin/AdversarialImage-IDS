# MNIST Quantization Filter Notes

These filters are model-independent NumPy transformations used before the
DeepDetector-style prediction comparison. They accept normalized MNIST images
with values in `[0, 1]`, temporarily operate in the original `0..255` pixel
scale, and return values clipped back to `[0, 1]`.

## Uniform scalar quantization

Scalar quantization maps each pixel to the left edge of a fixed-width interval
in the `0..255` scale. With `left=False`, the value is shifted to the center of
the interval. This reduces the number of possible pixel values and can suppress
small high-frequency adversarial perturbations.

## Non-uniform quantization

Non-uniform quantization estimates a per-image border from a 256-bin histogram.
The border is selected around the point where roughly half of the 784 MNIST
pixels have accumulated. Pixels at or below that border are set to zero, while
brighter pixels are collapsed to the border value.

| filter | parameter | expected effect |
| --- | --- | --- |
| scalar quantization | `interval=128` | strong reduction to coarse intensity levels |
| scalar quantization | `interval=64` | moderate intensity reduction |
| scalar quantization | `interval=43` | weaker reduction used for higher-entropy images |
| centered scalar quantization | `left=False` | places values near interval centers instead of left edges |
| non-uniform quantization | histogram border | collapses low-intensity background and bright strokes using an image-specific threshold |
