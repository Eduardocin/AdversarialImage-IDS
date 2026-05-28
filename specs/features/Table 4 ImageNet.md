SPEC - Table 4 ImageNet

## Objective

Implement and validate the ImageNet portion of Table 4 from the article.

Table 4 evaluates scalar uniform quantization with interval counts from 2 to
10. For ImageNet, the experiment checks whether quantization detects FGSM
adversarial examples by comparing the adversarial prediction before and after
the filter.

## Experimental Reference

- Dataset: local ImageNet subset.
- Split: training.
- Classes: goldfish, pineapple, digital clock.
- Model: BVLC GoogLeNet through Caffe.
- Attack: FGSM.
- Epsilon: `epsilon_255 = 1.0` in Caffe `0..255` input scale.
- Preprocess: RGB to CHW/BGR, raw scale 255, no mean subtraction.
- Filter: scalar uniform quantization.
- Intervals: 2, 3, 4, 5, 6, 7, 8, 9, 10.
- Metrics: recall, precision, F1.

## Inputs

The official runner is:

```bash
python scripts/run_experiment.py --experiment table_4_imagenet
```

The deprecated wrapper may still be used for targeted local runs:

```bash
python scripts/article_reproduction/table_4_imagenet.py \
  --data-root data/imagenet/train \
  --limit 50
```

## Expected Data

Recommended local layout:

```text
data/imagenet/train/
  goldfish/
  pineapple/
  digital_clock/
```

Expected labels:

| Class | ImageNet label |
| --- | --- |
| Goldfish | 1 |
| Pineapple | 953 |
| Digital clock | 530 |

## Outputs

Official outputs go to:

```text
results/experiments/table_4/imagenet/
```

The runtime writes:

```text
table_4_imagenet.csv
table_4_status.json
```

It must not write `table_4_imagenet_diagnostics.csv`.

The CSV format is:

```text
Dataset,Metric,2,3,4,5,6,7,8,9,10
ImageNet,Recall,<values>
ImageNet,Precision,<values>
ImageNet,F1 Score,<values>
```

## Business Rules

For each image:

1. Load local PNG/JPEG data.
2. Apply Caffe/GoogLeNet preprocessing: CHW, BGR, raw scale 255, no mean file.
3. Predict the clean image.
4. If `clean_pred != true_label`, increment `skipped_wrong_baseline` and skip
   the attack for that image.
5. Generate FGSM using Caffe gradients with `epsilon_255 = 1.0`.
6. Predict the adversarial image.
7. If `adv_pred == clean_pred`, increment `disturbed_failure` and skip detection
   evaluation for that image.
8. For each interval count `k` from 2 to 10:
   - quantize the clean image;
   - quantize the adversarial image;
   - if `C(x_clean) != C(Q(x_clean))`, increment FP;
   - if `C(x_adv) != C(Q(x_adv))`, increment TP;
   - otherwise increment FN.
9. Compute recall, precision, and F1 with zero-safe division.

## GPU Rule

`model.use_gpu: true` means the Caffe wrapper calls `caffe.set_mode_gpu()`.
There is no PyTorch-style `.to("cuda")` path for this model. GPU execution also
requires a Caffe build with CUDA support and a working CUDA runtime.

## Acceptance Criteria

- The official config lives in `configs/experiments.yaml`.
- The legacy `configs/article_reproduction/imagenet_table_4.yaml` file is not
  required.
- The model loads through the Caffe wrapper when Caffe/assets are available.
- The runner reads local ImageNet class folders.
- The result CSV has 3 metric rows and interval columns from 2 to 10.
- Interval 6 appears in the result.
- The runtime writes `table_4_status.json`.
- The runtime does not write `table_4_imagenet_diagnostics.csv`.
- If `attack_success = 0`, the run fails with a clear message.

Expected zero-attack message:

```text
FGSM did not generate any successful adversarial example.
Check epsilon scale, preprocessing, gradient sign, model wrapper, or labels.
```

## Out of Scope

- Changing FGSM math.
- Changing scalar quantization math.
- Generating per-image diagnostics by default.
- Changing Table 10.
