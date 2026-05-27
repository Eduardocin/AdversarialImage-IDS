# SPEC - Table 6 ImageNet: Adaptive Quantization

## Objective

Define the project-specific reproduction of Table 6 for ImageNet using entropy-defined adaptive scalar quantization.

This spec covers only the ImageNet pipeline in this repository. It must reuse the same Caffe/GoogLeNet/FGSM assumptions already defined for Table 4 ImageNet and the clean-baseline filtering rule.

The goal is to validate the adaptive quantization detector on the local ImageNet splits before using the ImageNet pipeline for later tables.

## Context

The original article reports Table 6 values for `Training` and `Validation`:

```text
Training:   TP=3482, FN=370, FP=146
Validation: TP=939,  FN=81,  FP=48
```

Those absolute counts are not expected in this repository because the local ImageNet data are stored as a smaller class subset under:

```text
data/imagenet/train/
data/imagenet/validation/
data/imagenet/test/
```

This spec therefore defines:

```text
Table 6 - ImageNet local adaptive quantization
```

The implementation must reproduce the experiment logic, counters, diagnostics, and output format. It must not try to force the article's absolute counts.

Relevant existing specs:

- `specs/features/Table 4 ImageNet.md`
- `specs/features/imagenet_clean_baseline_filter_spec.md`
- `specs/features/Table 7 ImageNet.md`

## Business Rules

- Use BVLC GoogLeNet through the existing Caffe wrapper.
- Use ImageNet images from the local project dataset.
- Use Caffe-style preprocessed image tensors in `CHW/BGR/0..255` space.
- Do not use TensorFlow, CleverHans, or TensorFlow logits for ImageNet FGSM.
- Do not use a Caffe mean file for this article reproduction path.
- Evaluate only images whose clean prediction matches the configured label.
- Generate FGSM with raw `epsilon_255 = 1.0`.
- A successful adversarial example is one where `adv_pred != clean_pred`.
- Failed attacks are counted as `disturbed_failure` and excluded from TP/FN evaluation.
- Apply only entropy-defined adaptive scalar quantization.
- Do not apply Table 7 spatial smoothing filters.
- Do not compare final counts directly against the original article Table 6.

## Dataset

The default dataset configuration must use these local splits:

```text
train:
  data/imagenet/train/goldfish
  data/imagenet/train/pineapple
  data/imagenet/train/digital_clock

validation:
  data/imagenet/validation/jellyfish
```

Expected class labels:

| Split | Class | Synset | Caffe/GoogLeNet label |
|---|---|---|---:|
| train | goldfish | n01443537 | 1 |
| train | pineapple | n07753275 | 953 |
| train | digital_clock | n03196217 | 530 |
| validation | jellyfish | n01910747 | 107 |

The script must support JPEG/JPG and PNG image files because the local ImageNet folders contain JPEG files.

## Model

Use:

```text
deepdetector.models.imagenet_wrappers.GoogLeNetCaffeWrapper
```

Required behavior:

- load BVLC GoogLeNet from `artifacts/models/imagenet/googlenet/`;
- use `deploy.prototxt`;
- use `bvlc_googlenet.caffemodel`;
- set `mean_file: null`;
- expose prediction for preprocessed Caffe tensors;
- expose gradient support for FGSM.

If Caffe or model assets are missing, the script should fail with the same style of clear setup message used by the existing ImageNet article reproduction scripts.

## Attack

Use FGSM in Caffe tensor space:

```text
epsilon_255 = 1.0
clip_min = 0.0
clip_max = 255.0 for preprocessed Caffe tensors
```

The attack must follow the same helper path as Table 4 ImageNet:

```text
preprocess_caffe_inputs(...)
generate_fgsm_caffe_image(...)
predict_caffe_label(...)
```

Expected attack logic:

```python
clean_pred = predict_caffe_label(model, clean_image)

if clean_pred != true_label:
    skipped_wrong_baseline += 1
    record_diagnostic(skip_reason="wrong_clean_prediction")
    continue

adversarial_image = generate_fgsm_caffe_image(
    model=model,
    image=clean_image,
    label=clean_pred,
    epsilon=1.0,
    clip_min=0.0,
    clip_max=255.0,
)

adv_pred = predict_caffe_label(model, adversarial_image)

if adv_pred == clean_pred:
    disturbed_failure += 1
    record_diagnostic(skip_reason="fgsm_failed_to_change_prediction")
    continue
```

If no successful adversarial example is generated for the whole run, abort with:

```text
FGSM did not generate any successful adversarial example.
Check epsilon scale, preprocessing, gradient sign, model wrapper, or labels.
```

## Adaptive Quantization

Use the entropy strategy associated with Table 6:

| Entropy range | Intervals | Scalar step in `0..255` space |
|---|---:|---:|
| `H < 4.0` | 2 | 128 |
| `4.0 <= H < 5.0` | 4 | 64 |
| `H >= 5.0` | 6 | 43 |

Expected helper behavior:

```python
def adaptive_quantization_step(entropy: float) -> int:
    if entropy < 4.0:
        return 128
    if entropy < 5.0:
        return 64
    return 43
```

The implementation should reuse existing quantization and entropy helpers where practical. If the existing helper is normalized-image specific, the ImageNet Table 6 path must preserve Caffe-scale `0..255` behavior like Table 4 ImageNet does.

## Entropy

For Caffe ImageNet tensors, entropy must be computed with the project helper that supports `CHW/0..255` image data:

```text
deepdetector.filters.entropy.image_entropy_255_chw
```

Entropy is the mean Shannon entropy across channels:

```text
entropy = mean(channel_entropy_1, channel_entropy_2, channel_entropy_3)
```

## Functional Requirements

- Add a runnable script:

```text
scripts/article_reproduction/table_6_imagenet.py
```

- Add a configuration file:

```text
configs/article_reproduction/imagenet_table_6.yaml
```

- The script must accept:

```bash
python scripts/article_reproduction/table_6_imagenet.py \
  --config configs/article_reproduction/imagenet_table_6.yaml \
  --limit 20
```

- CLI arguments must include:

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/article_reproduction/imagenet_table_6.yaml` | Experiment config |
| `--limit` | `None` | Optional per-split debug limit |
| `--epsilon` | value from config, default `1.0` | FGSM epsilon in `0..255` scale |
| `--output-dir` | value from config | Output directory override |

- The config must define:

```yaml
experiment:
  name: imagenet_table_6_adaptive_quantization
dataset:
  name: imagenet
  splits:
    train:
      - name: goldfish
        label: 1
        path: data/imagenet/train/goldfish
      - name: pineapple
        label: 953
        path: data/imagenet/train/pineapple
      - name: digital_clock
        label: 530
        path: data/imagenet/train/digital_clock
    validation:
      - name: jellyfish
        label: 107
        path: data/imagenet/validation/jellyfish
model:
  name: googlenet_caffe
  reference: BVLC GoogLeNet
  mean_file: null
attack:
  name: fgsm
  epsilon_255: 1.0
quantization:
  method: entropy_defined_adaptive_quantization
  entropy_thresholds:
    low: 4.0
    medium: 5.0
  interval_sizes:
    low_entropy: 128
    medium_entropy: 64
    high_entropy: 43
output:
  results_dir: results/imagenet/article_reproduction
  csv: table_6_imagenet.csv
  diagnostics_csv: table_6_imagenet_diagnostics.csv
```

## Experiment Flow

For each split, and for each configured class image:

```text
1. Load the local image file.
2. Preprocess it for Caffe/GoogLeNet into CHW/BGR/0..255.
3. Predict the clean image.
4. If clean_pred != true_label:
      increment skipped_wrong_baseline;
      write diagnostic row;
      skip attack and detector evaluation.
5. Compute entropy_clean.
6. Apply adaptive quantization to the clean image.
7. Predict filtered clean image.
8. Generate FGSM adversarial image.
9. Predict adversarial image.
10. If adv_pred == clean_pred:
      increment disturbed_failure;
      write diagnostic row;
      skip TP/FN evaluation.
11. Compute entropy_adv.
12. Apply adaptive quantization to the adversarial image.
13. Predict filtered adversarial image.
14. Count FP if filtered_clean_pred != clean_pred.
15. Count TP if filtered_adv_pred != adv_pred.
16. Otherwise count FN.
```

FP, TP, and FN must be counted only for valid clean-correct and attack-successful pairs.

## Metrics

Calculate per split:

```python
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * recall * precision / (recall + precision)
```

CSV values must be percentages:

```python
recall_percent = 100.0 * recall
precision_percent = 100.0 * precision
f1_percent = 100.0 * f1
```

If a denominator is zero, the metric value must be `0.0`.

If `TP + FN == 0` for the whole run, the script must fail because no valid adversarial example was evaluated.

## Output

Primary CSV:

```text
results/imagenet/article_reproduction/table_6_imagenet.csv
```

The CSV must have exactly these columns:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
```

Rows must be:

```text
train
validation
```

Diagnostic CSV:

```text
results/imagenet/article_reproduction/table_6_imagenet_diagnostics.csv
```

Diagnostic columns:

```csv
split,image_id,class_name,true_label,clean_pred,filtered_clean_pred,adv_pred,filtered_adv_pred,clean_correct,attack_success,entropy_clean,entropy_adv,clean_quantization_step,adv_quantization_step,is_fp,is_tp,is_fn,skip_reason
```

Allowed `skip_reason` values:

```text
none
wrong_clean_prediction
fgsm_failed_to_change_prediction
```

The script must also print one summary per split:

```text
Split:
Total images:
Clean correct:
Skipped wrong baseline:
FGSM success:
Disturbed failure:
TP:
FN:
FP:
Recall:
Precision:
F1:
```

## Non-Functional Requirements

- Keep the implementation small and consistent with `scripts/article_reproduction/table_4_imagenet.py`.
- Reuse existing helpers from `deepdetector.evaluation.table4_imagenet`, `deepdetector.attacks.fgsm_imagenet`, and `deepdetector.filters` where practical.
- Output must be deterministic for the same input files, config, and optional `--limit`.
- CSV files must use UTF-8 encoding and comma delimiters.
- Do not write generated datasets, adversarial image dumps, model weights, or large experiment artifacts to git-tracked paths.

## Acceptance Criteria

- `configs/article_reproduction/imagenet_table_6.yaml` documents the experiment parameters above.
- `scripts/article_reproduction/table_6_imagenet.py --limit 20` runs when Caffe assets are available.
- The script supports JPEG/JPG and PNG input files.
- The script uses Caffe-scale FGSM with `epsilon_255 = 1.0`.
- The script does not import TensorFlow or CleverHans for ImageNet FGSM.
- Clean baseline failures are skipped before attack generation.
- FGSM failures are skipped before TP/FN evaluation.
- `results/imagenet/article_reproduction/table_6_imagenet.csv` is generated.
- The primary CSV has exactly the columns `split,TP,FN,FP,recall_percent,precision_percent,f1_percent`.
- The primary CSV has exactly two rows: `train` and `validation`.
- `results/imagenet/article_reproduction/table_6_imagenet_diagnostics.csv` is generated.
- Diagnostics include `wrong_clean_prediction` rows when clean baseline failures occur.
- Diagnostics include `fgsm_failed_to_change_prediction` rows when FGSM does not change prediction.
- `total_images > 0` for `train`.
- `total_images > 0` for `validation`.
- `clean_correct > 0` for at least one split.
- At least one successful adversarial example is generated in the whole run.
- `TP + FN > 0` in the whole run.
- Automated tests cover adaptive step selection, metric calculation, CSV shape, diagnostic skip reasons, and zero-successful-attack failure.

## Error Cases

- Missing Caffe or GoogLeNet assets: fail with a clear setup message and do not write misleading metric CSVs.
- Missing dataset directory for a configured split/class: fail with the missing path in the error message.
- Empty configured split: fail before writing final metrics.
- No clean-correct images in the whole run: fail with a diagnostic message.
- No successful adversarial examples in the whole run: fail with the required FGSM diagnostic message.
- Unsupported image extension: skip only if it is not one of JPEG/JPG/PNG; do not treat it as an experiment image.

## Out Of Scope

- Reproducing exact Table 6 article counts.
- Changing the physical dataset layout.
- Downloading or generating ImageNet data.
- Training or modifying GoogLeNet.
- Implementing Table 4, Table 7, Table 8, Table 9, or Table 10 behavior.
- Implementing DeepFool or CW attacks.
- Applying spatial smoothing, `chooseCloserFilter`, or the final combined detector.

## Expected Relationship With Other Tables

Recommended order:

```text
Table 4 ImageNet
-> Table 6 ImageNet adaptive quantization
-> Table 7 ImageNet spatial smoothing
```

Table 4 validates scalar quantization and the Caffe FGSM path.
Table 6 validates entropy-defined adaptive scalar quantization.
Table 7 evaluates spatial smoothing on high-entropy ImageNet adversarial examples.
