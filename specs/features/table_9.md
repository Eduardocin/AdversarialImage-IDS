# SPEC - Table 9: Final FGSM Detection Filter

## Objective

Define the project-specific reproduction of Table 9 from the DeepDetector article, evaluating the final entropy-aware detection filter against FGSM adversarial examples.

The Table 9 reproduction must orchestrate two flows:

- MNIST M1 with FGSM `epsilon = 0.2`.
- ImageNet BVLC GoogLeNet/Caffe with FGSM `epsilon_255 = 1.0`.

The final result must aggregate raw counters from both flows by split and then calculate final metrics. It must not average per-flow metrics.

## Context

The article Table 9 reports the final detector against FGSM for `Training` and `Validation`.

Reference values:

| Split | TP | FN | FP | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Training | 3324 | 266 | 108 | 92.59 | 96.85 | 94.67 |
| Validation | 1028 | 61 | 35 | 94.40 | 96.71 | 95.54 |

These values are reference-only. The implementation must calculate local results from the configured datasets and must not hardcode article counts as computed output.

Related project specs:

- `specs/features/Table 4 ImageNet.md`
- `specs/features/table_6_imagenet.md`
- `specs/features/Table 7 ImageNet.md`
- `specs/features/imagenet_clean_baseline_filter_spec.md`

The Table 9 final filter is conceptually:

```text
Table 6 adaptive scalar quantization
+ high-entropy Table 7-style cross smoothing
+ choose the filtered pixel closest to the original pixel
```

## Business Rules

- Use only `Training` and `Validation` splits.
- Do not use the Table 2 `Test` split for this reproduction.
- Use MNIST test-set slices to reproduce the article's training/validation protocol.
- Use local ImageNet class folders already present in this project.
- Use clean-baseline filtering: samples whose clean prediction is wrong are excluded before attack and detection counting.
- Use FGSM failure filtering: samples where FGSM does not change the clean prediction are excluded from TP/FN counting.
- Count FP only for clean-correct and attack-successful pairs, matching the valid adversarial population used for TP/FN.
- Aggregate counters first:

```text
TP_total = sum(TP_i)
FN_total = sum(FN_i)
FP_total = sum(FP_i)
```

- Calculate final recall, precision, and F1 only after aggregation.
- Do not compare the local result by exact equality with the article values.
- Do not implement DeepFool, CW, Table 8, or Table 10 behavior as part of Table 9.

## Dataset

Use the Table 2 split definitions relevant to Table 9:

| Dataset | Training | Validation | Test |
|---|---|---|---|
| MNIST | indices `0..4499` | indices `4500..5499` | indices `5500..9999` |
| ImageNet | goldfish, pineapple, digital_clock | jellyfish | zebra, panda, cab |

Only these Table 9 inputs are in scope:

### MNIST

| Split | Source | Start | End |
|---|---|---:|---:|
| Training | MNIST test set | 0 | 4500 |
| Validation | MNIST test set | 4500 | 5500 |

MNIST model and attack:

```text
model = M1 clean baseline
attack = FGSM
epsilon = 0.2
value range = [0.0, 1.0]
```

### ImageNet

| Split | Class | Synset | Caffe/GoogLeNet label | Path |
|---|---|---|---:|---|
| Training | goldfish | n01443537 | 1 | `data/imagenet/train/goldfish` |
| Training | pineapple | n07753275 | 953 | `data/imagenet/train/pineapple` |
| Training | digital_clock | n03196217 | 530 | `data/imagenet/train/digital_clock` |
| Validation | jellyfish | n01910747 | 107 | `data/imagenet/validation/jellyfish` |

ImageNet model and attack:

```text
model = BVLC GoogLeNet through Caffe
attack = FGSM
epsilon_255 = 1.0
preprocessed input = CHW/BGR/0..255
mean_file = null
```

The ImageNet loader must support JPEG/JPG and PNG files.

## Final Filter

Create:

```text
src/deepdetector/filters/article_final.py
```

Public function:

```python
def article_final_detection_filter(image: np.ndarray) -> np.ndarray:
    ...
```

The function must preserve the input shape and return `float32`.

Entropy and filtering rules:

| Entropy range | Quantization | Smoothing |
|---|---:|---|
| `entropy < 4.0` | 2 intervals, step 128 | none |
| `4.0 <= entropy < 5.0` | 4 intervals, step 64 | none |
| `entropy >= 5.0` | 6 intervals, step 43 | cross 7x7 |

For low and medium entropy:

```text
output = scalar_quantization(image, selected_step)
```

For high entropy:

```text
quantized = scalar_quantization(image, step=43)
smoothed = cross_mean_filter(quantized, radius=3)
output = choose_closer_to_original_pixelwise(original=image, a=quantized, b=smoothed)
```

Pixelwise choose-closer rule:

```python
np.where(
    np.abs(quantized - original) <= np.abs(smoothed - original),
    quantized,
    smoothed,
)
```

The implementation must support both project image scales:

- MNIST normalized images in `[0.0, 1.0]`.
- ImageNet Caffe tensors in `CHW/BGR/0..255`.

For ImageNet Caffe tensors, quantization must operate directly in `0..255` space and entropy must use `image_entropy_255_chw`. The filter must not silently normalize Caffe tensors to `[0,1]` and then return normalized data to the Caffe model path.

Register the filter in:

```text
src/deepdetector/filters/registry.py
```

Expected registry key:

```python
"article_final"
```

Export it from:

```text
src/deepdetector/filters/__init__.py
```

## Configuration

Create:

```text
configs/article_reproduction/table_9.yaml
```

The YAML must be the single source of experiment configuration.

Required structure:

```yaml
experiment:
  name: table_9
  type: faithful_reproduction
  article_table: 9
  seed: 20170830
  objective: Reproduce Table 9 using the final entropy-aware detection filter against FGSM.

orchestration:
  flows:
    - mnist_m1_fgsm
    - imagenet_googlenet_fgsm
  aggregate_by: split
  aggregate_counts_before_metrics: true

splits:
  - Training
  - Validation

datasets:
  mnist:
    enabled: true
    flow: mnist_m1_fgsm
    model: M1
    split_source: test
    image_shape: [28, 28, 1]
    value_range: [0.0, 1.0]
    checkpoint_dir: artifacts/models/mnist/m1/clean_baseline/checkpoints
    attack:
      name: fgsm
      epsilon: 0.2
      clip_min: 0.0
      clip_max: 1.0
    slices:
      Training:
        start: 0
        end: 4500
      Validation:
        start: 4500
        end: 5500

  imagenet:
    enabled: true
    flow: imagenet_googlenet_fgsm
    model: googlenet_caffe
    image_shape: [224, 224, 3]
    value_range: [0.0, 1.0]
    model_assets:
      model_dir: artifacts/models/imagenet/googlenet
      deploy_proto: artifacts/models/imagenet/googlenet/deploy.prototxt
      caffemodel: artifacts/models/imagenet/googlenet/bvlc_googlenet.caffemodel
      mean_file: null
      use_gpu: false
    attack:
      name: fgsm
      epsilon_255: 1.0
      clip_min: 0.0
      clip_max: 255.0
    classes:
      Training:
        - name: goldfish
          label: 1
          path: data/imagenet/train/goldfish
        - name: pineapple
          label: 953
          path: data/imagenet/train/pineapple
        - name: digital_clock
          label: 530
          path: data/imagenet/train/digital_clock
      Validation:
        - name: jellyfish
          label: 107
          path: data/imagenet/validation/jellyfish

detection:
  filter_name: article_final
  method: prediction_change
  entropy_thresholds:
    low: 4.0
    medium: 5.0
  quantization:
    low_entropy:
      intervals: 2
      interval_size: 128
    medium_entropy:
      intervals: 4
      interval_size: 64
    high_entropy:
      intervals: 6
      interval_size: 43
  smoothing:
    enabled_for: high_entropy
    mask: cross
    size: 7
    radius: 3
  high_entropy_combination:
    rule: choose_closer_to_original_pixelwise

evaluation:
  exclude_clean_errors: true
  exclude_failed_attacks: true
  report_partial_results_by_flow: true

metrics:
  final_fields:
    - split
    - TP
    - FN
    - FP
    - recall_percent
    - precision_percent
    - f1_percent

reference:
  Training:
    TP: 3324
    FN: 266
    FP: 108
    recall_percent: 92.59
    precision_percent: 96.85
    f1_percent: 94.67
  Validation:
    TP: 1028
    FN: 61
    FP: 35
    recall_percent: 94.40
    precision_percent: 96.71
    f1_percent: 95.54

output:
  results_dir: results/article_reproduction/table_9
  final_csv: table_9.csv
  markdown: table_9.md
  status_json: status.json
```

## Functional Requirements

### Script

Create:

```text
scripts/article_reproduction/table_9.py
```

Supported commands:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml
```

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --dry-run
```

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --sample-size 8
```

CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/article_reproduction/table_9.yaml` | Table 9 config |
| `--dry-run` | false | Validate config, paths, registry, and sample discovery without full evaluation |
| `--sample-size` | `None` | Optional limit per split for MNIST and per class for ImageNet |
| `--output-dir` | config value | Optional output directory override |

### MNIST Flow

For each MNIST split:

```text
1. Restore M1 clean baseline.
2. Load the configured MNIST test slice.
3. Predict clean images.
4. Generate FGSM with epsilon=0.2.
5. Predict adversarial images.
6. Apply article_final to clean and adversarial images.
7. Predict filtered clean and filtered adversarial images.
8. Exclude clean errors and failed attacks.
9. Return TP, FN, FP and diagnostics for aggregation.
```

### ImageNet Flow

For each ImageNet split/class:

```text
1. Load local class-folder images.
2. Preprocess images for Caffe/GoogLeNet into CHW/BGR/0..255.
3. Predict clean images.
4. Exclude wrong clean predictions before attack generation.
5. Generate FGSM with epsilon_255=1.0.
6. Predict adversarial images.
7. Exclude FGSM failures before TP/FN evaluation.
8. Apply article_final to clean and adversarial Caffe tensors.
9. Predict filtered clean and filtered adversarial images.
10. Return TP, FN, FP and diagnostics for aggregation.
```

### Detection Rule

For a classifier `C`, an image `x`, and filter `T`:

```text
C(x) == C(T(x)) -> benign
C(x) != C(T(x)) -> adversarial
```

Counts:

```text
TP: C(x_adv) != C(T(x_adv))
FN: C(x_adv) == C(T(x_adv))
FP: C(x_clean) != C(T(x_clean))
```

### Aggregation

The aggregator must sum counters by split across enabled flows:

```text
Training = MNIST Training + ImageNet Training
Validation = MNIST Validation + ImageNet Validation
```

Then calculate:

```python
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * recall * precision / (recall + precision)
```

Final CSV values must use percentages:

```python
recall_percent = 100.0 * recall
precision_percent = 100.0 * precision
f1_percent = 100.0 * f1
```

If a denominator is zero, the metric must be `0.0`, and `status.json` must describe the missing valid population.

## Output

Generate:

```text
results/article_reproduction/table_9/table_9.csv
results/article_reproduction/table_9/table_9.md
results/article_reproduction/table_9/status.json
```

Final CSV columns must be exactly:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
```

Rows must be exactly:

```text
Training
Validation
```

The Markdown report must include:

- final aggregated table;
- reference Table 9 values;
- local minus reference deltas for recall, precision, and F1;
- notes about partial execution when `--sample-size` is used;
- notes about skipped flows when optional dependencies or assets are missing.

The status JSON must include:

```text
status
config_path
results_csv
markdown
enabled_flows
completed_flows
skipped_flows
sample_size
per_flow_counters
aggregate_counters
warnings
```

## Dry Run

`--dry-run` must validate:

- YAML can be loaded.
- Required flows are configured.
- Required splits are present.
- `article_final` is registered in `FILTER_REGISTRY`.
- MNIST config has checkpoint path and valid slices.
- A small MNIST sample can be loaded.
- ImageNet class directories are discoverable.
- ImageNet model asset paths are resolved, but Caffe model loading may be reported as blocked instead of failing the dry run.

`--dry-run` must write `status.json` and must not write a misleading final metrics CSV.

## Non-Functional Requirements

- Keep flow orchestration simple and explicit.
- Reuse existing helpers where practical:
  - MNIST article reproduction helpers from `deepdetector.evaluation.article_reproduction`.
  - ImageNet FGSM helpers from `deepdetector.attacks.fgsm_imagenet`.
  - Table 4/6 ImageNet loading and evaluation patterns.
  - Table 7 smoothing semantics where compatible.
- Do not introduce heavy experiment tracking tools.
- Output must be deterministic for the same config, data, and `--sample-size`.
- CSV files must use UTF-8 encoding and comma delimiters.
- Do not write generated datasets, model weights, adversarial dumps, or large artifacts to git-tracked paths.

## Acceptance Criteria

- `configs/article_reproduction/table_9.yaml` exists and declares `mnist_m1_fgsm` and `imagenet_googlenet_fgsm`.
- The YAML uses the Table 2 `Training` and `Validation` splits defined in this spec.
- `src/deepdetector/filters/article_final.py` exists.
- `article_final_detection_filter` implements the entropy-dependent final filter.
- The filter preserves normalized MNIST scale and Caffe ImageNet scale.
- `article_final` is registered in `FILTER_REGISTRY`.
- `article_final_detection_filter` is exported from `deepdetector.filters`.
- `scripts/article_reproduction/table_9.py` exists.
- The script supports `--config`, `--dry-run`, `--sample-size`, and `--output-dir`.
- `--dry-run` writes `status.json`.
- A small `--sample-size 8` run writes `table_9.csv`, `table_9.md`, and `status.json` when required model assets are available.
- The final CSV has exactly the columns `split,TP,FN,FP,recall_percent,precision_percent,f1_percent`.
- The final CSV has exactly two rows: `Training` and `Validation`.
- Metrics are calculated after summing counters, not by averaging per-flow metrics.
- The Markdown report compares local results with Table 9 reference values.
- Missing optional assets, such as Caffe/GoogLeNet assets, are reported in `status.json` without fabricating metrics.
- Automated tests cover:
  - YAML structure;
  - filter threshold behavior;
  - high-entropy choose-closer behavior;
  - filter registry entry;
  - aggregation by summed counters;
  - final CSV shape;
  - dry-run status behavior.

## Error Cases

- Missing config file: fail with the missing path.
- Invalid YAML: fail before running flows.
- Missing required split: fail with the split name.
- Missing MNIST checkpoint: write blocked/partial status and do not report complete metrics.
- Missing ImageNet class directory: fail in full run; report path in dry run.
- Missing Caffe or GoogLeNet assets: write blocked/partial status for the ImageNet flow.
- No clean-correct samples for a flow: report blocked/partial flow status.
- No successful FGSM examples for a flow: report blocked/partial flow status with the same FGSM diagnostic message used by ImageNet Table 4/6.
- No valid adversarial examples in any enabled flow: fail without writing a misleading final CSV.

## Out Of Scope

- Reproducing exact article counts as hardcoded outputs.
- Using the Table 2 `Test` split.
- Implementing Table 8 or Table 10.
- Implementing DeepFool or CW attacks.
- Training new classifiers.
- Changing the physical dataset layout.
- Downloading ImageNet data.
- Saving generated adversarial images as part of the default Table 9 run.
