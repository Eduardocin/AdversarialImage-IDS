# SPEC - Table 7 ImageNet

## Objective
- Define the expected behavior and output format for reproducing ImageNet
  Table 7: detecting high-entropy FGSM adversarial examples with spatial
  smoothing filters.

## Context
- The source article uses entropy to decide when spatial smoothing should be
  studied for ImageNet samples.
- Table 7 is the filter-search experiment for high-entropy FGSM examples on the
  ImageNet training split with BVLC GoogLeNet/Caffe.
- The experiment compares cross, diamond, and box spatial smoothing masks at
  sizes 3x3, 5x5, 7x7, and 9x9.
- Table 7 evaluates smoothing candidates only. Scalar quantization and the
  final adaptive filter belong to later tables.
- The output should keep the article-style pivot layout used for reporting
  Recall, Precision, and F1 Score.

## Business rules
- The official execution path is:
  `python scripts/run_experiment.py --experiment table_7`.
- The `table_7` entry in `configs/experiments.yaml` uses the ImageNet/Caffe
  Table 7 runner, not an MNIST filter-grid runner.
- Only images correctly classified before the attack are eligible for FGSM
  attack evaluation.
- If the FGSM adversarial image does not change the clean prediction, the sample
  is a disturbed failure and must not contribute to TP, FN, or FP.
- The high-entropy selection must be based on the clean/original image entropy,
  not on the adversarial image entropy.
- The default high-entropy threshold is `5.0`; a config value may override it,
  but missing config must fall back to `5.0`.
- A sample is eligible for Table 7 detector evaluation only when
  `clean_entropy > entropy_threshold`.
- The adversarial image entropy is diagnostic only. It must be counted for
  analysis, but it must not decide inclusion or exclusion.
- The Table 7 filter must apply only spatial smoothing. It must not call scalar
  quantization, adaptive quantization, or final-filter selection logic.
- The experiment writes only one CSV output for Table 7 results.
- The CSV file name is `table_7_imagnet.csv`.

## Functional requirements
- The experiment must load the configured ImageNet training split and reject
  split-path mismatches through the shared ImageNet validation helpers.
- The experiment must use BVLC GoogLeNet through the Caffe wrapper.
- The FGSM attack must use the shared ImageNet Caffe implementation in
  preprocessed CHW/BGR/0-255 space with `epsilon_255 = 1.0` by default.
- The experiment must support compatible pre-generated adversarial arrays when
  configured.
- The configured filter grid must evaluate exactly these 12 combinations unless
  a future spec changes the Table 7 search space:
  - `cross_3x3`
  - `cross_5x5`
  - `cross_7x7`
  - `cross_9x9`
  - `diamond_3x3`
  - `diamond_5x5`
  - `diamond_7x7`
  - `diamond_9x9`
  - `box_3x3`
  - `box_5x5`
  - `box_7x7`
  - `box_9x9`
- For each mask/size pair, the evaluator must compute these counters:
  - `total_images`
  - `clean_correct`
  - `skipped_wrong_baseline`
  - `attack_success`
  - `disturbed_failure`
  - `skipped_low_entropy_clean`
  - `n_high_entropy_clean`
  - `n_high_entropy_adversarial`
  - `tp`
  - `fn`
  - `fp`
  - `recall`
  - `precision`
  - `f1`
- Counter definitions:
  - `skipped_wrong_baseline`: clean image classified incorrectly before attack.
  - `attack_success`: clean-correct FGSM image whose prediction changed.
  - `disturbed_failure`: FGSM image whose prediction did not change.
  - `skipped_low_entropy_clean`: clean-correct, attack-success sample skipped
    because `clean_entropy <= entropy_threshold`.
  - `n_high_entropy_clean`: attack-success samples evaluated because
    `clean_entropy > entropy_threshold`.
  - `n_high_entropy_adversarial`: evaluated samples whose adversarial image
    entropy is also greater than the threshold; this is diagnostic only.
  - `TP`: adversarial prediction changes after the smoothing filter.
  - `FN`: adversarial prediction remains unchanged after the smoothing filter.
  - `FP`: clean prediction changes after the smoothing filter.
- Metrics must use zero-safe division:
  - `recall = tp / (tp + fn)`
  - `precision = tp / (tp + fp)`
  - `f1 = 2 * recall * precision / (recall + precision)`
- The script writes a pivot CSV with columns:
  `metric, cross_3x3, cross_5x5, cross_7x7, cross_9x9, diamond_3x3,
  diamond_5x5, diamond_7x7, diamond_9x9, box_3x3, box_5x5, box_7x7,
  box_9x9`.
- The pivot CSV must contain exactly these metric rows:
  - `Recall`
  - `Precision`
  - `F1 Score`
- The output directory is configurable via the experiment config.
- The status JSON must include the pivot CSV path on completed runs.
- The status JSON must include enough counters to diagnose sample selection:
  `total_images`, `clean_correct`, `skipped_wrong_baseline`,
  `attack_success`, `disturbed_failure`, `skipped_low_entropy_clean`,
  `n_high_entropy_clean`, and `n_high_entropy_adversarial`.

## Non-functional requirements
- Output must be deterministic for the same inputs and configuration.
- CSV and JSON files must use UTF-8 encoding.
- CSV files must use standard comma delimiters.
- The implementation should reuse existing ImageNet loading, Caffe
  preprocessing, clean-baseline filtering, FGSM cache, and spatial smoothing
  helpers where possible.
- The implementation must avoid writing generated datasets, model weights, or
  large local experiment artifacts into git-tracked paths.

## Acceptance criteria
- Running `python scripts/run_experiment.py --experiment table_7` dispatches to
  the ImageNet/Caffe Table 7 flow.
- Only one Table 7 CSV is written and its filename is `table_7_imagnet.csv`.
- The CSV has exactly three metric rows: `Recall`, `Precision`, `F1 Score`.
- The CSV has one metric column for each of the 12 Table 7 mask/size
  combinations.
- The evaluator filters high-entropy examples using clean image entropy.
- A sample with `clean_entropy <= 5.0` and `adversarial_entropy > 5.0` is
  skipped and increments `skipped_low_entropy_clean`.
- A sample with `clean_entropy > 5.0` and `adversarial_entropy <= 5.0` is
  evaluated normally and does not increment `n_high_entropy_adversarial`.
- The default entropy threshold is `5.0` when config omits the value.
- Adversarial entropy is used only as a diagnostic counter.
- Table 7 applies only spatial smoothing and does not call quantization helpers.
- Wrong clean-baseline predictions are skipped before attack evaluation.
- Disturbed failures are excluded from TP, FN, and FP.
- TP, FN, FP, Recall, Precision, and F1 Score use the definitions in this spec.
- The status JSON reports the pivot CSV path for completed runs.
- Automated tests cover clean-entropy selection, adversarial-entropy diagnostic
  behavior, default threshold, metric formulas, no-quantization behavior, the
  12 configured filter combinations, pivot CSV shape, clean-baseline filtering,
  and disturbed-failure exclusion.

## Error cases
- If Caffe or GoogLeNet assets are unavailable, the experiment must write a
  partial or blocked status and exit without producing misleading metrics.
- If no training images are loaded, the experiment must write a partial status.
- If no clean image is correctly classified before attack generation, the
  experiment must write a partial status.
- If FGSM cannot produce or load adversarial examples, the experiment must write
  a partial status with a diagnostic message.
- If the adversarial array shape is incompatible with the selected images, the
  experiment must fail explicitly.
- If no successful adversarial high-entropy examples are available for
  evaluation, the experiment must not report completed Table 7 metrics.

## Out of scope
- Changing the 12 Table 7 smoothing mask definitions.
- Selecting the five Table 8 filters dynamically from local Table 7 results.
- Applying scalar quantization, adaptive quantization, or the final Table 9
  detection filter.
- Adding raw per-image or per-filter CSV outputs unless a future spec requires
  it.
- Changing ImageNet labels, moving dataset files, or deleting images.
