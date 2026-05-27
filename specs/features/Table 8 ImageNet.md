SPEC - Table 8 ImageNet

Objective
- Define the expected behavior and output format for reproducing ImageNet Table 8:
  "Performance of the five superior spatial smoothing filters with the validation set."

Context
- Table 8 validates the best spatial smoothing filter candidates selected from the Table 7 ImageNet training experiment.
- The experiment uses the ImageNet validation split, not the Table 7 training split.
- The validation split must not include classes or paths assigned to the ImageNet
  test split.
- The detector compares model predictions before and after applying one spatial smoothing filter to clean and successful FGSM adversarial examples.
- The ImageNet path must stay consistent with the existing GoogLeNet/Caffe
  reproduction flow used by Table 4 and Table 7.

Business rules
- The five Table 8 filters are fixed from the article and must not be selected
  dynamically from local Table 7 results.
- The five filters are:
  - cross_5x5
  - cross_7x7
  - diamond_5x5
  - diamond_7x7
  - box_5x5
- Only validation images that are correctly classified before the attack are
  eligible for attack and detector evaluation.
- Validation configuration must fail explicitly if a class path points to the
  test split.
- If an adversarial image does not change the original clean prediction, it is a disturbed failure and must not contribute to TP, FN, or FP.
- Metrics use the same detector semantics as Table 7:
  - TP: adversarial prediction changes after the filter.
  - FN: adversarial prediction does not change after the filter.
  - FP: clean prediction changes after the filter.
- Recall, Precision, and F1 Score must use zero-safe division.

Functional requirements
- Add a Table 8 ImageNet experiment that loads validation images from the
  configured ImageNet validation split.
- The validation split must be configurable and should support the project
  class-folder layout used by ImageNet reproduction configs.
- The experiment must use BVLC GoogLeNet through the Caffe wrapper.
- The FGSM attack must use the shared ImageNet Caffe implementation in
  preprocessed CHW/BGR/0-255 space with epsilon_255 = 1.0 by default.
- The experiment must support pre-generated adversarial arrays when configured
  or passed by CLI, using the same compatibility behavior as Table 7.
- The experiment must evaluate only the five fixed Table 8 filters.
- The output directory must be configurable.
- The main CSV output must be a pivot CSV named table_8_imagenet.csv.
- The pivot CSV columns must be:
  metric, cross_5x5, cross_7x7, diamond_5x5, diamond_7x7, box_5x5.
- The pivot CSV must contain exactly these metric rows:
  - Recall
  - Precision
  - F1 Score
- The status JSON must include:
  - status
  - pivot_csv path when the run completes
  - total_images
  - clean_correct
  - skipped_wrong_baseline
  - attack_success
  - disturbed_failure

Non-functional requirements
- Output must be deterministic for the same inputs and configuration.
- CSV files must use UTF-8 encoding and standard comma delimiters.
- The implementation should reuse existing Table 7 spatial smoothing and metric helpers where possible.
- The implementation must not duplicate model loading, preprocessing, or FGSM
  logic when an existing shared helper is available.
- The experiment must avoid writing generated datasets, model weights, or large local experiment artifacts into git-tracked paths.

Acceptance criteria
- A Table 8 ImageNet feature has a config-driven experiment path.
- The experiment reads validation split images, not Table 7 training split images.
- The experiment does not read ImageNet test split paths for Table 8 validation.
- The experiment evaluates exactly five filters: cross_5x5, cross_7x7,
  diamond_5x5, diamond_7x7, and box_5x5.
- The experiment filters out images with clean_pred != true_label before attack evaluation.
- The experiment excludes disturbed failures from TP, FN, and FP.
- The output CSV is named table_8_imagenet.csv.
- The output CSV has exactly three metric rows: Recall, Precision, F1 Score.
- The output CSV has one metric column for each of the five fixed filters.
- The status JSON reports the pivot CSV path for completed runs.
- Automated tests cover the fixed filter set, pivot CSV shape, validation split configuration, clean-baseline filtering, and partial-status behavior.

Error cases
- If Caffe or GoogLeNet assets are unavailable, the experiment must write a
  partial status and exit without producing misleading metrics.
- If no validation images are loaded, the experiment must write a partial status.
- If the validation config points to a test split path, the experiment must fail
  explicitly instead of evaluating that class.
- If no validation image is correctly classified before the attack, the
  experiment must write a partial status.
- If FGSM cannot produce or load adversarial examples, the experiment must write a partial status with a diagnostic message.
- If the adversarial array shape is incompatible with the selected validation
  images, the experiment must fail explicitly.
- If no successful adversarial examples are available for evaluation, the
  experiment must not report completed Table 8 metrics.

Out of scope
- Re-selecting the five filters based on local Table 7 results.
- Changing Table 7 filter definitions or metric calculations.
- Implementing the final adaptive filtering rule from later tables.
- Adding raw per-image or per-filter CSV outputs unless a future spec requires it.
- Changing ImageNet labels, moving dataset files, or deleting images.
