SPEC - Table 7 ImageNet

Objective
- Define the expected output format for the ImageNet Table 7 reproduction.

Context
- The ImageNet Table 7 experiment evaluates spatial smoothing filters for high-entropy FGSM examples.
- The output should match the article-style pivot layout used for reporting metrics.

Business rules
- The experiment writes only one CSV output for Table 7 results.
- The CSV contains metric rows (Recall, Precision, F1 Score) and columns for each mask/size pair.
- The CSV file name is table_7_imagnet.csv.

Functional requirements
- The Table 7 script writes a pivot CSV with columns:
  metric, cross_3x3, cross_5x5, cross_7x7, cross_9x9,
  diamond_3x3, diamond_5x5, diamond_7x7, diamond_9x9,
  box_3x3, box_5x5, box_7x7, box_9x9.
- The output directory is configurable via the experiment config.
- The status JSON includes the pivot CSV path.

Non-functional requirements
- Output must be deterministic for the same inputs and configuration.
- The CSV uses UTF-8 encoding and standard comma delimiters.

Acceptance criteria
- Only one Table 7 CSV is written and its filename is table_7_imagnet.csv.
- The CSV has exactly three metric rows: Recall, Precision, F1 Score.
- The status JSON reports the pivot CSV path.

Error cases
- If required data are missing, the script should exit with a partial status as before.

Out of scope
- Changing the filter definitions or metric calculations.
- Adding raw per-filter CSV outputs for Table 7.
