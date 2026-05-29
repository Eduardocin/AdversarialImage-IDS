# Table 3 Output Shape

## Objective

Adjust the official Table 3 output to compare only the active quantization methods and include filter application time.

## Context

Table 3 is executed through `scripts/run_experiment.py --experiment table_3` and the consolidated `configs/experiments.yaml` entry. The legacy non-uniform quantization implementation should not appear in the official Table 3 output.

## Business Rules

- Table 3 must compare `scalar_quantization` and `nonuniform_quantization`.
- Table 3 must not include `nonuniform_quantization_legacy`.
- Table 3 CSV must not include a `filter_name` column.
- Table 3 CSV and JSON rows must include elapsed filter application time in seconds.
- Timing must not change detector counts or metric formulas.

## Functional Requirements

- `configs/experiments.yaml` must declare only two Table 3 filters.
- Table 3 must enable filter timing explicitly in its config.
- Table 3 must suppress the `filter_name` output column explicitly in its config.
- Other filter-grid experiments must keep their current output schema unless they opt in.

## Acceptance Criteria

- `table_3.filters` has exactly `scalar_quantization` and `nonuniform_quantization`.
- Running Table 3 produces rows with `time_s`.
- Table 3 CSV does not contain `filter_name`.
- Table 4/7/8 filter-grid CSV schemas are unchanged by default.

## Error Cases

- If timing is enabled but no images are loaded, timing should still write a numeric duration for the empty batch.

## Out of Scope

- Removing `nonuniform_quantization_legacy` from the filter factory.
- Changing quantization math.
- Changing metrics.
