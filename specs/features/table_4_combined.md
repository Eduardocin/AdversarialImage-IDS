# SPEC - Table 4 combined execution

## Objective

Make Table 4 executable as one article table while preserving its two distinct
experimental populations: MNIST and ImageNet.

## Context

The MNIST Table 4 flow is part of the consolidated experiment runner. The
ImageNet Table 4 flow exists as a faithful Caffe/GoogLeNet reproduction with a
different dataset, model, attack scale, and output shape.

Table 4 must not be represented as MNIST only, and the ImageNet reproduction
must not be hidden behind a separate result tree unrelated to Table 4.

## Business Rules

- `table_4` runs the MNIST and ImageNet Table 4 components in sequence.
- `table_4_mnist` runs only the MNIST component.
- `table_4_imagenet` runs only the ImageNet component.
- Each component must explicitly declare dataset, model, attack, samples/split,
  and filter or filter family.
- Global defaults may only define output filenames.
- The outputs must share the same Table 4 result root:
  `results/experiments/table_4/`.
- MNIST and ImageNet outputs must remain separated below that root because their
  CSV schemas are different.

## Functional Requirements

- `python scripts/run_experiment.py --experiment table_4` runs both components.
- `python scripts/run_experiment.py --experiment table_4_mnist` runs only MNIST.
- `python scripts/run_experiment.py --experiment table_4_imagenet` runs only
  ImageNet.
- MNIST writes standard `metrics.csv` and `metrics.json`.
- ImageNet writes `table_4_imagenet.csv` and `table_4_status.json`.
- `table_4` writes a manifest describing the component outputs.

## Non-Functional Requirements

- Do not alter filter math, FGSM generation, model architecture, or metrics.
- Keep the runner simple and avoid table-specific duplication beyond the
  ImageNet Caffe orchestration required by the existing spec.
- Keep all experiment parameters explicit in YAML.

## Acceptance Criteria

- `configs/experiments.yaml` contains `table_4`, `table_4_mnist`, and
  `table_4_imagenet`.
- `table_4.kind == composite`.
- `table_4.components == [table_4_mnist, table_4_imagenet]`.
- `table_4_mnist.output_dir == results/experiments/table_4/mnist`.
- `table_4_imagenet.output_dir == results/experiments/table_4/imagenet`.
- The consolidated runner can build configs for all three entries.
- The composite runner executes components in declared order.
- Tests cover config fidelity and composite dispatch.
- The ImageNet Table 4 runtime does not generate a diagnostics CSV.
- The legacy `configs/article_reproduction/imagenet_table_4.yaml` file is not
  required because the official config is `configs/experiments.yaml`.

## Error Cases

- Unknown component names fail with a clear unknown experiment error.
- Missing MNIST assets or checkpoints fail with the existing clear MNIST error.
- Missing ImageNet/Caffe assets write the ImageNet status JSON as before.

## Out of Scope

- Combining MNIST and ImageNet rows into one CSV.
- Changing Table 4 metrics.
- Changing ImageNet preprocessing or FGSM scale.
- Changing Table 10.
