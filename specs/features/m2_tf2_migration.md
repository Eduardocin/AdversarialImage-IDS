# SPEC - M2 TensorFlow 2 Compatibility

## Objective

Run the MNIST M2 Table 10 workflow in a TensorFlow 2 environment while
preserving the existing TF1 graph/session execution model and experiment
outputs.

## Context

The M2 workflow is currently tied to TensorFlow 1.15 and legacy Keras APIs.
Cluster environments increasingly provide TensorFlow 2 stacks, which makes the
M2 CW workflow difficult to run. The project already uses a minimal TF2
compatibility approach for Inception v3 by disabling eager execution and using
`tf.compat.v1`.

## Business Rules

- M2 must continue to run through the existing Table 10 M2 command paths.
- M2 must continue using saved CW adversarial arrays unless attack regeneration
  is explicitly requested.
- M2 CW-L2 generation must continue using `nn_robust_attacks.CarliniL2`.
- M2 CW-Linf generation must continue using `nn_robust_attacks.CarliniLi`.
- Local CW/CleverHans attack backends must not be introduced for M2.
- TensorFlow 2 compatibility must preserve graph/session execution semantics.

## Functional Requirements

- Provide a dedicated Conda environment file for M2 with TensorFlow 2.11.
- Disable TensorFlow eager execution before creating M2 placeholders,
  sessions, or Keras graph layers.
- Keep `tf.compat.v1.Session`, placeholders, and Saver checkpoint restore for
  M2.
- Build the M2 Keras graph with layer arguments compatible with both legacy
  Keras and TensorFlow 2 Keras.
- Keep the `M2NnRobustAdapter` input scale contract unchanged.
- Keep output paths, CSV/Markdown schemas, and adversarial manifests unchanged.

## Non-Functional Requirements

- Keep changes minimal and scoped to M2/MNIST TensorFlow compatibility.
- Do not migrate M2 to TensorFlow 2 eager execution.
- Do not rewrite the M2 model architecture.
- Do not change ImageNet, Caffe, or Inception v3 behavior.

## Acceptance Criteria

- The repository contains an M2 TensorFlow 2 environment file under `envs/`.
- M2 graph creation disables eager execution before placeholder creation.
- M2 model construction supports modern Keras Conv2D arguments.
- MNIST TensorFlow session creation tolerates Keras backends without
  `set_session` or `set_image_dim_ordering`.
- Existing M2 CW generation tests continue to pass.
- Tests cover the TF2 compatibility path without requiring a full cluster run.

## Error Cases

- Missing M2 checkpoints still raise a clear `IOError`.
- Missing `--nn-robust-attacks-root` during M2 attack generation still raises a
  clear `ValueError`.
- Existing adversarial arrays are still protected unless `--overwrite-attacks`
  is provided.

## Out of Scope

- Native TensorFlow 2 eager execution for M2.
- Re-training M2 in a new checkpoint format.
- Replacing `nn_robust_attacks`.
- Changing Table 10 metrics, article row numbers, or result schemas.
