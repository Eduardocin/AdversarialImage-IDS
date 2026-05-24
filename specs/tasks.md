# SDD Tasks

Use this file to track implementation tasks derived from specifications in `specs/features`.

## Backlog

- [x] Implement Table 4 ImageNet reproduction from `specs/features/Table 4 ImageNet.md`
  - [x] Add CLI support for `--data-root`, `--limit`, `--epsilon`, and `--output-dir`.
  - [x] Evaluate ImageNet PNGs with GoogLeNet/Caffe predictions and FGSM generated from model gradients.
  - [x] Write `table_4_imagenet.csv` and `table_4_imagenet_diagnostics.csv`.
  - [x] Fail explicitly when FGSM produces zero successful adversarial examples.
  - [x] Add automated tests for interval rows, metrics, diagnostics, and zero-attack failure.
- [x] Refactor ImageNet FGSM to match `Train_FGSM_ImageNet.py`
  - [x] Make the main ImageNet FGSM path Caffe-only, without TensorFlow/CleverHans.
  - [x] Generate adversarial images in preprocessed Caffe space `CHW/BGR/0..255`.
  - [x] Apply `epsilon_255=1.0` as a raw pixel step and clip to `[0,255]`.
  - [x] Share the same attack helper across Table 4, Table 7, and GoogLeNet FGSM scripts.
  - [x] Add tests covering clean-baseline skip, Caffe-scale epsilon, and no TensorFlow import in the main path.
