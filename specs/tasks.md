# SDD Tasks

Use this file to track implementation tasks derived from specifications in `specs/features`.

## Backlog

- [x] Implement Table 9 final FGSM detector from `specs/features/table_9.md`
  - [x] Add `configs/article_reproduction/table_9.yaml`.
  - [x] Add `src/deepdetector/filters/article_final.py`.
  - [x] Register and export `article_final`.
  - [x] Add `scripts/article_reproduction/table_9.py`.
  - [x] Orchestrate MNIST M1 FGSM and ImageNet GoogLeNet FGSM flows.
  - [x] Aggregate TP, FN, and FP by split before calculating metrics.
  - [x] Write `table_9.csv`, `table_9.md`, and `status.json`.
  - [x] Add automated tests for config, filter behavior, registry, aggregation, CSV shape, and dry-run status.
- [x] Implement Table 6 ImageNet adaptive quantization from `specs/features/table_6_imagenet.md`
  - [x] Add `configs/article_reproduction/imagenet_table_6.yaml`.
  - [x] Add `scripts/article_reproduction/table_6_imagenet.py`.
  - [x] Reuse the Caffe-scale GoogLeNet FGSM path from Table 4 ImageNet.
  - [x] Write `table_6_imagenet.csv` and `table_6_imagenet_diagnostics.csv`.
  - [x] Add automated tests for adaptive steps, metrics, diagnostics, CSV shape, and zero-attack failure.
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
- [x] Add FGSM diagnostic guard for mean-subtracted ImageNet tensors
  - [x] Fail explicitly when the Caffe FGSM helper receives negative preprocessed values before `[0,255]` clipping.
  - [x] Add a regression test showing that mean-subtracted Caffe tensors are rejected for the article reproduction path.
- [x] Update Table 7 ImageNet output to pivot-only CSV named `table_7_imagnet.csv`.
- [x] Fix Table 7 ImageNet methodological fidelity for high-entropy spatial smoothing.
  - [x] Apply only the spatial smoothing filter in `table7_filter`, without scalar quantization.
  - [x] Count FP, TP, and FN over the same valid high-entropy adversarial pairs.
  - [x] Add regression tests for filter-only behavior and compatible Table 7 populations.
- [x] Preserve Table 7 spatial smoothing borders.
  - [x] Apply masked mean values only where the full mask fits inside the image.
  - [x] Preserve border pixels exactly instead of using reflect padding.
  - [x] Add a regression test for unchanged spatial borders.
- [ ] Implement Table 8 ImageNet validation spatial smoothing from `specs/features/Table 8 ImageNet.md`
  - [x] Add config-driven Table 8 ImageNet experiment for the validation split.
  - [x] Evaluate exactly the five fixed superior spatial smoothing filters.
  - [x] Write pivot CSV `table_8_imagenet.csv` and Table 8 status JSON.
  - [x] Preserve clean-baseline filtering and disturbed-failure exclusion.
  - [x] Add automated tests for filters, pivot output, validation config, and partial statuses.
  - [x] Run focused pytest validation.
