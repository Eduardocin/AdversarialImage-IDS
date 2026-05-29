# Adversarial Example Cache

## Objective

Persist generated adversarial examples under `artifacts/adversarial_examples/` and reuse them on later experiment runs when the experiment inputs match.

## Context

The official experiment runner regenerates FGSM adversarial examples every time a configured experiment is executed. This is unnecessary for deterministic experiment inputs such as MNIST FGSM slices and ImageNet Table 4 FGSM attacks.

## Business Rules

- Cached adversarial examples must be stored under `artifacts/adversarial_examples/` by default.
- Cache keys must be derived from the material inputs that affect adversarial generation.
- A cache hit must avoid calling the FGSM generation function again.
- A cache miss must generate adversarial examples, write them to disk, and continue the experiment.
- Cache artifacts are generated experiment artifacts and must not be versioned.
- Caching must not change filter math, attack math, or metric formulas.

## Functional Requirements

- MNIST FGSM experiments must cache adversarial images and baseline clean/adversarial predictions.
- `filter_grid` MNIST experiments must reuse the cached adversarial set across repeated `run_experiment.py` executions.
- `split_eval` MNIST experiments must cache each configured split independently.
- ImageNet Table 4 must cache the valid successful FGSM attacks and baseline counters.
- ImageNet Table 6 must reuse split-level FGSM caches from
  `artifacts/adversarial_examples/imagenet/fgsm/{split}/adversarial_examples.npy`
  when they are present and compatible.
- Cache use may be disabled with `attack.cache: false`.
- Cache location may be overridden with `attack.cache_dir`.

## Non-Functional Requirements

- Cache files must have deterministic names.
- Cache metadata must be human-readable.
- Cache loading must validate the basic array shape before reuse.
- Generated cache files must remain out of git.

## Acceptance Criteria

- Running `table_7` twice with unchanged config loads MNIST FGSM adversarial examples from cache on the second run.
- Running `table_6` and `table_9` can reuse identical MNIST split attack caches.
- Running `table_4_imagenet` twice with unchanged config loads cached successful ImageNet FGSM attacks on the second run.
- Running `table_6` with existing compatible ImageNet FGSM split caches does not call the ImageNet FGSM generator again for those splits.
- Automated tests cover cache miss, cache hit, and generated artifact ignore behavior.

## Error Cases

- If a cache file is missing, the experiment regenerates it.
- If a cache file is malformed or incompatible with the loaded samples, the experiment regenerates it.
- If `attack.cache` is false, no cache read or write occurs.

## Out of Scope

- Caching training outputs or model checkpoints.
- Caching filtered images.
- Changing adversarial attack implementations.
- Adding a cache eviction policy.
