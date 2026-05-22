# MNIST FGSM

## Configuration

- samples: 4500
- epsilons: 0.2
- clip_min: 0.0
- clip_max: 1.0
- train_dir: `C:\Users\Eduar\OneDrive\Documentos\GitHub\AdversarialImage-IDS\results\mnist\clean_baseline\checkpoints`
- load_model: True

## Metrics

| epsilon | clean_accuracy | adversarial_accuracy | attack_success_rate | clean_errors | attack_failures | successful_attacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.989111 | 0.434889 | 0.568861 | 49 | 1919 | 2532 |

## Semantics

`adversarial_accuracy` is the fraction of adversarial images still classified as the true label. `attack_success_rate` is computed only over clean images classified correctly before perturbation.
