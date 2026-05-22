# MNIST FGSM

## Configuration

- samples: 10
- epsilons: 0.2
- clip_min: 0.0
- clip_max: 1.0
- train_dir: `C:\Users\Eduar\OneDrive\Documentos\GitHub\AdversarialImage-IDS\results\mnist\clean_baseline\checkpoints`
- load_model: True

## Metrics

| epsilon | clean_accuracy | adversarial_accuracy | attack_success_rate | clean_errors | attack_failures | successful_attacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 1.000000 | 0.600000 | 0.400000 | 0 | 6 | 4 |

## Semantics

`adversarial_accuracy` is the fraction of adversarial images still classified as the true label. `attack_success_rate` is computed only over clean images classified correctly before perturbation.
