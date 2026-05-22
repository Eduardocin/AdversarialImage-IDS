# MNIST M2 CW L2

## Configuration

- samples: 1000
- start_index: 9000
- kappas: 0, 0.5, 1, 2, 4
- batch_size: 32
- max_iterations: 1000
- learning_rate: 0.01
- binary_search_steps: 5
- train_dir: `C:\Users\Eduar\OneDrive\Documentos\GitHub\AdversarialImage-IDS\results\mnist\m2_cw\clean_baseline\checkpoints`

| kappa | n_total | clean_accuracy | adversarial_accuracy | attack_success_rate | mean_l2 | median_l2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1000 | 0.994000 | 0.247000 | 0.751509 | 1.281979 | 1.243137 |
| 0.5 | 1000 | 0.994000 | 0.250000 | 0.748491 | 1.316017 | 1.277976 |
| 1 | 1000 | 0.994000 | 0.250000 | 0.748491 | 1.363081 | 1.320637 |
| 2 | 1000 | 0.994000 | 0.252000 | 0.746479 | 1.442989 | 1.398835 |
| 4 | 1000 | 0.994000 | 0.257000 | 0.741449 | 1.614395 | 1.579258 |
