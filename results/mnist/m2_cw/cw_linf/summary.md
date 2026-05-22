# MNIST M2 CW Linf

## Configuration

- samples: 1000
- start_index: 9000
- batch_size: 32
- max_iterations: 1000
- learning_rate: 0.01
- confidence: 0.0
- initial_tau: 1.0
- const: 1.0
- tau_decay: 0.9
- tau_check_interval: 50

| n_total | clean_accuracy | adversarial_accuracy | attack_success_rate | mean_linf | median_linf | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 0.994000 | 0.290000 | 0.708249 | 0.367141 | 0.392157 | executed |
