# MNIST M2 CW Linf

## Configuration

- samples: 2
- start_index: 9998
- batch_size: 1
- max_iterations: 1
- learning_rate: 0.01
- confidence: 0.0
- initial_tau: 1.0
- const: 1.0
- tau_decay: 0.9
- tau_check_interval: 1

| n_total | clean_accuracy | adversarial_accuracy | attack_success_rate | mean_linf | median_linf | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 1.000000 | 1.000000 | 0.000000 | 0.010000 | 0.010000 | executed |

This is a local TF1 CW Linf-style implementation because CleverHans 3.1.0 does not provide Carlini-Wagner Linf for this stack.
