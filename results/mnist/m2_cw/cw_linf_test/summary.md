# MNIST M2 CW Linf

## Configuration

- samples: 10
- start_index: 9990
- batch_size: 2
- max_iterations: 50
- learning_rate: 0.01
- confidence: 0.0
- initial_tau: 1.0
- const: 1.0
- tau_decay: 0.9
- tau_check_interval: 10

| n_total | clean_accuracy | adversarial_accuracy | attack_success_rate | mean_linf | median_linf | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 1.000000 | 0.500000 | 0.500000 | 0.275311 | 0.262851 | executed |

This is a local TF1 CW Linf-style implementation because CleverHans 3.1.0 does not provide Carlini-Wagner Linf for this stack.
