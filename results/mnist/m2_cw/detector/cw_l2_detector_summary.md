# MNIST M2 CW L2 Detector

| attack | norm | kappa | n_total | #F | TP | FN | FP | RTP | RTP% | recall | precision | f1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CW | L2 | 0.0 | 1000 | 247 | 685 | 62 | 5 | 677 | 98.83 | 91.70 | 99.28 | 95.34 |
| CW | L2 | 0.5 | 1000 | 250 | 661 | 83 | 5 | 649 | 98.18 | 88.84 | 99.25 | 93.76 |
| CW | L2 | 1.0 | 1000 | 250 | 649 | 95 | 5 | 634 | 97.69 | 87.23 | 99.24 | 92.85 |
| CW | L2 | 2.0 | 1000 | 252 | 608 | 134 | 5 | 598 | 98.36 | 81.94 | 99.18 | 89.74 |
| CW | L2 | 4.0 | 1000 | 257 | 526 | 211 | 5 | 517 | 98.29 | 71.37 | 99.06 | 82.97 |

Metrics are percentages on the same scale as Table 10. Clean errors and attack failures are recorded separately.
