# Table 10 - MNIST FGSM Test Rows 1-4

| Attack/Model | Article #F | Our #F | Article TP | Our TP | Article FN | Our FN | Article FP | Our FP | Article RTP | Our RTP | Article RTP% | Our RTP% | Article recall | Our recall | Article precision | Our precision | Article F1 | Our F1 | Delta F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FGSM (eps=0.1)/M1 | 4026 | 4125 | 410 | 328 | 40 | 47 | 24 | 34 | 398 | 321 | 97.07% | 97.87% | 91.11% | 87.47% | 94.47% | 90.61% | 92.76% | 89.01% | -3.75 |
| FGSM (eps=0.2)/M1 | 1910 | 2603 | 2467 | 1791 | 106 | 106 | 32 | 34 | 2430 | 1750 | 98.50% | 97.71% | 95.88% | 94.41% | 98.72% | 98.14% | 97.28% | 96.24% | -1.04 |
| FGSM (eps=0.3)/M1 | 455 | 654 | 3856 | 3675 | 172 | 171 | 32 | 34 | 3768 | 3560 | 97.71% | 96.87% | 95.73% | 95.55% | 99.18% | 99.08% | 97.42% | 97.29% | -0.13 |
| FGSM (eps=0.4)/M1 | 132 | 188 | 4078 | 4058 | 273 | 254 | 32 | 34 | 3820 | 3723 | 93.67% | 91.74% | 93.73% | 94.11% | 99.22% | 99.17% | 96.40% | 96.57% | +0.17 |

Article values are from Liang et al., Table 10 rows 1-4.
#F denotes samples whose FGSM image is still classified as the true label.
The reproduction uses MNIST test digits 5500-9999 and the final entropy-aware detection filter.
