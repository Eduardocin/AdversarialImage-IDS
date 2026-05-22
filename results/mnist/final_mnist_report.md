# MNIST Filter Comparison

## 1. Resultados consolidados

| filter_name | n_total | n_discarded | TP | FP | FN | TN | TTP | precision | recall | f1 | ttp_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar_128 | 8 | 2 | 6 | 0 | 0 | 0 | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| scalar_64 | 8 | 2 | 6 | 0 | 0 | 0 | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| scalar_43 | 8 | 2 | 0 | 0 | 6 | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| nonuniform | 8 | 2 | 6 | 6 | 0 | 0 | 4 | 0.500000 | 1.000000 | 0.666667 | 0.666667 |
| entropy_adaptive | 8 | 2 | 6 | 0 | 0 | 0 | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| box_3 | 8 | 2 | 0 | 0 | 6 | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| box_5 | 8 | 2 | 1 | 2 | 5 | 4 | 0 | 0.333333 | 0.166667 | 0.222222 | 0.000000 |
| cross_3 | 8 | 2 | 0 | 0 | 6 | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| diamond_3 | 8 | 2 | 0 | 0 | 6 | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 2. Analise automatica

- Maior recall: scalar_128, scalar_64, nonuniform, entropy_adaptive (recall=1.000000)
- Maior precision: scalar_128, scalar_64, entropy_adaptive (precision=1.000000)
- Maior ttp_rate: scalar_128, scalar_64, entropy_adaptive (ttp_rate=1.000000)
- Menor FP: scalar_128, scalar_64, scalar_43, entropy_adaptive, box_3, cross_3, diamond_3 (FP=0)

## 3. Observacoes sobre descartes

| filter_name | n_total | n_discarded | discarded_clean_error | discarded_attack_failed |
| --- | ---: | ---: | ---: | ---: |
| scalar_128 | 8 | 2 | 0 | 2 |
| scalar_64 | 8 | 2 | 0 | 2 |
| scalar_43 | 8 | 2 | 0 | 2 |
| nonuniform | 8 | 2 | 0 | 2 |
| entropy_adaptive | 8 | 2 | 0 | 2 |
| box_3 | 8 | 2 | 0 | 2 |
| box_5 | 8 | 2 | 0 | 2 |
| cross_3 | 8 | 2 | 0 | 2 |
| diamond_3 | 8 | 2 | 0 | 2 |
