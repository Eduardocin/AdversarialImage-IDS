# MNIST Prediction-Change Detector

| filtro | n_total | n_descartados | TP | FP | FN | TTP | precision | recall | f1 | ttp_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar_interval_128 | 4500 | 1968 | 2342 | 23 | 190 | 2295 | 0.990275 | 0.924961 | 0.956504 | 0.979932 |
| scalar_interval_64 | 4500 | 1968 | 2315 | 9 | 217 | 2278 | 0.996127 | 0.914297 | 0.953460 | 0.984017 |
| scalar_interval_43 | 4500 | 1968 | 159 | 4 | 2373 | 47 | 0.975460 | 0.062796 | 0.117996 | 0.295597 |
| nonuniform_quantization | 4500 | 1968 | 2287 | 2175 | 245 | 2040 | 0.512550 | 0.903239 | 0.653989 | 0.891998 |

Descartes incluem erro limpo e ataque FGSM que nao mudou a classe verdadeira.
TTP e o subconjunto dos TP em que a filtragem tambem corrige a classe para o rotulo verdadeiro.
