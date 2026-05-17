# Scripts

Indice operacional dos scripts do projeto. A pasta esta organizada por fluxo, e
os scripts usam `_project_root.py` para descobrir a raiz do repositorio a partir
de qualquer subpasta.

## Fluxo MNIST M1 + FGSM

| Script | Papel | Saidas principais |
| --- | --- | --- |
| `mnist_m1_fgsm/train_mnist.py` | Treina ou restaura o baseline MNIST M1 | `results/mnist/clean_baseline/` |
| `mnist_m1_fgsm/generate_mnist_fgsm.py` | Gera adversariais FGSM | `results/mnist/fgsm/` |
| `mnist_m1_fgsm/evaluate_mnist_detector.py` | Avalia detectores por mudanca de predicao | `results/mnist/detector/` |
| `mnist_m1_fgsm/evaluate_mnist_entropy_detector.py` | Avalia detector com filtro por entropia | `results/mnist/entropy/` |
| `mnist_m1_fgsm/run_mnist_filter_comparison.py` | Orquestra comparacao de filtros MNIST | `results/mnist/final_mnist_results.csv` |

Ordem usual:

```bash
python scripts/mnist_m1_fgsm/train_mnist.py --load-model
python scripts/mnist_m1_fgsm/generate_mnist_fgsm.py --load-model
python scripts/mnist_m1_fgsm/run_mnist_filter_comparison.py
```

## Reproducao de tabelas MNIST do artigo

| Script | Tabela/objetivo | Saidas principais |
| --- | --- | --- |
| `article_reproduction/table_3_uniform_vs_nonuniform.py` | Tabela 3 | `results/mnist/article_reproduction/table_3_*` |
| `article_reproduction/table_4_scalar_quantization_intervals.py` | Tabela 4 | `results/mnist/article_reproduction/table_4_*` |
| `article_reproduction/table_6_adaptive_quantization.py` | Tabela 6 | `results/mnist/article_reproduction/table_6_*` |
| `article_reproduction/table_10_mnist_fgsm_test.py` | Tabela 10, MNIST FGSM | `results/mnist/article_reproduction/table_10_mnist_fgsm_test.*` |

## Fluxo MNIST M2 + CW

| Script | Papel | Saidas principais |
| --- | --- | --- |
| `mnist_m2_cw/train_mnist_m2.py` | Treina ou restaura o baseline M2 | `results/mnist/m2_cw/clean_baseline/` |
| `mnist_m2_cw/generate_mnist_cw_l2.py` | Gera CW L2 para kappas do artigo | `results/mnist/m2_cw/cw_l2/` |
| `mnist_m2_cw/generate_mnist_cw_linf.py` | Registra CW Linf ou limitacao | `results/mnist/m2_cw/cw_linf/` |
| `mnist_m2_cw/evaluate_mnist_m2_cw_detector.py` | Avalia detector em CW L2/Linf | `results/mnist/m2_cw/detector/` |
| `mnist_m2_cw/run_all_mnist_m2_cw_experiments.py` | Orquestra o fluxo M2 + CW | `results/mnist/m2_cw/` |

Ordem usual:

```bash
python scripts/mnist_m2_cw/run_all_mnist_m2_cw_experiments.py --kappas 0.0,0.5,1.0,2.0,4.0 --samples 1000 --start-index 9000
```

## Utilitarios

| Script | Papel |
| --- | --- |
| `dev/smoke_test.py` | Verifica rapidamente imports/dependencias principais |

## Estrutura

```text
scripts/
  _project_root.py
  mnist_m1_fgsm/
  mnist_m2_cw/
  article_reproduction/
  dev/
```
