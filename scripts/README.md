# Scripts

Indice operacional dos scripts do projeto. A pasta esta organizada por fluxo e
os imports funcionam com o pacote instalado em modo editavel (`pip install -e .`).

## Fluxo MNIST M1 + FGSM

| Script | Papel | Saidas principais |
| --- | --- | --- |
| `mnist/m1_fgsm/train.py` | Treina ou restaura o baseline MNIST M1 | modelo em `artifacts/models/mnist/m1/`; resumo em `results/mnist/clean_baseline/` |
| `mnist/m1_fgsm/generate_attack.py` | Gera adversariais FGSM | imagens em `artifacts/adversarial_examples/mnist/m1/fgsm/`; resumo em `results/mnist/fgsm/` |
| `mnist/m1_fgsm/evaluate_detector.py` | Avalia detectores por mudanca de predicao | `results/mnist/detector/` |
| `mnist/m1_fgsm/evaluate_entropy.py` | Avalia detector com filtro por entropia | `results/mnist/entropy/` |
| `mnist/m1_fgsm/run_comparison.py` | Orquestra comparacao de filtros MNIST | `results/mnist/final_mnist_results.csv` |

Ordem usual:

```bash
python scripts/mnist/m1_fgsm/train.py --load-model
python scripts/mnist/m1_fgsm/generate_attack.py --load-model
python scripts/mnist/m1_fgsm/run_comparison.py
```

## Comparacoes MNIST com valores de referencia

| Script | Objetivo | Saidas principais |
| --- | --- | --- |
| `article_reproduction/table_3.py` | Quantizacao uniforme vs nao uniforme | `results/mnist/article_reproduction/table_3_*` |
| `article_reproduction/table_4.py` | Intervalos de quantizacao escalar | `results/mnist/article_reproduction/table_4_*` |
| `article_reproduction/table_6.py` | Quantizacao adaptativa | `results/mnist/article_reproduction/table_6_*` |
| `article_reproduction/table_10.py` | Detector MNIST FGSM | `results/mnist/article_reproduction/table_10_mnist_fgsm_test.*` |

## Fluxo MNIST M2 + CW

| Script | Papel | Saidas principais |
| --- | --- | --- |
| `mnist/m2_cw/train.py` | Treina ou restaura o baseline M2 | modelo em `artifacts/models/mnist/m2/`; resumo em `results/mnist/m2_cw/clean_baseline/` |
| `mnist/m2_cw/generate_attack_l2.py` | Gera CW L2 para kappas configurados | imagens em `artifacts/adversarial_examples/mnist/m2/cw_l2/`; resumo em `results/mnist/m2_cw/cw_l2/` |
| `mnist/m2_cw/generate_attack_linf.py` | Gera CW Linf com implementacao local TF1 | imagens em `artifacts/adversarial_examples/mnist/m2/cw_linf/`; resumo em `results/mnist/m2_cw/cw_linf/` |
| `mnist/m2_cw/evaluate_detector.py` | Avalia detector em CW L2/Linf | `results/mnist/m2_cw/detector/` |
| `mnist/m2_cw/run_experiments.py` | Orquestra o fluxo M2 + CW | `results/mnist/m2_cw/` |

Ordem usual:

```bash
python scripts/mnist/m2_cw/run_experiments.py --kappas 0.0,0.5,1.0,2.0,4.0 --samples 1000 --start-index 9000
```

## Utilitarios

| Script | Papel |
| --- | --- |
| `dev/smoke_test.py` | Verifica rapidamente imports/dependencias principais |

## Estrutura

```text
scripts/
  mnist/
    m1_fgsm/
    m2_cw/
  article_reproduction/
  dev/
```
