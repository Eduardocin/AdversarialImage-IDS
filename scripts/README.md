# Scripts

Indice operacional dos scripts do projeto. A pasta `scripts/` e uma interface
de execucao; a logica experimental comum fica em `src/deepdetector`.

## Experimentos Principais

As Tabelas 3-9 usam um unico ponto de entrada e a configuracao consolidada em
`configs/experiments.yaml`.

```bash
python scripts/run_experiment.py --experiment table_3
python scripts/run_experiment.py --experiment table_4
python scripts/run_experiment.py --experiment table_6
python scripts/run_experiment.py --experiment table_7
python scripts/run_experiment.py --experiment table_8
python scripts/run_experiment.py --experiment table_9
python scripts/run_experiment.py --experiment table_10_m1
python scripts/run_experiment.py --experiment table_10_googlenet
python scripts/run_experiment.py --experiment table_10_caffenet
python scripts/run_experiment.py --experiment table_10_m2
python scripts/run_experiment.py --experiment table_10_inception_v3
```

`table_4` executa os dois componentes da tabela em sequencia. Para rodar apenas
um lado:

```bash
python scripts/run_experiment.py --experiment table_4_mnist
python scripts/run_experiment.py --experiment table_4_imagenet
```

Table 5 nao aparece no caminho operacional porque nao ha script, config ou
resultado correspondente no inventario atual do codigo.

Cada execucao simples escreve:

```text
results/<experiment_id>/metrics.csv
results/<experiment_id>/metrics.json
```

Tables 7 e 8 ImageNet sao excecoes de formato: escrevem pivots
`table_7_imagnet.csv` e `table_8_imagenet.csv`, com seus respectivos
`table_*_status.json`.

Table 4 e excecao composta: seus resultados ficam em
`results/table_4/mnist/` e `results/table_4/imagenet/`,
com `manifest.json` na raiz da tabela. Table 6 e Table 9 tambem executam MNIST
e ImageNet internamente, mas escrevem somente os agregados oficiais em
`results/table_6/` e `results/table_9/`.

Table 10 tambem e separada por grupo de modelo. Os grupos ImageNet escrevem
`metrics.csv`, `metrics.json` e `manifest.json` em
`results/table_10/<grupo>/`. O M2 CW fica separado em
`results/table_10/M2_cw_l2/` e `results/table_10/M2_cw_Linf/`.

Para preparar checkpoints Fashion-MNIST antes dos experimentos publicos:

```bash
python scripts/train_fashion_mnist_checkpoint.py --model m3
python scripts/train_fashion_mnist_checkpoint.py --model m2
```

Depois execute os experimentos pelo runner central:

```bash
python scripts/run_experiment.py --experiment fashion_mnist_fgsm_m3
python scripts/run_experiment.py --experiment fashion_mnist_cw_l2_m2
```

Para gerar adversariais CW do M2, use o auxiliar nomeado por acao/tabela/escopo:

```bash
python scripts/article_reproduction/mnist_table_10_m2_cw.py \
  --generate-attacks \
  --overwrite-attacks \
  --nn-robust-attacks-root nn_robust_attacks
```

Esse fluxo M2 usa `nn_robust_attacks.CarliniL2` para CW-L2 e
`nn_robust_attacks.CarliniLi` para CW-Linf.

## Fluxos Auxiliares

| Script | Papel |
| --- | --- |
| `dev/smoke_test.py` | Verifica rapidamente imports/dependencias principais. |
| `imagenet/download_caffe_imagenet_assets.py` | Baixa ativos Caffe para a trilha ImageNet. |

Scripts historicos de reproducao permanecem fora do caminho operacional
principal quando ainda existirem, mas os experimentos oficiais devem passar por
`scripts/run_experiment.py`.
