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
results/experiments/<experiment_id>/metrics.csv
results/experiments/<experiment_id>/metrics.json
```

Table 4 e excecao composta: seus resultados ficam em
`results/experiments/table_4/mnist/` e `results/experiments/table_4/imagenet/`,
com `manifest.json` na raiz da tabela.

## Fluxos Auxiliares

| Script | Papel |
| --- | --- |
| `dev/smoke_test.py` | Verifica rapidamente imports/dependencias principais. |
| `imagenet/download_caffe_imagenet_assets.py` | Baixa ativos Caffe para a trilha ImageNet. |

Table 10 fica preservada como excecao historica fora do fluxo oficial desta
limpeza em `scripts/article_reproduction/table_10.py` e
`scripts/article_reproduction/table_10_m2.py`.
