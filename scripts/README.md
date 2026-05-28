# Scripts

Indice operacional dos scripts do projeto. A pasta `scripts/` e uma interface
de execucao; a logica experimental comum fica em `src/deepdetector`.

## Experimentos Principais

As Tabelas 6-9 usam um unico ponto de entrada e a configuracao consolidada em
`configs/experiments.yaml`.

```bash
python scripts/run_experiment.py --experiment table_6
python scripts/run_experiment.py --experiment table_7
python scripts/run_experiment.py --experiment table_8
python scripts/run_experiment.py --experiment table_9
```

Cada execucao escreve apenas:

```text
results/experiments/<experiment_id>/metrics.csv
results/experiments/<experiment_id>/metrics.json
```

## Fluxos Auxiliares

| Script | Papel |
| --- | --- |
| `dev/smoke_test.py` | Verifica rapidamente imports/dependencias principais. |
| `imagenet/download_caffe_imagenet_assets.py` | Baixa ativos Caffe para a trilha ImageNet. |
| `imagenet/googlenet_fgsm.py` | Utilitario FGSM ImageNet/GoogLeNet. |
| `imagenet/process_imagenet.py` | Prepara subconjuntos ImageNet locais. |

Os fluxos MNIST legados em `scripts/mnist/` continuam disponiveis para treino,
geracao de ataques e analises especificas. Scripts historicos em
`scripts/article_reproduction/` nao fazem parte do caminho operacional
principal das Tabelas 6-9.
