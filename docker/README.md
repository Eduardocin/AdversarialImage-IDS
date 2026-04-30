# Ambiente Docker

Este diretório define o ambiente de replicação do DeepDetector original. A base `bvlc/caffe:cpu` mantém Caffe e Python 2, enquanto `requirements-legacy.txt` fixa as bibliotecas antigas usadas por CleverHans/TensorFlow 1.x.

## Build
```bash
./docker/build_docker.sh
```

## Container descartável
```bash
./docker/run_docker.sh
```
Para executar um comando e remover o container ao final:
```bash
./docker/run_docker.sh python scripts/check_legacy_environment.py
```

## Container persistente
```bash
./docker/start_docker.sh
```

Esse script cria, inicia ou reentra no container `adversarialimage-ids-legacy`, montando a raiz do repositório em `/workspace`.

## Verificação rápida
Dentro do container:
```bash
python scripts/check_legacy_environment.py
```
Se todos os imports passarem, o ambiente está pronto para iniciar a replicação dos scripts em `src/original/DeepDetector`.
