# Detector de Exemplos Adversarios com Reducao de Ruido

Este projeto implementa um pipeline moderno para detectar exemplos adversarios em redes neurais. O detector nao treina a rede-alvo: ele recebe um modelo ja treinado, gera ataques contra esse modelo e verifica se a predicao muda depois de aplicar transformacoes de reducao de ruido.

O foco inicial e MNIST. A estrutura ja deixa espaco para evoluir depois para ImageNet e redes pre-treinadas modernas.

## Artigo de Referencia

B. Liang, H. Li, M. Su, X. Li, W. Shi and X. Wang, "Detecting Adversarial Image Examples in Deep Networks with Adaptive Noise Reduction", 2017.

Este repositorio e uma replicacao modernizada do metodo, priorizando uma implementacao em Python/PyTorch, separacao clara entre rede-alvo, ataque e detector, e validacao incremental.

## Como Funciona

```text
Dataset MNIST
     |
     v
Rede-alvo ja treinada
     |
     +--> predicao da imagem limpa
     |
     +--> ataque FGSM gera imagem adversarial
                 |
                 v
        transformacoes do detector
        - quantizacao uniforme
        - quantizacao nao uniforme
        - filtros box / diamond / cross
                 |
                 v
        comparar predicoes antes/depois
                 |
                 v
        metricas: TP, FN, FP, TTP, precision, recall
```

## Fluxo MNIST

O MNIST usa a seguinte divisao por indice do conjunto de teste:

```text
train:      0-4499
validation: 4500-5499
test:       5500-9999
```

A fase `train_detector` seleciona parametros do detector usando FGSM. Ela nao treina a rede-alvo.

```text
train_detector_mnist.py
    escolhe a melhor configuracao do detector no split train

validate_detector_mnist.py
    mede a configuracao escolhida no split validation

test_detector_mnist.py
    mede a configuracao escolhida no split test
```

## Instalar

```bash
conda create -n adversarialimage-ids python=3.11 -y
conda activate adversarialimage-ids
pip install -r requirements.txt
```

## Preparar o MNIST

```bash
python scripts/download_mnist_test.py
```

Saidas principais:

```text
data/mnist/mnist_splits.npz
data/mnist/images/train
data/mnist/images/validation
data/mnist/images/test
```

## Executar

Os scripts esperam uma rede-alvo ja treinada/exportada em TorchScript:

```text
checkpoints/mnist/target.pt
```

Selecionar parametros do detector:

```bash
python scripts/train_detector_mnist.py \
  --target-model-path checkpoints/mnist/target.pt
```

Validar:

```bash
python scripts/validate_detector_mnist.py \
  --target-model-path checkpoints/mnist/target.pt
```

Testar:

```bash
python scripts/test_detector_mnist.py \
  --target-model-path checkpoints/mnist/target.pt
```

Para uma execucao curta:

```bash
python scripts/train_detector_mnist.py \
  --target-model-path checkpoints/mnist/target.pt \
  --debug
```

A configuracao escolhida e salva em:

```text
outputs/detectors/mnist_fgsm_detector.json
```

## Estrutura

```text
.
|-- scripts/
|   |-- download_mnist_test.py
|   |-- train_detector_mnist.py
|   |-- validate_detector_mnist.py
|   `-- test_detector_mnist.py
|
|-- src/
|   |-- attacks/        # FGSM e futuros ataques
|   |-- datasets/       # loaders e splits
|   |-- detector/       # transformacoes, detector e selecao de parametros
|   |-- models/         # carregamento de redes-alvo
|   `-- utils/          # seed, metricas e utilitarios
|
|-- tests/
|-- data/
|-- checkpoints/
|-- outputs/
|-- environment.yml
`-- requirements.txt
```

## Testes

```bash
python -m unittest discover -s tests
```

Alguns testes sao pulados automaticamente quando PyTorch nao esta instalado no ambiente ativo.

## Status

- MNIST split por indice implementado.
- FGSM implementado em PyTorch.
- Detector por mudanca de predicao implementado.
- Varredura de parametros do detector implementada para MNIST.
- Validacao e teste MNIST disponiveis por scripts.
- ImageNet e ataques DeepFool/CW ainda nao foram migrados.

## Roadmap

### Fase 1: Replicacao MNIST

- [x] Preparar o dataset MNIST com split por indice.
- [x] Implementar FGSM em PyTorch.
- [x] Implementar transformacoes do detector.
- [x] Implementar selecao de parametros do detector.
- [x] Implementar validacao e teste do detector.
- [ ] Definir ou importar uma rede-alvo MNIST ja treinada e exportada em TorchScript.
- [ ] Executar o fluxo completo em modo debug.
- [ ] Executar o fluxo completo no split final de teste.

### Fase 2: Ataques Adicionais

- [ ] Implementar ou integrar DeepFool em PyTorch.
- [ ] Implementar ou integrar C&W.
- [ ] Comparar FGSM, DeepFool e C&W no mesmo protocolo.

### Fase 3: ImageNet

- [ ] Preparar subset ImageNet.
- [ ] Integrar redes modernas pre-treinadas.
- [ ] Adaptar transformacoes para imagens RGB maiores.
- [ ] Avaliar custo computacional por ataque.
