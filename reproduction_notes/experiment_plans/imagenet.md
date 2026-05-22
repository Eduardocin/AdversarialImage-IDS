# Plano de Experimento ImageNet

## 1. Dependências externas

Arquivos necessários que não entram no repositório:

- `data/imagenet/val/`: imagens do conjunto de validação ILSVRC 2012.
- `data/imagenet/val_labels.csv`: CSV sem header com `filename,label_index`.
- `artifacts/models/imagenet/googlenet/deploy.prototxt`: definicao da rede GoogLeNet para Caffe.
- `artifacts/models/imagenet/googlenet/bvlc_googlenet.caffemodel`: pesos treinados da GoogLeNet.
- `artifacts/models/imagenet/googlenet/ilsvrc_2012_mean.npy`: media de canal usada pelo preprocessamento Caffe.
- `artifacts/models/imagenet/alexnet/`: assets BVLC AlexNet, disponiveis por espelho NVIDIA Box. Este modelo e semelhante a familia CaffeNet, mas deve ser marcado como substituicao metodologica porque o peso oficial CaffeNet nao foi baixado.

Fontes de obtenção:

- Caffe Model Zoo: https://github.com/BVLC/caffe/wiki/Model-Zoo
- BVLC GoogLeNet no repositório Caffe: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
- Detalhes locais de download Caffe: `reproduction_notes/caffe_model_downloads.md`
- ImageNet ILSVRC 2012: https://www.image-net.org/challenges/LSVRC/2012/
- Download ImageNet: https://www.image-net.org/download

O diretório de imagens deve conter os nomes usados no CSV, por exemplo
`ILSVRC2012_val_00000001.JPEG`. Os rótulos devem usar os índices de classe
esperados pelo modelo usado na avaliação.

## 2. Caminho sem Caffe

Quando Caffe não está disponível, `scripts/imagenet/googlenet_fgsm.py` registra
`status: bloqueado_caffe` em `results/imagenet/googlenet_fgsm/status.json` e
encerra com código de sucesso. Esse comportamento permite consolidar a matriz de
status sem interromper as demais trilhas.

A validação parcial sem Caffe cobre:

- carregamento e validação do YAML;
- leitura do CSV e tratamento de imagens ausentes ou corrompidas;
- criação de `status.json`;
- teste isolado do gerador FGSM em um grafo TF1 simples;
- teste dos filtros selecionados em imagens HWC normalizadas.

Um wrapper alternativo pode ser usado futuramente para validar o fluxo com outra
arquitetura ImageNet. Essa alternativa deve ser marcada como nao comparavel a
trilha GoogLeNet Caffe, porque muda arquitetura e preprocessamento.

## 3. Diferenças em relação ao MNIST

- O FGSM ImageNet usa `eps=4/255`, enquanto MNIST usa `eps=0.2`. A escala menor
  preserva uma perturbação de poucos níveis em canais RGB normalizados.
- O preprocessamento ImageNet é específico por modelo: resize para 224x224,
  conversão RGB/BGR para Caffe, escala `[0, 255]` e subtração de média por
  canal. MNIST usa imagens 28x28x1 já normalizadas em `[0, 1]`.
- A inferência e a geração de ataques devem usar batches menores por causa do
  custo de memória de imagens 224x224x3 e modelos ImageNet.
- A detecção aplica filtros em imagens NHWC normalizadas antes do preprocessamento
  específico do modelo, preservando a mesma interface dos filtros NumPy.
