# Notas de Configuracao do Caffe

## Quando o Caffe e necessario

O Caffe e necessario na trilha ImageNet quando a reproducao precisa preservar o
fluxo original baseado em modelos ImageNet compativeis com Caffe.

A trilha MNIST pode comecar com TensorFlow 1.x, Keras e CleverHans. A trilha
ImageNet nao deve ser tratada como equivalente enquanto o carregamento do modelo
Caffe, preprocessamento, rotulos e avaliacao nao forem reproduzidos ou enquanto
o desvio nao estiver documentado explicitamente.

## Papel esperado na reproducao

- Carregar definicoes e pesos de modelos ImageNet compativeis com Caffe.
- Preservar o preprocessamento ImageNet usado pela implementacao original.
- Gerar e avaliar exemplos adversariais contra a mesma familia de modelos usada
  no projeto de referencia.
- Comparar o comportamento do detector antes e depois da reducao adaptativa de
  ruido.

## Ambiente Conda

A trilha ImageNet espera o modulo Python `caffe` do pacote Caffe do
`conda-forge`. Ele esta declarado em `envs/environment.yml` como dependencia Conda
porque `pycaffe` nao e instalado via `pip`.

Para criar ou atualizar o ambiente local:

```bash
conda env create -f envs/environment.yml
conda activate adversarialimage-ids-legacy
python scripts/dev/smoke_test.py
```

Para um ambiente existente:

```bash
conda env update -n adversarialimage-ids-legacy -f envs/environment.yml
conda activate adversarialimage-ids-legacy
python scripts/dev/smoke_test.py
```

A fonte esperada do pacote e `conda-forge::caffe`. A reproducao atual esta
configurada para CPU por padrao (`use_gpu: false` nos YAMLs ImageNet). Se uma
build com GPU for usada depois, registre aqui as versoes de CUDA/cuDNN e
qualquer mudanca de canal antes de executar experimentos.

## Assets dos modelos

Os assets dos modelos sao artefatos locais. A trilha ImageNet executavel hoje
espera GoogLeNet nestes caminhos:

- `artifacts/models/imagenet/googlenet/deploy.prototxt`
- `artifacts/models/imagenet/googlenet/bvlc_googlenet.caffemodel`
- `artifacts/models/imagenet/googlenet/ilsvrc_2012_mean.npy`

Baixe com:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet
```

O downloader generico tambem conhece o AlexNet, que foi o fallback verificavel
mais proximo encontrado durante a busca por pesos CaffeNet:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --list-models
python scripts/imagenet/download_caffe_imagenet_assets.py --model alexnet
```

A definicao do GoogLeNet vem de `BVLC/caffe` em
`models/bvlc_googlenet/deploy.prototxt`. Os pesos pre-treinados vem da URL
documentada no `readme.md` desse diretorio, com SHA1
`405fc5acd08a3bb12de8ee5e23a96bec22f08204`. Como o host antigo da BVLC pode
ficar indisponivel, o downloader do projeto tenta primeiro um espelho da
DeepDetect e ainda exige que o SHA1 oficial bata.

Veja `reproduction_notes/caffe_model_downloads.md` para caminhos, fontes,
decisoes sobre espelhos e a justificativa de por que CaffeNet e Inception v3 nao
ficam expostos como downloads suportados.
