# Downloads de Modelos Caffe ImageNet

Este projeto mantem os arquivos de modelos Caffe como artefatos locais em
`artifacts/models/imagenet/`. Eles sao ignorados pelo Git porque os arquivos de
pesos sao grandes.

## Script de download

Use o downloader generico para os assets Caffe ImageNet atualmente suportados
neste repositorio:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --list-models
python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet
python scripts/imagenet/download_caffe_imagenet_assets.py --model alexnet
```

O downloader verifica SHA1 para os pesos BVLC quando o hash oficial e conhecido.

## Status de verificacao

Verificado nesta maquina em 2026-05-22:

- `googlenet`: funciona. O `bvlc_googlenet.caffemodel` local bate com o SHA1
  oficial da BVLC.
- `alexnet`: funciona. O espelho NVIDIA Box baixa um modelo Caffe binario e o
  arquivo bate com o SHA1 oficial do BVLC AlexNet.

CaffeNet e Inception v3 nao ficam mais expostos pelo downloader porque nao foi
possivel verificar downloads diretos e reproduziveis dos pesos:

- BVLC CaffeNet: o `deploy.prototxt` baixa pelo GitHub da BVLC, mas as URLs
  oficiais de `bvlc_reference_caffenet.caffemodel` em
  `dl.caffe.berkeleyvision.org` deram timeout. A URL NVIDIA Box encontrada na
  busca e de AlexNet, nao de CaffeNet, e o SHA1 e diferente.
- Inception v3 Caffe: o prototxt GeekLiB/soeaver baixa, mas os pesos publicados
  estao no Baidu Pan. A API publica `share/list` do Baidu retornou `errno=9019`
  / `errmsg=need verify`, entao o arquivo nao pode ser baixado por um script
  reproduzivel sem verificacao/sessao de navegador. A URL Wayback testada para
  `inception_v3.caffemodel` retornou 404.

## BVLC GoogLeNet

Arquivos esperados:

```text
artifacts/models/imagenet/googlenet/deploy.prototxt
artifacts/models/imagenet/googlenet/bvlc_googlenet.caffemodel
artifacts/models/imagenet/googlenet/ilsvrc_2012_mean.npy
```

Comando:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --model googlenet
```

Fontes:

- Definicao do modelo: `BVLC/caffe`, `models/bvlc_googlenet/deploy.prototxt`.
- URL oficial dos pesos: `http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel`.
- Espelho usado primeiro por este projeto:
  `https://www.deepdetect.com/downloads/platform/pretrained/caffe/googlenet/bvlc_googlenet.caffemodel`.
- SHA1 esperado: `405fc5acd08a3bb12de8ee5e23a96bec22f08204`.

Motivo do espelho: o host antigo da BVLC frequentemente fica indisponivel. O
espelho so e aceito se o SHA1 bater com o hash oficial da BVLC.

## BVLC AlexNet

Arquivos esperados:

```text
artifacts/models/imagenet/alexnet/deploy.prototxt
artifacts/models/imagenet/alexnet/bvlc_alexnet.caffemodel
artifacts/models/imagenet/alexnet/ilsvrc_2012_mean.npy
```

Comando:

```bash
python scripts/imagenet/download_caffe_imagenet_assets.py --model alexnet
```

Fontes:

- Definicao do modelo: `BVLC/caffe`, `models/bvlc_alexnet/deploy.prototxt`.
- Espelho usado primeiro por este projeto:
  `https://nvidia.box.com/shared/static/5j264j7mky11q8emy4q14w3r8hl5v6zh.caffemodel`.
- URL oficial de fallback: `http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel`.
- SHA1 esperado: `9116a64c0fbe4459d18f4bb6b56d647b63920377`.

## Nota metodologica

O artigo menciona CaffeNet, mas os pesos oficiais do BVLC CaffeNet nao puderam
ser baixados de uma fonte direta e reproduzivel durante este setup. O BVLC
AlexNet foi baixado e verificado no lugar dele. AlexNet e arquiteturalmente
proximo da familia CaffeNet, mas nao e o mesmo modelo pre-treinado: o arquivo de
pesos oficial e o SHA1 diferem de `bvlc_reference_caffenet.caffemodel`.

Qualquer experimento que use AlexNet no lugar de CaffeNet deve reportar isso
como desvio metodologico, nao como reproducao fiel do CaffeNet.

## Status de integracao

- GoogLeNet esta conectado aos YAMLs e scripts ImageNet existentes.
- Os assets AlexNet estao disponiveis e verificados, mas ainda falta criar
  wrapper/config antes de usa-los nos scripts de tabelas do artigo.
- CaffeNet e Inception v3 foram excluidos intencionalmente do downloader
  suportado ate que URLs diretas e verificaveis dos pesos sejam encontradas.
