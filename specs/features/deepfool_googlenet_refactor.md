# Spec — Refatorar DeepFool/GoogLeNet para suportar deploy_removeSoftmax

## 1. Objetivo

Refatorar o suporte DeepFool para a versão BVLC GoogLeNet de forma que o projeto possa:

- gerar ataques DeepFool usando um modelo de ataque sem softmax (`deploy_removeSoftmax.prototxt`);
- continuar usando o modelo original com softmax (`deploy_original.prototxt`) para classificação e detecção no pipeline regular;
- manter a interface de experimentos e o fluxo atual de Table 10 sem criar uma nova estrutura paralela.

## 2. Contexto

O código atual usa um wrapper Caffe para GoogLeNet que aceita apenas um único `deploy.prototxt` e um único `caffemodel`.

No repositório original do DeepFool, os autores removem a camada softmax do modelo usado apenas para geração de ataques, porque a versão com softmax produz gradientes menos eficazes para a regra de atualização do DeepFool.

O segundo repositório citado documenta claramente o uso de dois arquivos distintos:

- `deploy_removeSoftmax.prototxt` para gerar DeepFool;
- `deploy_original.prototxt` para classificar e detectar imagens.

Isso mostra que apenas baixar e colocar arquivos `.prototxt` em `artifacts` NÃO é suficiente para rodar os experimentos, porque o pipeline atual precisa de suporte explícito para instanciar e usar dois modelos diferentes em momentos distintos.

## 3. Requisitos funcionais

### 3.1 Suporte de configuração

Estender a configuração da Table 10 / GoogLeNet para aceitar:

- `model.deploy_proto`: o prototxt padrão usado para previsão/classificação;
- `model.attack_deploy_proto`: o prototxt opcional usado apenas para gerar gradientes em DeepFool;
- `model.caffemodel`: pesos Caffe compartilhados entre as duas definições de rede;
- `model.mean_file`, `model.use_gpu`, `model.batch_size` como hoje.

### 3.2 Wrapper de modelo

Refatorar o wrapper `GoogLeNetCaffeWrapper` ou adicionar um novo wrapper de ataque que:

- carrega a definição padrão para previsão e detecção;
- opcionalmente carrega uma segunda definição sem softmax para o caminho de ataque;
- expõe `predict_batch`, `predict_preprocessed_batch`, `predict`, `predict_label` usando o modelo original;
- expõe `gradient(image, class_id)` usando o modelo sem softmax quando estiver configurado;
- mantém comportamento legado quando `attack_deploy_proto` não estiver presente.

### 3.3 Integração com DeepFool

O ataque `generate_deepfool` deve continuar recebendo um único objeto `model`. O objeto pode encapsular internamente:

- a lógica de previsão com o modelo original;
- a lógica de gradiente com o modelo sem softmax;
- ou a lógica de fallback para um único modelo quando `attack_deploy_proto` não existir.

O `generate_deepfool` não deve ser responsável por carregar dois prototxts diretamente.

### 3.3.1 O que deve permanecer em `src/deepdetector/attacks/deepfool.py`

O arquivo `deepfool.py` deve manter apenas a implementação genérica do algoritmo DeepFool.

- `generate_deepfool` permanece como a interface pública do ataque.
- A lógica de batching, validação de parâmetros, atualização iterativa e clipping permanece no arquivo.
- As funções auxiliares `_as_image_batch`, `_scores`, `_gradient`, `_class_candidates` e `_deepfool_one` devem continuar existindo como helpers do ataque.
- O arquivo deve continuar usando o contrato genérico de wrappers de modelo: `predict_preprocessed_batch`, `predict_batch`, `scores`, `gradient`, ou `callable(model)`.
- Qualquer mudança deve ser limitada a ajustar `deepfool.py` para trabalhar com o modelo wrapper revisado que fornece gradientes via rede sem softmax.

### 3.3.2 O que deve sair de `src/deepdetector/attacks/deepfool.py`

O arquivo não deve conter lógica específica de modelo ou de configuração de GoogLeNet.

- Não deve carregar `deploy.prototxt`, `attack_deploy_proto`, ou qualquer arquivo de modelo.
- Não deve instanciar wrappers Caffe, nem chamar Caffe diretamente.
- Não deve conter ramificações específicas para GoogLeNet, ImageNet ou redes sem softmax.
- Não deve conhecer detalhes de configuração de experimento ou paths em `artifacts`.

### 3.4 Fluxo de experimentos Table 10

A linha 7 de `table_10_googlenet` deve usar esta refatoração sem precisar de um runner especial.

A configuração deve continuar sendo:

```yaml
attack:
  name: deepfool
  max_iter: 50
  overshoot: 0.02
  clip_min: 0.0
  clip_max: 1.0
```

Mas o `model` deve poder fornecer o `attack_deploy_proto` quando disponível.

## 4. Critérios de aceitação

1. `artifacts/models/imagenet/googlenet/` deve poder armazenar tanto o `deploy_original.prototxt` quanto o `deploy_removeSoftmax.prototxt`.
2. O pipeline deve usar `deploy_original.prototxt` para inferência normal e o `deploy_removeSoftmax.prototxt` somente para a geração de gradientes DeepFool.
3. Se `attack_deploy_proto` não estiver configurado, o wrapper deve usar `deploy_proto` para ambos os caminhos e continuar funcionando com o comportamento legado.
4. O `deepfool` attack registry deve continuar funcionando pelo dispatcher de ataques existente.
5. Devem existir testes que cubram:
   - criação do wrapper com `attack_deploy_proto` opcional;
   - fallback para o mesmo `deploy_proto` quando `attack_deploy_proto` estiver ausente;
   - `generate_deepfool` usando um modelo que devolve gradiente a partir da rede sem softmax;
   - `table_10_googlenet` mantendo o status `implemented`/`planned` e sem `blocked` para linha 7.
6. A refatoração não deve criar uma nova pasta de runner ou um novo tipo de experimento específico para DeepFool.

## 5. Regras de implementação

- Não alterar os resultados oficiais da Table 10 além do necessário para habilitar o ataque DeepFool.
- Não traduzir o `deploy_removeSoftmax.prototxt` para outro formato; use o prototxt como especificado.
- Não transformar `generate_deepfool` em um ataque específico de GoogLeNet.
- Manter as abstrações de wrapper e attack separadas.
- Não escrever arquivos em disco durante a geração de ataques.

## 6. Caso de uso típico

1. O usuário configura o experimento com `model.deploy_proto` e `model.attack_deploy_proto`.
2. O runner da Table 10 instancia o wrapper.
3. O fluxo de avaliação usa o wrapper para classificar imagens limpas.
4. O fluxo de ataque chama `generate_deepfool(model, images, ...)`.
5. O wrapper calcula gradientes pelo modelo sem softmax e retorna os adversários.
6. O detector ou avaliador usa o modelo original para validar o ataque.

## 7. Fora do escopo

- Suporte a outras redes além de BVLC GoogLeNet para este refactor.
- Modificações no `scripts/article_reproduction/*` além de ajustes de wrapper/config necessários.
- Reposicionar experimentos fora da Table 10.
- Alterar o formato de saída de `metrics.csv` ou `metrics.json` existentes.
- Adicionar novo runner específico para `deploy_removeSoftmax.prototxt`.
