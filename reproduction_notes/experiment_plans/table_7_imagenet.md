# Table 7 ImageNet Plan

## Objetivo

Reproduzir a Table 7 do artigo para medir o efeito isolado de filtros espaciais
em exemplos adversariais ImageNet de alta entropia. Nesta etapa, cada filtro
candidato e avaliado diretamente depois de quantizacao escalar com 6 intervalos
(`step=43` em escala 0-255).

## Dataset

O experimento usa a trilha ImageNet de treino configurada em
`configs/article_reproduction/imagenet_table_7.yaml`. As classes esperadas sao:

- Goldfish, label ImageNet 1.
- Pineapple, label ImageNet 953.
- Clock, label ImageNet 530.

Os dados locais devem estar em `data/train/Goldfish`, `data/train/Pineapple` e
`data/train/Clock`.

## Modelo

O classificador e o BVLC GoogLeNet via Caffe, carregado pelo wrapper
`GoogLeNetCaffeWrapper`. A defesa nao altera o modelo; ela atua apenas como
pre-processamento antes da nova predicao.

## Ataque

O ataque e FGSM com `epsilon_255=1.0`, equivalente a `1/255` quando as imagens
estao normalizadas em `[0, 1]`. O runner aceita adversariais salvos por
`--adv-path` ou por `attack.adversarial_path`; se o modelo expuser handles
TensorFlow compativeis, o script tambem pode gerar os exemplos.

## Alta Entropia

A Table 7 considera apenas imagens com entropia maior que `5.0`. A entropia e
calculada por canal em imagens `C x H x W` na escala `[0, 255]`, usando
histograma de 256 niveis e entropia de Shannon. O valor final e a media das
entropias dos canais.

## Mascaras Avaliadas

Cada imagem passa por:

1. Quantizacao escalar com `step=43`.
2. Media espacial com uma mascara candidata.

As mascaras avaliadas sao:

- `cross`: tamanhos 3, 5, 7 e 9.
- `diamond`: tamanhos 3, 5, 7 e 9.
- `box`: tamanhos 3, 5, 7 e 9.

## Contagens

Para exemplos adversariais de alta entropia, o detector compara `C(x_adv)` com
`C(T(x_adv))`. Se a predicao muda, conta `TP`; se nao muda, conta `FN`.

Para imagens benignas de alta entropia, o detector compara `C(x)` com
`C(T(x))`. Se a predicao muda, conta `FP`; se nao muda, nao adiciona erro.

As metricas finais sao:

- `Recall = TP / (TP + FN)`.
- `Precision = TP / (TP + FP)`.
- `F1 = 2 * Recall * Precision / (Recall + Precision)`.

## Diferenca Para Table 9

A Table 7 avalia filtros espaciais candidatos isoladamente. Ela nao usa a regra
final `chooseCloserFilter` nem a combinacao adaptativa da Table 9. A Table 9
escolhe o filtro final e combina quantizacao e suavizacao por uma regra de
proximidade; essa decisao metodologica fica fora deste experimento.
