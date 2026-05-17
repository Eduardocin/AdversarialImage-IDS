# MNIST reproduction notes

## Decisões de implementação

- Os parâmetros experimentais principais permanecem alinhados ao fluxo MNIST usado no projeto: `eps=0.2`, `n=4500` exemplos por padrão, imagens `28x28x1` em `[0, 1]` e checkpoint salvo com `tf.train.Saver`.
- A avaliação descarta pares em que a imagem limpa já é classificada incorretamente e pares em que o ataque não muda a classe verdadeira.
- A execução usa TensorFlow 1.x em modo grafo com `tf.Session`, `tf.placeholder`, Keras standalone com backend TensorFlow e predições por `cleverhans.utils_tf.model_argmax`.
- Os ajustes de organização não mudam a regra experimental: filtros em módulos separados, registry central sem estado, argumentos via `argparse` e saídas consolidadas em CSV e Markdown.

## Limitações conhecidas

- O ambiente é legado: Python 3.6, TensorFlow 1.15.x, Keras standalone anterior a 2.0, CleverHans 3.1.0 e execução por sessão.
- O contador da faixa `mid` na avaliação por entropia tem um bug documentado: quando uma amostra adversarial da faixa `mid` não é detectada, o falso negativo é contado em `highFN`. Esse comportamento permanece documentado sem correção nesta sprint.
- Os filtros de média `box`, `cross` e `diamond` não estão presentes no código de referência original; eles foram adicionados como filtros locais comparáveis dentro da mesma interface `filter_fn(image) -> image`.

## Como reproduzir

Execute os scripts em ordem a partir da raiz do repositório:

```bash
python scripts/train_mnist.py --epochs 6 --batch-size 128 --learning-rate 0.001
python scripts/generate_mnist_fgsm.py --epsilons 0.2 --samples 4500 --load-model
python scripts/run_mnist_filter_comparison.py --epsilon 0.2 --samples 4500
```

Entradas esperadas:

- Checkpoint TensorFlow em `results/mnist/clean_baseline/checkpoints`.
- Adversariais FGSM em `results/mnist/fgsm/eps_0p2/adversarial_examples.npy`.
- Dados MNIST carregados pelo helper de dados do CleverHans.

Saídas esperadas:

- `results/mnist/final_mnist_results.csv`: uma linha por filtro do registry.
- `results/mnist/final_mnist_report.md`: tabela completa, análise automática e contagem de descartes por filtro.
