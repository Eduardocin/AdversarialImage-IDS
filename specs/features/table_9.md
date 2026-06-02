# Spec — Table 9: Reprodução do Filtro Final contra FGSM

## 1. Objetivo

Implementar a reprodução da **Table 9** do artigo *Detecting Adversarial Image Examples in Deep Neural Networks with Adaptive Noise Reduction*, avaliando o **filtro final de detecção** contra ataques **FGSM**.

A execução deve orquestrar dois fluxos:

1. **MNIST M1 + FGSM**
2. **ImageNet GoogLeNet/Caffe + FGSM**

O resultado final deve agregar os contadores dos dois fluxos por split:

- `Training`
- `Validation`

E gerar a tabela final no formato:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
````

---

## 2. Estado atual da branch

Na branch `fix/table7`, já existem partes importantes que devem ser reutilizadas:

* filtros básicos no `FILTER_REGISTRY`;
* filtro da Table 7 para suavização espacial;
* infraestrutura ImageNet/Caffe;
* infraestrutura de quantização adaptativa da Table 6;
* scripts de reprodução para outras tabelas.

Para a Table 9, o fluxo oficial deve ser descrito e executado pelos seguintes
arquivos:

```text
configs/experiments.yaml
src/deepdetector/experiments/table9_runner.py
src/deepdetector/filters/article_final.py
```

Também é necessário registrar o filtro final no registry como:

```python
"article_final"
```

---

## 3. Dataset correto

A divisão correta dos datasets, conforme a Table 2 do artigo, é:

| Dataset  | Training                                               | Validation        | Test                                          |
| -------- | ------------------------------------------------------ | ----------------- | --------------------------------------------- |
| MNIST    | No. `0~4499`                                           | No. `4500~5499`   | No. `5500~9999`                               |
| ImageNet | Goldfish `(648)`<br>Pineapple `(520)`<br>Clock `(455)` | Jellyfish `(618)` | Zebra `(503)`<br>Panda `(501)`<br>Cab `(485)` |

Para a **Table 9**, esta sessão deve usar apenas:

* `Training`
* `Validation`

O split `Test` não entra nesta reprodução.

---

## 4. Resultado de referência da Table 9

| Split      |   TP |  FN |  FP | Recall | Precision |     F1 |
| ---------- | ---: | --: | --: | -----: | --------: | -----: |
| Training   | 3324 | 266 | 108 | 92.59% |    96.85% | 94.67% |
| Validation | 1028 |  61 |  35 | 94.40% |    96.71% | 95.54% |

Esses valores devem ser usados apenas como referência no relatório.

O código **não deve hardcodar esses valores como resultado calculado**.

---

## 5. Configuração principal

A configuração oficial deve residir em:

```text
configs/experiments.yaml
```

Esse YAML é a fonte única de configuração operacional da Table 9.

### YAML proposto

```yaml
table_9:
  kind: table_9
  output_dir: results/experiments/table_9
  datasets: [mnist, imagenet]
  split_order: [train, validation]
  mnist:
    dataset:
      name: mnist
      split: test
      slices:
        - name: Training
          start: 0
          end: 4500
        - name: Validation
          start: 4500
          end: 5500
    attack:
      name: fgsm
      epsilon: 0.2
      clip_min: 0.0
      clip_max: 1.0
    filter:
      name: proposed_detection_filter
      type: proposed_detection_filter
  imagenet:
    dataset:
      name: imagenet
      splits:
        train:
          - name: goldfish
            label: 1
            path: data/imagenet/train/goldfish
          - name: pineapple
            label: 953
            path: data/imagenet/train/pineapple
          - name: digital_clock
            label: 530
            path: data/imagenet/train/digital_clock
        validation:
          - name: jellyfish
            label: 107
            path: data/imagenet/validation/jellyfish
    attack:
      name: fgsm
      epsilon_255: 1.0
      clip_min: 0.0
      clip_max: 255.0
    filter:
      name: proposed_detection_filter
      type: proposed_detection_filter
```

---

## 6. Configs de ataques

Para a Table 9, **não usar YAML separado em `configs/attacks/`**.

Os ataques devem ficar inline em `table_9.yaml`, porque o ataque é parte do contrato experimental da tabela.

Exemplo MNIST:

```yaml
attack:
  name: fgsm
  epsilon: 0.2
  clip_min: 0.0
  clip_max: 1.0
```

Exemplo ImageNet:

```yaml
attack:
  name: fgsm
  epsilon_255: 1.0
  clip_min: 0.0
  clip_max: 255.0
```

A pasta `configs/attacks/` só deve ser mantida se houver reutilização real por múltiplos experimentos genéricos. Caso contrário, esses YAMLs devem ser removidos ou movidos para:

```text
configs/archive/attacks/
```

---

## 7. Filtro final da Table 9

Criar:

```text
src/deepdetector/filters/article_final.py
```

Função pública esperada:

```python
def article_final_detection_filter(image: np.ndarray) -> np.ndarray:
    ...
```

O filtro final combina:

1. quantização adaptativa da Table 6;
2. suavização espacial inspirada na Table 7;
3. escolha pixel a pixel do valor filtrado mais próximo do valor original.

---

## 8. Regra do filtro final

### 8.1 Por entropia

| Faixa de entropia      |              Quantização | Suavização            |
| ---------------------- | -----------------------: | --------------------- |
| `entropy < 4.0`        | 2 intervalos, step `128` | Não                   |
| `4.0 <= entropy < 5.0` |  4 intervalos, step `64` | Não                   |
| `entropy >= 5.0`       |  6 intervalos, step `43` | Sim, cross mask `7x7` |

### 8.2 Alta entropia

Para imagens com entropia maior ou igual a `5.0`:

```python
quantized = scalar_quantization_or_step_quantization(image, step=43)
smoothed = cross_smoothing_7x7(quantized)

output = np.where(
    np.abs(quantized - image) <= np.abs(smoothed - image),
    quantized,
    smoothed,
)
```

O `cross_smoothing_7x7` deve usar máscara cross de raio `3`.

---

## 9. Atenção à escala das imagens

A implementação deve suportar três espaços de imagem:

1. **MNIST normalizado**

   * formato: `HWC` ou `NHWC`
   * escala: `[0.0, 1.0]`

2. **ImageNet Inception v3**

   * formato: `HWC` ou `NHWC`
   * escala: `[-0.5, 0.5]`
   * o filtro deve converter internamente para `[0.0, 255.0]` quando calcular entropia, quantização e suavização, e restaurar para `[-0.5, 0.5]` antes de retornar.

3. **ImageNet Caffe**

   * formato: `CHW`
   * canais: `BGR`
   * escala: `[0.0, 255.0]`

O filtro final deve preservar a escala de entrada:

* se a imagem entra em `[0.0, 1.0]`, deve sair em `[0.0, 1.0]`;
* se a imagem entra em `[-0.5, 0.5]`, deve sair em `[-0.5, 0.5]`;
* se a imagem entra em `[0.0, 255.0]`, deve sair em `[0.0, 255.0]`.

Para ImageNet, a quantização deve acontecer em espaço de pixel `0..255`, preservando o layout original no retorno.

Não aplicar `step=128`, `step=64` ou `step=43` diretamente em imagem normalizada `[0,1]` ou centralizada `[-0.5,0.5]` sem conversão apropriada.

---

## 10. Registro do filtro

Atualizar:

```text
src/deepdetector/filters/registry.py
```

Adicionar:

```python
from deepdetector.filters.article_final import article_final_detection_filter
```

E registrar:

```python
("article_final", article_final_detection_filter)
```

Também atualizar:

```text
src/deepdetector/filters/__init__.py
```

Para exportar:

```python
article_final_detection_filter
```

---

## 11. Uso da infraestrutura existente

A Table 9 deve reutilizar infraestrutura já existente sempre que possível.

### 11.1 Table 6

Reutilizar a lógica de:

* entropia;
* quantização adaptativa;
* step `128`, `64`, `43`;
* tratamento de escala Caffe quando aplicável.

### 11.2 Table 7

A Table 7 fornece suavização espacial, mas **não substitui o filtro final da Table 9**.

O filtro da Table 9 deve usar suavização cross `7x7` apenas para imagens de alta entropia e depois aplicar a regra de escolha pixel a pixel.

### 11.3 ImageNet/Caffe

Reutilizar o wrapper existente de `GoogLeNetCaffeWrapper`.

O fluxo ImageNet deve usar o mesmo padrão dos scripts ImageNet já existentes:

* carregar imagem local;
* converter para entrada Caffe;
* predizer com Caffe;
* gerar FGSM via gradiente;
* aplicar filtro;
* predizer imagem filtrada.

---

## 12. Script principal

O ponto único de execução da Table 9 deve ser o runner centralizado:

```bash
python scripts/run_experiment.py --experiment table_9
```

Não deve existir script público separado para a Table 9.

---

## 13. Argumentos do script

A Table 9 usa apenas os argumentos globais de `scripts/run_experiment.py`.

| Argumento | Obrigatório | Descrição |
| --- | --- | --- |
| `--experiment table_9` | Sim | Seleciona a reprodução oficial da Table 9 |

---

## 14. Validação de configuração

A configuração consolidada deve declarar:

* `kind: table_9`;
* componentes internos `mnist` e `imagenet`;
* splits `train` e `validation`;
* filtro `proposed_detection_filter` nos dois componentes;
* paths ImageNet/Caffe necessários para o fluxo GoogLeNet.

A execução completa pode falhar se os assets Caffe/GoogLeNet estiverem ausentes.
Ela não deve fabricar métricas ImageNet quando esses assets não existem.

---

## 15. Execução reduzida

Execuções reduzidas devem ser feitas por ajustes explícitos em
`configs/experiments.yaml`, por exemplo `imagenet.dataset.n_samples`, e não por
um CLI paralelo da Table 9.

---

## 16. Regra de detecção

Para uma imagem `x` e seu filtro `T(x)`:

```text
C(x) == C(T(x))  -> benigno
C(x) != C(T(x))  -> adversarial
```

Para exemplos adversariais:

* `TP`: `C(x_adv) != C(T(x_adv))`
* `FN`: `C(x_adv) == C(T(x_adv))`

Para imagens limpas:

* `FP`: `C(x_clean) != C(T(x_clean))`

---

## 17. Exclusões

A Table 9 deve excluir:

1. imagens limpas que o modelo original classifica errado;
2. ataques FGSM que não alteram a predição original.

Essas exclusões devem ser controladas por:

```yaml
evaluation:
  exclude_clean_errors: true
  exclude_failed_attacks: true
```

---

## 18. Agregação correta

A Table 9 combina resultados MNIST e ImageNet.

A agregação deve ser feita por split:

```text
Training = MNIST Training + ImageNet Training
Validation = MNIST Validation + ImageNet Validation
```

A métrica deve ser calculada **após somar os contadores**.

Correto:

```text
TP_total = sum(TP_i)
FN_total = sum(FN_i)
FP_total = sum(FP_i)

Recall = TP_total / (TP_total + FN_total)
Precision = TP_total / (TP_total + FP_total)
F1 = 2 * Recall * Precision / (Recall + Precision)
```

Incorreto:

```text
F1_final = média(F1_mnist, F1_imagenet)
```

Não calcular média simples de métricas por fluxo.

---

## 19. Saídas esperadas

A execução padrão deve gerar apenas:

```text
results/experiments/table_9/metrics.csv
results/experiments/table_9/metrics.json
```

Não gerar por padrão:

```text
table_9_diagnostics.csv
table_9.md
status.json
```

Se diagnóstico for útil durante desenvolvimento, ele deve ser opcional, por exemplo via:

```bash
--debug
```

Mas não deve fazer parte do fluxo padrão nem dos critérios de aceite.

---

## 20. CSV final

O CSV final deve ter exatamente as colunas:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
```

E exatamente duas linhas:

```text
train
validation
```

Exemplo:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
train,3324,266,108,92.59,96.85,94.67
validation,1028,61,35,94.40,96.71,95.54
```

Os valores acima são exemplo de referência do artigo. Os valores gerados localmente podem variar conforme assets, modelo, ambiente e subset disponível.

---

## 21. Markdown final

O fluxo oficial da Table 9 não deve gerar Markdown.

---

## 22. Status JSON

O fluxo oficial da Table 9 não deve gerar `status.json`. O resultado
estruturado deve ficar em `metrics.json`.

---

## 23. Validação mínima

Não é necessário criar uma suíte extensa de testes automatizados para esta sessão.

A validação mínima deve ser:

1. `python scripts/run_experiment.py --experiment table_9` usa `kind: table_9`;
2. a configuração declara `mnist` e `imagenet` como componentes internos;
3. `metrics.csv` tem o schema correto;
4. `metrics.csv` tem as linhas `train` e `validation`;
5. `article_final` existe no `FILTER_REGISTRY`;
6. métricas são calculadas após somar contadores.

Testes automatizados opcionais:

```text
test_article_final_filter_is_registered
test_article_final_filter_preserves_inception_centered_scale
test_table_9_aggregation_sums_counts_before_metrics
test_table_9_csv_schema
```

---

## 24. Critérios de aceite

A implementação será considerada pronta quando:

1. existir:

```text
configs/experiments.yaml
```

2. existir:

```text
src/deepdetector/experiments/table9_runner.py
```

3. existir:

```text
src/deepdetector/filters/article_final.py
```

4. `article_final` estiver registrado em:

```text
src/deepdetector/filters/registry.py
```

5. `article_final_detection_filter` estiver exportado em:

```text
src/deepdetector/filters/__init__.py
```

6. o comando abaixo executar:

```bash
python scripts/run_experiment.py --experiment table_9
```

7. a execução gerar:

```text
metrics.csv
metrics.json
```

8. o CSV final tiver exatamente:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
```

9. o CSV final tiver exatamente os splits:

```text
train
validation
```

10. o fluxo não gerar `table_9_diagnostics.csv`, `table_9.md` ou `status.json` por padrão;

11. métricas forem calculadas por soma de contadores antes de `Recall`, `Precision` e `F1`.

---

## 25. Diagrama do fluxo

```mermaid
flowchart TD
    A[configs/experiments.yaml] --> B[scripts/run_experiment.py --experiment table_9]

    B --> C[Carregar YAML]
    C --> D[Resolver filtro proposed_detection_filter]

    D --> H[Fluxo MNIST M1 + FGSM]
    D --> I[Fluxo ImageNet GoogLeNet + FGSM]

    H --> H1[Training MNIST 0-4499]
    H --> H2[Validation MNIST 4500-5499]

    I --> I1[Training Goldfish, Pineapple, Clock]
    I --> I2[Validation Jellyfish]

    H1 --> J[Gerar FGSM]
    H2 --> J
    I1 --> K[Gerar FGSM Caffe]
    I2 --> K

    J --> L[Aplicar filtro article_final]
    K --> L

    L --> M[Comparar C(x) com C(T(x))]
    M --> N[Computar TP, FN, FP]

    N --> O[Agregar por split]
    O --> P[Somar contadores]
    P --> Q[Calcular Recall, Precision, F1]

    Q --> R[metrics.csv]
    Q --> S[metrics.json]
```

---

## 26. Comando final

```bash
python scripts/run_experiment.py --experiment table_9
```
