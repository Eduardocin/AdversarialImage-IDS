````md
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

Porém, para a Table 9 ainda é necessário garantir a existência dos seguintes arquivos:

```text
configs/article_reproduction/table_9.yaml
scripts/article_reproduction/table_9.py
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

Criar:

```text
configs/article_reproduction/table_9.yaml
```

Esse YAML deve ser a fonte única de configuração da reprodução da Table 9.

### YAML proposto

```yaml
experiment:
  name: table_9
  article_table: 9
  objective: Reproduce Table 9 using the final entropy-aware detection filter against FGSM.
  attack_method: fgsm
  evaluated_method: article_final_detection_filter

orchestration:
  flows:
    - mnist_m1_fgsm
    - imagenet_googlenet_fgsm
  aggregate_by: split
  aggregate_counts_before_metrics: true

splits:
  - Training
  - Validation

datasets:
  mnist:
    enabled: true
    flow: mnist_m1_fgsm
    model: M1
    checkpoint_dir: artifacts/models/mnist/m1/clean_baseline/checkpoints
    image_shape: [28, 28, 1]
    value_range: [0.0, 1.0]
    attack:
      name: fgsm
      epsilon: 0.2
      clip_min: 0.0
      clip_max: 1.0
    slices:
      Training:
        start: 0
        end: 4500
      Validation:
        start: 4500
        end: 5500

  imagenet:
    enabled: true
    flow: imagenet_googlenet_fgsm
    model: googlenet_caffe
    image_shape: [224, 224, 3]
    value_range: [0.0, 1.0]
    model_assets:
      model_dir: artifacts/models/imagenet/googlenet
      deploy_proto: artifacts/models/imagenet/googlenet/deploy.prototxt
      caffemodel: artifacts/models/imagenet/googlenet/bvlc_googlenet.caffemodel
      mean_file: null
      use_gpu: false
      batch_size: 32
    attack:
      name: fgsm
      epsilon_255: 1.0
      clip_min: 0.0
      clip_max: 255.0
    classes:
      Training:
        goldfish:
          label_id: 1
          expected_count: 648
          path: data/imagenet/train/goldfish
        pineapple:
          label_id: 953
          expected_count: 520
          path: data/imagenet/train/pineapple
        clock:
          label_id: 530
          expected_count: 455
          path: data/imagenet/train/digital_clock
      Validation:
        jellyfish:
          label_id: 107
          expected_count: 618
          path: data/imagenet/validation/jellyfish

detection:
  filter_name: article_final
  method: prediction_change
  rule: adversarial_if_prediction_changes_after_filtering

evaluation:
  exclude_clean_errors: true
  exclude_failed_attacks: true
  dry_run_sample_size: 4

metrics:
  fields:
    - split
    - TP
    - FN
    - FP
    - recall_percent
    - precision_percent
    - f1_percent

reference:
  Training:
    TP: 3324
    FN: 266
    FP: 108
    recall_percent: 92.59
    precision_percent: 96.85
    f1_percent: 94.67
  Validation:
    TP: 1028
    FN: 61
    FP: 35
    recall_percent: 94.40
    precision_percent: 96.71
    f1_percent: 95.54

output:
  results_dir: results/article_reproduction/table_9
  csv: table_9.csv
  markdown: table_9.md
  status_json: status.json
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

A implementação deve suportar dois espaços de imagem:

1. **MNIST normalizado**

   * formato: `HWC` ou `NHWC`
   * escala: `[0.0, 1.0]`

2. **ImageNet Caffe**

   * formato: `CHW`
   * canais: `BGR`
   * escala: `[0.0, 255.0]`

O filtro final deve preservar a escala de entrada:

* se a imagem entra em `[0.0, 1.0]`, deve sair em `[0.0, 1.0]`;
* se a imagem entra em `[0.0, 255.0]`, deve sair em `[0.0, 255.0]`.

Para ImageNet, a quantização deve acontecer no espaço Caffe `CHW/BGR/0..255`.

Não aplicar `step=128`, `step=64` ou `step=43` diretamente em imagem normalizada `[0,1]` sem conversão apropriada.

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

Criar:

```text
scripts/article_reproduction/table_9.py
```

Esse script deve ser o ponto único de execução da Table 9.

Ele deve suportar:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml
```

Também deve suportar:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --dry-run
```

E:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --sample-size 8
```

---

## 13. Argumentos do script

| Argumento       | Obrigatório | Descrição                                                           |
| --------------- | ----------- | ------------------------------------------------------------------- |
| `--config`      | Não         | Caminho para `table_9.yaml`                                         |
| `--dry-run`     | Não         | Valida configuração e dependências sem executar reprodução completa |
| `--sample-size` | Não         | Executa fluxo real com amostra pequena                              |
| `--output-dir`  | Não         | Sobrescreve diretório de saída do YAML                              |

---

## 14. Dry run

O `--dry-run` deve validar:

* leitura do YAML;
* existência dos splits `Training` e `Validation`;
* existência dos fluxos `mnist_m1_fgsm` e `imagenet_googlenet_fgsm`;
* existência do filtro `article_final` no registry;
* resolução dos paths principais;
* capacidade de carregar pequena amostra de MNIST, se disponível;
* existência ou ausência dos assets ImageNet/Caffe.

O `dry-run` **não deve falhar apenas porque Caffe não está instalado**.

Se Caffe ou os assets do GoogLeNet estiverem ausentes, o fluxo ImageNet deve ser marcado como:

```text
blocked_imagenet_caffe
```

ou status equivalente em `status.json`.

O dry-run deve gerar:

```text
results/article_reproduction/table_9/status.json
```

---

## 15. Sample size

O argumento `--sample-size` deve executar o fluxo real com poucas amostras.

Comportamento esperado:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --sample-size 8
```

Deve:

* usar no máximo 8 amostras por split no MNIST;
* usar no máximo 8 amostras por classe no ImageNet;
* aplicar ataque FGSM;
* aplicar filtro final;
* calcular métricas;
* gerar `table_9.csv`;
* gerar `table_9.md`;
* gerar `status.json`.

Se ImageNet estiver bloqueado por falta de Caffe/assets, o script deve registrar isso no `status.json` e não fabricar métricas para ImageNet.

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
results/article_reproduction/table_9/table_9.csv
results/article_reproduction/table_9/table_9.md
results/article_reproduction/table_9/status.json
```

Não gerar por padrão:

```text
table_9_diagnostics.csv
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
Training
Validation
```

Exemplo:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
Training,3324,266,108,92.59,96.85,94.67
Validation,1028,61,35,94.40,96.71,95.54
```

Os valores acima são exemplo de referência do artigo. Os valores gerados localmente podem variar conforme assets, modelo, ambiente e subset disponível.

---

## 21. Markdown final

O arquivo:

```text
results/article_reproduction/table_9/table_9.md
```

Deve conter:

* descrição curta do experimento;
* tabela final obtida;
* tabela de referência do artigo;
* observações sobre execução parcial, se houver;
* indicação de fluxos bloqueados, se houver;
* path do CSV gerado.

Não incluir diagnóstico detalhado por imagem.

---

## 22. Status JSON

O arquivo:

```text
results/article_reproduction/table_9/status.json
```

Deve conter, no mínimo:

```json
{
  "status": "completed | partial | blocked | failed",
  "config_path": "configs/article_reproduction/table_9.yaml",
  "results_csv": "results/article_reproduction/table_9/table_9.csv",
  "markdown": "results/article_reproduction/table_9/table_9.md",
  "enabled_flows": ["mnist_m1_fgsm", "imagenet_googlenet_fgsm"],
  "completed_flows": [],
  "skipped_flows": [],
  "sample_size": null,
  "warnings": []
}
```

Campos opcionais:

```json
{
  "aggregate_counters": {},
  "per_flow_counters": {}
}
```

Esses campos opcionais não devem ser requisito rígido.

---

## 23. Validação mínima

Não é necessário criar uma suíte extensa de testes automatizados para esta sessão.

A validação mínima deve ser:

1. `--dry-run` executa e escreve `status.json`;
2. `--sample-size 8` executa quando os assets necessários estão disponíveis;
3. `table_9.csv` tem o schema correto;
4. `table_9.csv` tem as linhas `Training` e `Validation`;
5. `article_final` existe no `FILTER_REGISTRY`;
6. métricas são calculadas após somar contadores.

Testes automatizados opcionais:

```text
test_article_final_filter_is_registered
test_table_9_aggregation_sums_counts_before_metrics
test_table_9_csv_schema
```

---

## 24. Critérios de aceite

A implementação será considerada pronta quando:

1. existir:

```text
configs/article_reproduction/table_9.yaml
```

2. existir:

```text
scripts/article_reproduction/table_9.py
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
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --dry-run
```

7. o comando abaixo executar com assets disponíveis:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --sample-size 8
```

8. a execução gerar:

```text
table_9.csv
table_9.md
status.json
```

9. o CSV final tiver exatamente:

```csv
split,TP,FN,FP,recall_percent,precision_percent,f1_percent
```

10. o CSV final tiver exatamente os splits:

```text
Training
Validation
```

11. o fluxo não gerar `table_9_diagnostics.csv` por padrão;

12. a ausência de Caffe/GoogLeNet não causar falha fatal em `--dry-run`;

13. métricas forem calculadas por soma de contadores antes de `Recall`, `Precision` e `F1`.

---

## 25. Diagrama do fluxo

```mermaid
flowchart TD
    A[configs/article_reproduction/table_9.yaml] --> B[scripts/article_reproduction/table_9.py]

    B --> C[Carregar YAML]
    C --> D[Resolver filtro article_final no FILTER_REGISTRY]

    D --> E{Modo de execução}

    E -->|dry-run| F[Validar config, splits, filtro e assets]
    F --> G[status.json]

    E -->|execução| H[Fluxo MNIST M1 + FGSM]
    E -->|execução| I[Fluxo ImageNet GoogLeNet + FGSM]

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

    Q --> R[table_9.csv]
    Q --> S[table_9.md]
    Q --> T[status.json]
```

---

## 26. Comandos finais

Dry run:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --dry-run
```

Execução pequena:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml \
  --sample-size 8
```

Execução completa:

```bash
python scripts/article_reproduction/table_9.py \
  --config configs/article_reproduction/table_9.yaml
```

```
```