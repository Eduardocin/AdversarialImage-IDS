# SPEC — ImageNet New Classes Separate Selected Table 10 Evaluation

## Objective

Cumprir o requisito do projeto de avaliar o sistema reproduzido em um conjunto de dados novo, executando uma seleção de **três experimentos da Table 10** sobre um novo subconjunto ImageNet composto pelas classes:

```text
ambulance
scholar_bus
soccer_ball
```

Os três experimentos devem ser o mais próximo possível das respectivas linhas da Table 10 original, mudando apenas a população avaliada. Cada combinação deve ter uma execução pública separada:

```bash
python scripts/run_experiment.py --experiment imagenet_new_classes_fgsm_googlenet
python scripts/run_experiment.py --experiment imagenet_new_classes_deepfool_caffenet
python scripts/run_experiment.py --experiment imagenet_new_classes_cw_l2_inception_v3
```

Estes experimentos não substituem a reprodução oficial da Table 10. Eles formam uma extensão comparativa enxuta que reutiliza o mesmo schema, os mesmos modelos ImageNet compatíveis, a mesma lógica de detecção e a mesma forma de contagem, mas executa somente as três combinações definidas nesta spec.

---

## Context

A Table 10 do artigo avalia o DeepDetector com combinações de ataque, modelo e dataset. Para esta extensão, somente três combinações ImageNet devem ser repetidas no novo subconjunto:

| No. | Attack/Model                         | Dataset |
| --: | ------------------------------------ | ------- |
|   5 | `FGSM (ε=1/255)/GoogLeNet`           | ImageNet |
|   8 | `DeepFool/CaffeNet`                  | ImageNet |
|  14 | `CW L2 (κ=0.0)/Inception v3`         | ImageNet |

Todas as outras linhas da Table 10 ficam fora desta extensão. As linhas MNIST ficam fora porque o novo conjunto de dados é ImageNet; as demais linhas ImageNet ficam fora para manter o experimento restrito às três combinações escolhidas.

A regra central de detecção permanece:

```text
C(x_adv) != C(T(x_adv))
```

Se a predição da imagem adversarial muda após a transformação `T`, o exemplo é considerado detectado.

---

## Dataset Choice Justification

As classes escolhidas foram:

| Classe        | Papel no novo subconjunto | Justificativa |
| ------------- | ------------------------- | ------------- |
| `ambulance`   | substitui a primeira classe estrutural da Table 10 | veículo com estrutura visual bem definida, rodas, janelas, luzes e padrões de cor |
| `scholar_bus`  | substitui a segunda classe estrutural da Table 10 | veículo visualmente relacionado a `ambulance`, útil para avaliar classes semanticamente próximas |
| `soccer_ball` | substitui a classe visualmente distinta da Table 10 | objeto compacto, não veicular, com formato e textura diferentes |

A Table 10/Inception v3 usa uma população de três classes com quotas `40/40/20`. Esta extensão deve preservar essa forma experimental:

| Classe nova   | Quota clean-correct |
| ------------- | ------------------: |
| `ambulance`   | 40 |
| `scholar_bus`  | 40 |
| `soccer_ball` | 20 |

Cada experimento deve tentar avaliar 100 imagens limpas corretamente classificadas, preenchendo as quotas a partir de candidatos adicionais quando houver erros de classificação limpa.

---

## Business Rules

1. Devem existir exatamente três experimentos públicos: `imagenet_new_classes_fgsm_googlenet`, `imagenet_new_classes_deepfool_caffenet` e `imagenet_new_classes_cw_l2_inception_v3`.
2. Cada experimento público deve executar somente uma combinação ataque/modelo.
3. A população avaliada deve conter somente `ambulance`, `scholar_bus` e `soccer_ball`.
4. As quotas devem ser `40/40/20`, na ordem `ambulance`, `scholar_bus`, `soccer_ball`.
5. Cada experimento deve selecionar amostras clean-correct até preencher suas quotas, quando candidatos suficientes existirem.
6. Clean errors podem ser registrados no `manifest.json`, mas não devem reduzir a avaliação final abaixo de 100 imagens se houver candidatos de reposição disponíveis.
7. As linhas executadas devem corresponder somente às linhas selecionadas da Table 10: `5`, `8` e `14`.
8. O campo `no` deve preservar o número da linha original da Table 10.
9. O campo `dataset` no CSV/JSON deve ser `ImageNet-NewClasses`, para diferenciar esta extensão da reprodução oficial.
10. O schema de `metrics.csv` e `metrics.json` deve ser o schema oficial da Table 10.
11. O cálculo de TP, FN, FP, RTP, recall, precision e F1 deve reutilizar os helpers atuais da Table 10 sempre que possível.
12. A transformação `T` deve ser o filtro oficial usado pela Table 10 no projeto, por padrão `proposed_detection_filter`.
13. A mesma transformação `T` deve ser usada para imagens limpas e adversariais.
14. Cada linha deve usar o mesmo ataque, modelo, hiperparâmetros principais, domínio de entrada e pré-processamento da linha correspondente da Table 10.
15. Não deve haver médias simples por classe ou por experimento; métricas agregadas devem ser derivadas da soma dos contadores.

---

## Scope

Esta spec cobre:

* configuração e execução separada dos três experimentos públicos;
* execução das linhas selecionadas da Table 10 sobre `ambulance`, `scholar_bus` e `soccer_ball`;
* seleção reprodutível de amostras com quotas `40/40/20`;
* uso de GoogLeNet, CaffeNet e Inception v3 nos mesmos papéis da Table 10;
* uso de FGSM, DeepFool e CW L2 nos mesmos papéis da Table 10;
* geração de outputs separados por experimento;
* preservação do schema oficial da Table 10;
* registro de status e erros por linha em `manifest.json`.

---

## Out of Scope

Não implementar nesta spec:

* linhas MNIST da Table 10;
* linhas ImageNet não selecionadas da Table 10;
* M1;
* M2;
* `FGSM (ε=2/255)/GoogLeNet`;
* `DeepFool/GoogLeNet`;
* `CW L2` com `κ=0.5`, `κ=1.0`, `κ=2.0` ou `κ=4.0`;
* `CW L∞/Inception v3`;
* novos ataques;
* alteração da matemática dos ataques;
* alteração da matemática do filtro;
* download automático de datasets ou pesos;
* relatórios Markdown;
* diagnósticos públicos fora do `manifest.json`;
* comparação automática com os resultados oficiais do artigo;
* agregação final em um CSV único;
* experimento público composto que rode as três combinações de uma vez.

---

## Public Execution Interface

As únicas interfaces públicas permitidas são:

```bash
python scripts/run_experiment.py --experiment imagenet_new_classes_fgsm_googlenet
python scripts/run_experiment.py --experiment imagenet_new_classes_deepfool_caffenet
python scripts/run_experiment.py --experiment imagenet_new_classes_cw_l2_inception_v3
```

Não deve existir um experimento público agregado ou comandos por classe:

```bash
python scripts/run_experiment.py --experiment imagenet_new_classes_image_models
python scripts/run_experiment.py --experiment imagenet_new_classes
python scripts/run_experiment.py --experiment ambulance
python scripts/run_experiment.py --experiment scholar_bus
python scripts/run_experiment.py --experiment soccer_ball
```

---

## Dataset

### Expected Structure

```text
data/
└── imagenet/
    └── new_test/
        ├── ambulance/
        ├── scholar_bus/
        └── soccer_ball/
```

Cada diretório deve conter apenas imagens pertencentes à respectiva classe.

### Dataset Validation

Antes da execução, validar:

* se as três classes existem localmente;
* se cada classe possui pelo menos 40, 40 e 20 candidatos respectivamente, antes do filtro clean-correct;
* se os arquivos possuem extensão de imagem suportada;
* se a configuração declara labels compatíveis com cada família de modelo;
* se os modelos selecionados são compatíveis com ImageNet.

Se uma classe estiver ausente:

```text
Missing ImageNet class directory: data/imagenet/new_test/ambulance
```

Se uma classe não tiver candidatos suficientes:

```text
Insufficient ImageNet candidates for class soccer_ball: required at least 20, found 12.
```

Se não houver clean-correct suficiente para preencher uma quota:

```text
Insufficient clean-correct ImageNet samples for class scholar_bus and model googlenet: required 40, found 37.
```

Se houver par modelo/dataset incompatível:

```text
Invalid model-domain pair: m1 is not compatible with imagenet.
```

---

## Configuration

A configuração deve residir em:

```text
configs/experiments.yaml
```

Estrutura esperada:

```yaml
imagenet_new_classes_fgsm_googlenet:
  kind: table_10_group
  output_dir: results/experiments/imagenet_new_classes/fgsm_googlenet
  dataset:
    name: imagenet
    split: new_test
    images_dir: data/imagenet/new_test
    image_size: 224
    image_shape: [224, 224, 3]
    value_range: [0.0, 1.0]
    shuffle: false
    require_clean_correct: true
    class_order: [ambulance, scholar_bus, soccer_ball]
    class_quotas:
      ambulance: 40
      scholar_bus: 40
      soccer_ball: 20
    class_indices:
      ambulance: 407
      scholar_bus: 779
      soccer_ball: 805
  model:
    name: googlenet_caffe
  evaluation:
    seed: 20170830
  filter:
    name: proposed_detection_filter
    type: proposed_detection_filter
  model_group: googlenet
  dataset_label: ImageNet-NewClasses
  rows:
    - "no": 5
      attack_model: "FGSM (ε=1/255)/GoogLeNet"
      status: implemented
      attack:
        name: fgsm
        epsilon: 0.00392156862745098
        clip_min: 0.0
        clip_max: 255.0

imagenet_new_classes_deepfool_caffenet:
  kind: table_10_group
  output_dir: results/experiments/imagenet_new_classes/deepfool_caffenet
  dataset:
    name: imagenet
    split: new_test
    images_dir: data/imagenet/new_test
    image_size: 227
    image_shape: [227, 227, 3]
    value_range: [0.0, 1.0]
    shuffle: false
    require_clean_correct: true
    class_order: [ambulance, scholar_bus, soccer_ball]
    class_quotas:
      ambulance: 40
      scholar_bus: 40
      soccer_ball: 20
    class_indices:
      ambulance: 407
      scholar_bus: 779
      soccer_ball: 805
  model:
    name: caffenet
  evaluation:
    seed: 20170830
  filter:
    name: proposed_detection_filter
    type: proposed_detection_filter
  model_group: caffenet
  dataset_label: ImageNet-NewClasses
  rows:
    - "no": 8
      attack_model: "DeepFool/CaffeNet"
      status: implemented
      attack:
        name: deepfool

imagenet_new_classes_cw_l2_inception_v3:
  kind: table_10_group
  output_dir: results/experiments/imagenet_new_classes/cw_l2_inception_v3
  dataset:
    name: imagenet
    split: new_test
    images_dir: data/imagenet/new_test
    image_size: 299
    image_shape: [299, 299, 3]
    value_range: [-0.5, 0.5]
    shuffle: false
    require_clean_correct: true
    class_order: [ambulance, scholar_bus, soccer_ball]
    class_quotas:
      ambulance: 40
      scholar_bus: 40
      soccer_ball: 20
    class_indices:
      ambulance: <inception_v3_label_index>
      scholar_bus: <inception_v3_label_index>
      soccer_ball: <inception_v3_label_index>
  model:
    name: inception_v3
  evaluation:
    seed: 20170830
  filter:
    name: proposed_detection_filter
    type: proposed_detection_filter
  model_group: inception_v3
  dataset_label: ImageNet-NewClasses
  rows:
    - "no": 14
      attack_model: "CW L2 (κ=0.0)/Inception v3"
      status: implemented
      attack:
        name: cw_l2
        kappa: 0.0
        clip_min: -0.5
        clip_max: 0.5
```

Os índices Caffe/GoogLeNet acima usam a convenção ImageNet 1000 classes já usada pelo projeto. Os índices Inception v3 devem ser preenchidos a partir do label map compatível com o grafo TensorFlow usado localmente, porque esse wrapper usa saída de 1008 dimensões.

---

## Processing Flow

Para cada experimento:

1. Carregar candidatos das classes `ambulance`, `scholar_bus` e `soccer_ball`.
2. Aplicar a ordem determinística por classe.
3. Executar predição limpa `clean_pred = C(x)`.
4. Descartar candidatos em que `clean_pred != y_true`.
5. Continuar lendo candidatos até preencher as quotas `40/40/20`.
6. Gerar `x_adv` com o ataque configurado para a única row do experimento.
7. Calcular `adv_pred = C(x_adv)`.
8. Se `adv_pred == clean_pred`, contar `num_failures += 1` e excluir a amostra de TP/FN/RTP.
9. Aplicar `T(x_adv)` e calcular `filtered_adv_pred`.
10. Aplicar `T(x)` e calcular `filtered_clean_pred`.
11. Agregar as métricas da row com os helpers de Table 10.

Para GoogLeNet e CaffeNet, respeitar o domínio de ataque já usado pela reprodução Caffe. Para Inception v3, preservar o domínio `[-0.5, 0.5]` e `299x299`.

---

## Counting Rules

As métricas devem seguir a semântica da Table 10:

| Campo | Regra |
| ----- | ----- |
| `num_failures` | número de ataques que não alteram a predição limpa, isto é, `adv_pred == clean_pred` |
| `tp` | adversarial bem-sucedido detectado por `filtered_adv_pred != adv_pred` |
| `fn` | adversarial bem-sucedido não detectado por `filtered_adv_pred == adv_pred` |
| `fp` | imagem limpa cuja predição muda após o filtro, isto é, `filtered_clean_pred != clean_pred` |
| `rtp` | true positives em que `filtered_adv_pred == y_true` |
| `rtp_percent` | `rtp / tp * 100`, zero-safe |
| `recall` | `tp / (tp + fn) * 100`, zero-safe |
| `precision` | `tp / (tp + fp) * 100`, zero-safe |
| `f1` | `2 * precision * recall / (precision + recall)`, zero-safe |

Clean errors não entram em `num_failures`, `tp`, `fn`, `fp` ou `rtp`.

---

## Output Artifacts

A execução deve gerar somente:

```text
results/
└── experiments/
    └── imagenet_new_classes/
        ├── fgsm_googlenet/
        │   ├── metrics.csv
        │   ├── metrics.json
        │   └── manifest.json
        ├── deepfool_caffenet/
        │   ├── metrics.csv
        │   ├── metrics.json
        │   └── manifest.json
        └── cw_l2_inception_v3/
            ├── metrics.csv
            ├── metrics.json
            └── manifest.json
```

Não deve existir `metrics.csv` agregado na raiz de `results/experiments/imagenet_new_classes/`.

### CSV Format

Cada `metrics.csv` deve seguir exatamente:

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
```

Exemplo ilustrativo para `fgsm_googlenet/metrics.csv`:

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
5,FGSM (ε=1/255)/GoogLeNet,ImageNet-NewClasses,4,82,14,3,79,96.34,85.42,96.47,90.62
```

Os números são apenas ilustrativos.

### JSON Format

Cada `metrics.json` deve usar a estrutura da Table 10:

```json
{
  "table": 10,
  "dataset_group": "imagenet_new_classes",
  "model_group": "googlenet",
  "rows": [
    {
      "no": 5,
      "attack_model": "FGSM (ε=1/255)/GoogLeNet",
      "dataset": "ImageNet-NewClasses",
      "num_failures": 4,
      "tp": 82,
      "fn": 14,
      "fp": 3,
      "rtp": 79,
      "rtp_percent": 96.34,
      "recall": 85.42,
      "precision": 96.47,
      "f1": 90.62
    }
  ]
}
```

### Manifest

Cada `manifest.json` deve registrar, no mínimo:

* `dataset_group: imagenet_new_classes`;
* `model_group`;
* identificador do experimento público;
* classes avaliadas;
* quotas requeridas;
* quantidade de candidatos lidos por classe;
* quantidade de clean errors por classe;
* quantidade final clean-correct por classe;
* status de cada row;
* `blocked_reason` ou `error` quando uma row não puder ser avaliada.

---

## Architecture Constraints

Toda execução deve partir de:

```text
scripts/run_experiment.py
```

A lógica deve residir em:

```text
src/deepdetector/
```

Os experimentos devem reutilizar o runner e os helpers existentes da Table 10 sempre que possível. Nenhuma lógica experimental deve ser implementada diretamente em scripts.

Módulos esperados ou reutilizáveis:

```text
src/deepdetector/data/imagenet.py
src/deepdetector/data/imagenet_subset.py
src/deepdetector/evaluation/tables/table_10.py
src/deepdetector/filters/
src/deepdetector/attacks/
src/deepdetector/models/imagenet_wrappers.py
```

---

## Non-functional Requirements

* A seleção de dados deve ser determinística.
* A seed padrão deve ser `20170830`, como na reprodução da Table 10.
* A execução deve preservar o padrão de artefatos pequenos do projeto.
* O código deve evitar duplicação entre este experimento e a Table 10 oficial.
* A implementação deve manter os dados gerados, pesos e outputs grandes fora do git.
* Testes devem cobrir o schema, a seleção `40/40/20`, o descarte clean-correct e a ausência de artefatos proibidos.

---

## Forbidden Artifacts

Não criar:

```text
results/experiments/imagenet_new_classes_image_models/metrics.csv
results/experiments/imagenet_new_classes_image_models/metrics.json
results/experiments/imagenet_new_classes_image_models/report.md
results/experiments/imagenet_new_classes_image_models/diagnostic.json
results/experiments/imagenet_new_classes_image_models/debug/
results/experiments/imagenet_new_classes_image_models/ambulance/
results/experiments/imagenet_new_classes_image_models/scholar_bus/
results/experiments/imagenet_new_classes_image_models/soccer_ball/
results/experiments/imagenet_new_classes/metrics.csv
results/experiments/imagenet_new_classes/metrics.json
results/experiments/imagenet_new_classes/report.md
results/experiments/imagenet_new_classes/diagnostic.json
results/experiments/imagenet_new_classes/debug/
results/experiments/imagenet_new_classes/ambulance/
results/experiments/imagenet_new_classes/scholar_bus/
results/experiments/imagenet_new_classes/soccer_ball/
```

---

## Acceptance Criteria

- [ ] O experimento FGSM/GoogLeNet é executado por `python scripts/run_experiment.py --experiment imagenet_new_classes_fgsm_googlenet`.
- [ ] O experimento DeepFool/CaffeNet é executado por `python scripts/run_experiment.py --experiment imagenet_new_classes_deepfool_caffenet`.
- [ ] O experimento CW L2/Inception v3 é executado por `python scripts/run_experiment.py --experiment imagenet_new_classes_cw_l2_inception_v3`.
- [ ] Nenhum experimento público agregado executa as três combinações de uma vez.
- [ ] Cada experimento usa exclusivamente as classes `ambulance`, `scholar_bus` e `soccer_ball`.
- [ ] Cada experimento tenta preencher as quotas clean-correct `40/40/20`.
- [ ] Clean errors são repostos por candidatos adicionais quando disponíveis.
- [ ] `imagenet_new_classes_fgsm_googlenet` executa somente a linha 5 da Table 10: `FGSM (ε=1/255)/GoogLeNet`.
- [ ] `imagenet_new_classes_deepfool_caffenet` executa somente a linha 8 da Table 10: `DeepFool/CaffeNet`.
- [ ] `imagenet_new_classes_cw_l2_inception_v3` executa somente a linha 14 da Table 10: `CW L2 (κ=0.0)/Inception v3`.
- [ ] Nenhuma linha MNIST da Table 10 é executada.
- [ ] Nenhuma linha ImageNet fora de `5`, `8` e `14` é executada.
- [ ] O campo `no` preserva os números originais da Table 10.
- [ ] O campo `dataset` é `ImageNet-NewClasses`.
- [ ] Cada experimento escreve `metrics.csv`, `metrics.json` e `manifest.json`.
- [ ] Cada `metrics.csv` possui exatamente o schema `no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1`.
- [ ] Cada `metrics.json` possui os mesmos valores semânticos do CSV.
- [ ] `num_failures`, `tp`, `fn`, `fp`, `rtp`, `rtp_percent`, `recall`, `precision` e `f1` seguem as regras da Table 10.
- [ ] A implementação valida compatibilidade entre modelo e dataset.
- [ ] A implementação usa o filtro oficial `proposed_detection_filter` salvo configuração equivalente já aceita pela Table 10.
- [ ] A implementação não cria CSV/JSON agregado na raiz de `results/experiments/imagenet_new_classes/`.
- [ ] Nenhum relatório Markdown, diagnóstico público ou diretório por classe é produzido.

---

## Error Cases

* Classe ausente deve falhar com erro claro.
* Classe sem candidatos suficientes deve falhar com erro claro.
* Quota clean-correct impossível de preencher deve falhar com erro claro e registrar contexto no manifest quando o manifest já puder ser escrito.
* Modelo incompatível com ImageNet deve falhar antes da execução das rows.
* Row sem ataque registrado deve falhar com erro claro.
* Modelo sem método necessário para o ataque, como gradiente para DeepFool, deve falhar com erro claro.
* Configuração sem label index compatível com o modelo deve falhar antes de selecionar amostras.
