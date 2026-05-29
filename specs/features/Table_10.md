# Spec — Estrutura da Table 10 por Modelo

## 1. Objetivo

Estruturar a **Table 10** do artigo base separando os experimentos por **modelo**.

A organização deve seguir os campos oficiais da Table 10:

```text
No. | Attack/Model | Dataset | #F | TP | FN | FP | RTP | RTP% | Recall | Precision | F1
```

Não deve existir diretório ou saída agregada. Cada grupo de modelo é independente.

---

## 2. Campos oficiais da Table 10

Todas as saídas da Table 10 devem seguir este schema:

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
```

## Mapeamento dos campos

| Campo do artigo | Nome no arquivo | Descrição |
|---|---|---|
| No. | `no` | Número original da linha na Table 10 |
| Attack/Model | `attack_model` | Texto do ataque e modelo |
| Dataset | `dataset` | Dataset usado |
| #F | `num_failures` | Número de falhas do ataque |
| TP | `tp` | True positives |
| FN | `fn` | False negatives |
| FP | `fp` | False positives |
| RTP | `rtp` | Recovered true positives |
| RTP% | `rtp_percent` | Percentual de RTP |
| Recall | `recall` | Recall em percentual |
| Precision | `precision` | Precision em percentual |
| F1 | `f1` | F1-score em percentual |

---

## 3. Organização por modelo

A Table 10 deve ser organizada em cinco grupos:

```text
table_10/
├── m1/
├── googlenet/
├── caffenet/
├── m2/
└── inception_v3/
```

Não criar:

```text
table_10/aggregate/
```

Cada grupo deve conter apenas as linhas associadas ao respectivo modelo.

---

## 4. Grupos da Table 10

## 4.1. Grupo M1

O grupo `m1` representa os experimentos FGSM em MNIST com o modelo M1.

| No. | Attack/Model | Dataset |
|---:|---|---|
| 1 | FGSM (ε=0.1)/M1 | MNIST |
| 2 | FGSM (ε=0.2)/M1 | MNIST |
| 3 | FGSM (ε=0.3)/M1 | MNIST |
| 4 | FGSM (ε=0.4)/M1 | MNIST |

Diretório:

```text
results/experiments/table_10/m1/
```

Arquivos:

```text
results/experiments/table_10/m1/
├── metrics.csv
├── metrics.json
└── manifest.json
```

---

## 4.2. Grupo GoogLeNet

O grupo `googlenet` representa os experimentos em ImageNet com o modelo GoogLeNet.

| No. | Attack/Model | Dataset |
|---:|---|---|
| 5 | FGSM (ε=1/255)/GoogLeNet | ImageNet |
| 6 | FGSM (ε=2/255)/GoogLeNet | ImageNet |
| 7 | DeepFool/GoogLeNet | ImageNet |

Diretório:

```text
results/experiments/table_10/googlenet/
```

Arquivos:

```text
results/experiments/table_10/googlenet/
├── metrics.csv
├── metrics.json
└── manifest.json
```

---

## 4.3. Grupo CaffeNet

O grupo `caffenet` representa os experimentos em ImageNet com o modelo CaffeNet.

| No. | Attack/Model | Dataset |
|---:|---|---|
| 8 | DeepFool/CaffeNet | ImageNet |

Diretório:

```text
results/experiments/table_10/caffenet/
```

Arquivos:

```text
results/experiments/table_10/caffenet/
├── metrics.csv
├── metrics.json
└── manifest.json
```

Status inicial recomendado:

```text
blocked
```

Motivo:

```text
CaffeNet ainda não está implementado.
```

---

## 4.4. Grupo M2

O grupo `m2` representa os experimentos CW em MNIST com o modelo M2.

| No. | Attack/Model | Dataset |
|---:|---|---|
| 9 | CW L2 (κ=0.0)/M2 | MNIST |
| 10 | CW L2 (κ=0.5)/M2 | MNIST |
| 11 | CW L2 (κ=1.0)/M2 | MNIST |
| 12 | CW L2 (κ=2.0)/M2 | MNIST |
| 13 | CW L2 (κ=4.0)/M2 | MNIST |
| 19 | CW L∞/M2 | MNIST |

Diretório:

```text
results/experiments/table_10/m2/
```

Arquivos:

```text
results/experiments/table_10/m2/
├── metrics.csv
├── metrics.json
└── manifest.json
```

---

## 4.5. Grupo Inception v3

O grupo `inception_v3` representa os experimentos CW em ImageNet com Inception v3.

| No. | Attack/Model | Dataset |
|---:|---|---|
| 14 | CW L2 (κ=0.0)/Inception v3 | ImageNet |
| 15 | CW L2 (κ=0.5)/Inception v3 | ImageNet |
| 16 | CW L2 (κ=1.0)/Inception v3 | ImageNet |
| 17 | CW L2 (κ=2.0)/Inception v3 | ImageNet |
| 18 | CW L2 (κ=4.0)/Inception v3 | ImageNet |
| 20 | CW L∞/Inception v3 | ImageNet |

Diretório:

```text
results/experiments/table_10/inception_v3/
```

Arquivos:

```text
results/experiments/table_10/inception_v3/
├── metrics.csv
├── metrics.json
└── manifest.json
```

Status inicial recomendado:

```text
blocked
```

Motivo:

```text
Inception v3 ainda não está implementado.
```

---

## 5. Estrutura final de diretórios

A estrutura oficial da Table 10 deve ser:

```text
results/
└── experiments/
    └── table_10/
        ├── m1/
        │   ├── metrics.csv
        │   ├── metrics.json
        │   └── manifest.json
        ├── googlenet/
        │   ├── metrics.csv
        │   ├── metrics.json
        │   └── manifest.json
        ├── caffenet/
        │   ├── metrics.csv
        │   ├── metrics.json
        │   └── manifest.json
        ├── m2/
        │   ├── metrics.csv
        │   ├── metrics.json
        │   └── manifest.json
        └── inception_v3/
            ├── metrics.csv
            ├── metrics.json
            └── manifest.json
```

---

## 6. Configuração sugerida da Table 10

A configuração da Table 10 deve ser organizada por modelo.

```yaml
table_10:
  title: "Detection results of the proposed method"
  schema:
    - no
    - attack_model
    - dataset
    - num_failures
    - tp
    - fn
    - fp
    - rtp
    - rtp_percent
    - recall
    - precision
    - f1

  groups:
    m1:
      model: m1
      dataset: mnist
      output_dir: results/experiments/table_10/m1
      rows:
        - no: 1
          attack_model: "FGSM (ε=0.1)/M1"
          attack:
            name: fgsm
            epsilon: 0.1
          status: implemented

        - no: 2
          attack_model: "FGSM (ε=0.2)/M1"
          attack:
            name: fgsm
            epsilon: 0.2
          status: implemented

        - no: 3
          attack_model: "FGSM (ε=0.3)/M1"
          attack:
            name: fgsm
            epsilon: 0.3
          status: implemented

        - no: 4
          attack_model: "FGSM (ε=0.4)/M1"
          attack:
            name: fgsm
            epsilon: 0.4
          status: implemented

    googlenet:
      model: googlenet
      dataset: imagenet
      output_dir: results/experiments/table_10/googlenet
      rows:
        - no: 5
          attack_model: "FGSM (ε=1/255)/GoogLeNet"
          attack:
            name: fgsm
            epsilon: 0.00392156862745098
          status: planned

        - no: 6
          attack_model: "FGSM (ε=2/255)/GoogLeNet"
          attack:
            name: fgsm
            epsilon: 0.00784313725490196
          status: planned

        - no: 7
          attack_model: "DeepFool/GoogLeNet"
          attack:
            name: deepfool
          status: planned

    caffenet:
      model: caffenet
      dataset: imagenet
      output_dir: results/experiments/table_10/caffenet
      rows:
        - no: 8
          attack_model: "DeepFool/CaffeNet"
          attack:
            name: deepfool
          status: blocked
          blocked_reason: "CaffeNet ainda não está implementado."

    m2:
      model: m2
      dataset: mnist
      output_dir: results/experiments/table_10/m2
      rows:
        - no: 9
          attack_model: "CW L2 (κ=0.0)/M2"
          attack:
            name: cw_l2
            kappa: 0.0
          status: implemented

        - no: 10
          attack_model: "CW L2 (κ=0.5)/M2"
          attack:
            name: cw_l2
            kappa: 0.5
          status: planned

        - no: 11
          attack_model: "CW L2 (κ=1.0)/M2"
          attack:
            name: cw_l2
            kappa: 1.0
          status: planned

        - no: 12
          attack_model: "CW L2 (κ=2.0)/M2"
          attack:
            name: cw_l2
            kappa: 2.0
          status: planned

        - no: 13
          attack_model: "CW L2 (κ=4.0)/M2"
          attack:
            name: cw_l2
            kappa: 4.0
          status: planned

        - no: 19
          attack_model: "CW L∞/M2"
          attack:
            name: cw_linf
          status: planned

    inception_v3:
      model: inception_v3
      dataset: imagenet
      output_dir: results/experiments/table_10/inception_v3
      rows:
        - no: 14
          attack_model: "CW L2 (κ=0.0)/Inception v3"
          attack:
            name: cw_l2
            kappa: 0.0
          status: blocked
          blocked_reason: "Inception v3 ainda não está implementado."

        - no: 15
          attack_model: "CW L2 (κ=0.5)/Inception v3"
          attack:
            name: cw_l2
            kappa: 0.5
          status: blocked
          blocked_reason: "Inception v3 ainda não está implementado."

        - no: 16
          attack_model: "CW L2 (κ=1.0)/Inception v3"
          attack:
            name: cw_l2
            kappa: 1.0
          status: blocked
          blocked_reason: "Inception v3 ainda não está implementado."

        - no: 17
          attack_model: "CW L2 (κ=2.0)/Inception v3"
          attack:
            name: cw_l2
            kappa: 2.0
          status: blocked
          blocked_reason: "Inception v3 ainda não está implementado."

        - no: 18
          attack_model: "CW L2 (κ=4.0)/Inception v3"
          attack:
            name: cw_l2
            kappa: 4.0
          status: blocked
          blocked_reason: "Inception v3 ainda não está implementado."

        - no: 20
          attack_model: "CW L∞/Inception v3"
          attack:
            name: cw_linf
          status: blocked
          blocked_reason: "Inception v3 ainda não está implementado."
```

---

## 7. Saídas esperadas por modelo

## 7.1. `m1/metrics.csv`

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
1,FGSM (ε=0.1)/M1,MNIST,,,,,,,,,
2,FGSM (ε=0.2)/M1,MNIST,,,,,,,,,
3,FGSM (ε=0.3)/M1,MNIST,,,,,,,,,
4,FGSM (ε=0.4)/M1,MNIST,,,,,,,,,
```

## 7.2. `googlenet/metrics.csv`

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
5,FGSM (ε=1/255)/GoogLeNet,ImageNet,,,,,,,,,
6,FGSM (ε=2/255)/GoogLeNet,ImageNet,,,,,,,,,
7,DeepFool/GoogLeNet,ImageNet,,,,,,,,,
```

## 7.3. `caffenet/metrics.csv`

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
8,DeepFool/CaffeNet,ImageNet,,,,,,,,,
```

## 7.4. `m2/metrics.csv`

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
9,CW L2 (κ=0.0)/M2,MNIST,,,,,,,,,
10,CW L2 (κ=0.5)/M2,MNIST,,,,,,,,,
11,CW L2 (κ=1.0)/M2,MNIST,,,,,,,,,
12,CW L2 (κ=2.0)/M2,MNIST,,,,,,,,,
13,CW L2 (κ=4.0)/M2,MNIST,,,,,,,,,
19,CW L∞/M2,MNIST,,,,,,,,,
```

## 7.5. `inception_v3/metrics.csv`

```csv
no,attack_model,dataset,num_failures,tp,fn,fp,rtp,rtp_percent,recall,precision,f1
14,CW L2 (κ=0.0)/Inception v3,ImageNet,,,,,,,,,
15,CW L2 (κ=0.5)/Inception v3,ImageNet,,,,,,,,,
16,CW L2 (κ=1.0)/Inception v3,ImageNet,,,,,,,,,
17,CW L2 (κ=2.0)/Inception v3,ImageNet,,,,,,,,,
18,CW L2 (κ=4.0)/Inception v3,ImageNet,,,,,,,,,
20,CW L∞/Inception v3,ImageNet,,,,,,,,,
```

---

## 8. Manifest por modelo

Cada modelo deve ter seu próprio `manifest.json`.

## Exemplo para `googlenet`

```json
{
  "table": 10,
  "model_group": "googlenet",
  "dataset": "ImageNet",
  "rows": [5, 6, 7],
  "schema": [
    "no",
    "attack_model",
    "dataset",
    "num_failures",
    "tp",
    "fn",
    "fp",
    "rtp",
    "rtp_percent",
    "recall",
    "precision",
    "f1"
  ],
  "outputs": {
    "metrics_csv": "results/experiments/table_10/googlenet/metrics.csv",
    "metrics_json": "results/experiments/table_10/googlenet/metrics.json"
  }
}
```

## Exemplo para `inception_v3`

```json
{
  "table": 10,
  "model_group": "inception_v3",
  "dataset": "ImageNet",
  "rows": [14, 15, 16, 17, 18, 20],
  "status": "blocked",
  "blocked_reason": "Inception v3 ainda não está implementado.",
  "schema": [
    "no",
    "attack_model",
    "dataset",
    "num_failures",
    "tp",
    "fn",
    "fp",
    "rtp",
    "rtp_percent",
    "recall",
    "precision",
    "f1"
  ],
  "outputs": {
    "metrics_csv": "results/experiments/table_10/inception_v3/metrics.csv",
    "metrics_json": "results/experiments/table_10/inception_v3/metrics.json"
  }
}
```

---

## 9. Regras de preenchimento

## 9.1. Linhas não executadas

Linhas ainda não executadas devem aparecer com métricas vazias no CSV e `null` no JSON.

CSV:

```csv
5,FGSM (ε=1/255)/GoogLeNet,ImageNet,,,,,,,,,
```

JSON:

```json
{
  "no": 5,
  "attack_model": "FGSM (ε=1/255)/GoogLeNet",
  "dataset": "ImageNet",
  "num_failures": null,
  "tp": null,
  "fn": null,
  "fp": null,
  "rtp": null,
  "rtp_percent": null,
  "recall": null,
  "precision": null,
  "f1": null
}
```

## 9.2. Linhas bloqueadas

Linhas bloqueadas devem aparecer no grupo do modelo com métricas vazias.

O motivo de bloqueio deve ficar apenas no `manifest.json`, não no `metrics.csv`.

## 9.3. Linhas executadas

Linhas executadas devem preencher todos os campos métricos.

Exemplo:

```csv
5,FGSM (ε=1/255)/GoogLeNet,ImageNet,270,841,98,88,718,85.37,89.56,90.53,90.04
```

---

## 10. Comandos por modelo

A Table 10 deve permitir comandos por grupo de modelo:

```bash
python scripts/run_experiment.py --experiment table_10_m1
python scripts/run_experiment.py --experiment table_10_googlenet
python scripts/run_experiment.py --experiment table_10_caffenet
python scripts/run_experiment.py --experiment table_10_m2
python scripts/run_experiment.py --experiment table_10_inception_v3
```

---

## 11. Critérios de aceite

- A Table 10 está separada por modelo.
- Existe grupo `m1`.
- Existe grupo `googlenet`.
- Existe grupo `caffenet`.
- Existe grupo `m2`.
- Existe grupo `inception_v3`.
- Cada grupo possui `metrics.csv`.
- Cada grupo possui `metrics.json`.
- Cada grupo possui `manifest.json`.
- Cada linha preserva o `No.` original do artigo.
- O schema segue os campos da Table 10.
- Linhas não executadas aparecem com métricas vazias.
- Linhas bloqueadas aparecem com métricas vazias.
- Motivos de bloqueio ficam no `manifest.json`.
- Não há referência a resultados legacy.
- A execução é feita por modelo.

---

## 12. Definition of Done

A estrutura estará pronta quando forem possíveis os seguintes arquivos:

```text
results/experiments/table_10/m1/metrics.csv
results/experiments/table_10/m1/metrics.json
results/experiments/table_10/m1/manifest.json

results/experiments/table_10/googlenet/metrics.csv
results/experiments/table_10/googlenet/metrics.json
results/experiments/table_10/googlenet/manifest.json

results/experiments/table_10/caffenet/metrics.csv
results/experiments/table_10/caffenet/metrics.json
results/experiments/table_10/caffenet/manifest.json

results/experiments/table_10/m2/metrics.csv
results/experiments/table_10/m2/metrics.json
results/experiments/table_10/m2/manifest.json

results/experiments/table_10/inception_v3/metrics.csv
results/experiments/table_10/inception_v3/metrics.json
results/experiments/table_10/inception_v3/manifest.json
```