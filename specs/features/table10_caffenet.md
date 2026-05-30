# Spec — Implementar Table 10 / ImageNet / CaffeNet

## 1. Objetivo

Implementar o grupo `caffenet` da **Table 10**, referente ao experimento em **ImageNet** com o modelo **CaffeNet**.

Esta linha é:

| No. | Attack/Model | Dataset |
|---:|---|---|
| 8 | `DeepFool/CaffeNet` | ImageNet |

O comportamento deve ser integrado ao runner `table_10_group` existente, sem criar um script paralelo específico.

---

## 2. Contexto

A configuração `table_10_caffenet` já existe em `configs/experiments.yaml`, mas está bloqueada porque a implementação do modelo CaffeNet ainda não está disponível.

O repositório original `OwenSec/DeepDetector/Test/DeepFool/CaffeNet` fornece os principais elementos de referência:

* `deploy_original.prototxt` — modelo CaffeNet com softmax para classificação e detecção.
* `deploy_removeSoftmax.prototxt` — modelo CaffeNet sem softmax usado para a geração de ataques DeepFool.
* `README.MD` — explica que o DeepFool é criado com o modelo sem softmax, mas a classificação/detecção deve usar o modelo original.

A implementação aqui deve reproduzir essa separação de papéis: `deploy_original.prototxt` para inferência normal e `deploy_removeSoftmax.prototxt` como modelo de ataque.

---

## 3. Estado atual esperado

A configuração `table_10_caffenet` deve continuar como `kind: table_10_group` e deve ser atualizada para refletir o suporte correto a CaffeNet.

O grupo deve gerar saída em `results/experiments/table_10/caffenet/` com:

* `metrics.csv`
* `metrics.json`
* `manifest.json`

A linha 8 deve preservar o número original do artigo.

---

## 4. Escopo

Esta spec cobre:

* integração do grupo `table_10_caffenet` com o runner `table_10_group`;
* implementação estrutural do modelo `caffenet` no pipeline de execução de Table 10;
* suporte a `deploy_original.prototxt` e `deploy_removeSoftmax.prototxt` no wrapper de modelo;
* geração de `metrics.csv`, `metrics.json` e `manifest.json` para o grupo CaffeNet;
* manutenção do schema oficial da Table 10;
* preservação do fluxo de execução comum sem scripts específicos.

---

## 5. Fora do escopo

Não implementar nesta tarefa:

* Inception v3;
* GoogLeNet;
* CW ou FGSM além do DeepFool/CaffeNet;
* novo runner em `scripts/`;
* relatórios adicionais em `.md`;
* conversão de pesos ou download de modelos externos;
* agregação geral de Table 10 fora do grupo CaffeNet.

---

## 6. Requisitos funcionais

* Deve existir um wrapper ou adaptador `caffenet` compatível com o runner Table 10.
* O wrapper deve aceitar parâmetros separados para:
  * `deploy_proto` — modelo original com softmax;
  * `attack_deploy_proto` — modelo sem softmax para gerar gradientes DeepFool.
* O attack `deepfool` deve ser integrado ao fluxo de experimentos já existentes.
* O `table_10_caffenet` deve suportar a linha 8 com `attack_model: "DeepFool/CaffeNet"`.
* O `manifest.json` deve conter `blocked_reason` quando a linha não estiver implementada.
* O `metrics.csv` deve preservar o schema da Table 10 e deixar métricas vazias para linhas não executadas.

---

## 7. Requisitos não funcionais

* Reutilizar o runner `table_10_group` e os componentes de métricas já presentes.
* Evitar lógica duplicada entre `caffenet` e outros grupos da Table 10.
* Documentar no código a distinção entre modelo de inferência e modelo de ataque.
* Manter compatibilidade com a configuração atual de ImageNet usada por Table 10.

---

## 8. Critérios de aceite

- [ ] O `configs/experiments.yaml` contém `table_10_caffenet` com `kind: table_10_group`.
- [ ] O experimento `table_10_caffenet` pode ser invocado por `python scripts/run_experiment.py --experiment table_10_caffenet`.
- [ ] O grupo `caffenet` escreve `results/experiments/table_10/caffenet/metrics.csv` e `metrics.json`.
- [ ] O grupo `caffenet` escreve `results/experiments/table_10/caffenet/manifest.json`.
- [ ] A linha 8 mantém `no: 8` e `attack_model: "DeepFool/CaffeNet"`.
- [ ] O wrapper reporta `attack_deploy_proto`/`deploy_proto` separados de forma compatível com o original.
- [ ] As métricas vazias são mantidas para linhas bloqueadas.
- [ ] A implementação não cria scripts ou diretórios extras fora do fluxo Table 10.

---

## 9. Observações de implementação

* O repositório original separa claramente o modelo de ataque sem softmax e o modelo de classificação com softmax. Essa separação deve ser preservada na implementação.
* A linha 8 deve permanecer bloqueada até que o DeepFool/CaffeNet seja realmente funcional.
* Se a implementação for realizada, o `blocked_reason` deve ser removido de `configs/experiments.yaml` e do `manifest.json`.
* O nome do modelo local deve ser `caffenet` para seguir o padrão de `model_group` usado pelo runner.
