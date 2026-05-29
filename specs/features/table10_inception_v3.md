# Spec — Implementar Table 10 / ImageNet / Inception v3

## 1. Objetivo

Implementar o grupo `inception_v3` da **Table 10**, referente aos experimentos em **ImageNet** com o modelo **Inception v3**.

Este grupo contém as linhas:

| No. | Attack/Model | Dataset |
|---:|---|---|
| 14 | `CW L2 (κ=0.0)/Inception v3` | ImageNet |
| 15 | `CW L2 (κ=0.5)/Inception v3` | ImageNet |
| 16 | `CW L2 (κ=1.0)/Inception v3` | ImageNet |
| 17 | `CW L2 (κ=2.0)/Inception v3` | ImageNet |
| 18 | `CW L2 (κ=4.0)/Inception v3` | ImageNet |
| 20 | `CW L∞/Inception v3` | ImageNet |

A implementação deve integrar o modelo ao runner Table 10 existente, gerar resultados oficiais e preservar o schema da Table 10 sem criar scripts paralelos específicos.

---

## 2. Contexto

O repositório já define `configs/experiments.yaml` com `table_10_inception_v3` e os seis ataques CW, mas todas as linhas estão marcadas como `blocked` e a implementação de `inception_v3` não existe.

O código de referência do repositório original `OwenSec/DeepDetector/Test/CW` mostra dois pontos importantes:

* `modified_setup_inception.py` implementa um wrapper `InceptionModel` para Inception v3 com:
  * `image_size = 299`
  * `num_labels = 1008`
  * `predict`/`my_predict` usando `tf.import_graph_def` e `softmax/logits:0`
  * normalização de entrada em `[-0.5, 0.5]` e escalonamento para `(0.5 + img) * 255` antes da inferência.
* `Test_CWLi_ImageNet.py` usa a classe `ImageNet` desse wrapper para carregar imagens das classes zebra/panda/cab e invocar ataques `CarliniLi`.

O modelo Inception v3 nesta spec deve atender ao mesmo pré-processamento e saída de logits, mantendo a compatibilidade com o fluxo CW da Table 10.

---

## 3. Estado atual esperado

A configuração `table_10_inception_v3` deve permanecer como `kind: table_10_group` e ser atualizada para refletir o suporte real ao Inception v3.

A configuração correta deve incluir, no mínimo:

* `dataset.image_size: 299`
* `dataset.image_shape: [299, 299, 3]`
* `dataset.value_range: [-0.5, 0.5]` ou equivalente de pré-processamento
* `model.name: inception_v3`
* `rows` com os seis ataques CW listados

A execução deve produzir em `results/experiments/table_10/inception_v3/`:

* `metrics.csv`
* `metrics.json`
* `manifest.json`

---

## 4. Escopo

Esta spec cobre:

* registro e execução do grupo `table_10_inception_v3` via `scripts/run_experiment.py`;
* implementação da integração do modelo `inception_v3` com o runner Table 10;
* geração de `metrics.csv` e `metrics.json` no diretório oficial;
* preservação do schema oficial da Table 10;
* suporte a ataques `cw_l2` com κ = 0.0, 0.5, 1.0, 2.0, 4.0 e `cw_linf`;
* uso de entrada 299x299 com pré-processamento de Inception;
* manutenção do manifest para indicar linhas bloqueadas ou ainda não executadas.

---

## 5. Fora do escopo

Não implementar nesta spec:

* outro modelo que não seja Inception v3;
* mudanças na estrutura geral da Table 10 fora do grupo `inception_v3`;
* criação de scripts paralelos específicos apenas para Inception;
* relatórios `.md` adicionais;
* suporte a classes extras além da amostra zebra/panda/cab durante o primeiro ciclo de integração;
* alteração da organização de saída para além de `metrics.csv`, `metrics.json` e `manifest.json`.

---

## 6. Requisitos funcionais

* Deve existir um wrapper `inception_v3` que aceite entradas de tamanho `299x299x3` e produza logits ou probabilidades compatíveis com o ataque CW.
* O wrapper deve aplicar o mesmo pré-processamento de entrada do original: normalização centralizada em `[-0.5, 0.5]` com escalonamento para `255` antes de alimentar o grafo.
* O ataque `cw_l2` deve aceitar parâmetros `kappa` fixos conforme as linhas da Table 10.
* O ataque `cw_linf` deve ser suportado como outra linha do grupo.
* O `table_10_group` deve ser capaz de produzir os seis resultados conforme o schema oficial.
* O `manifest.json` deve conter `blocked_reason` para linhas que ainda não puderam ser avaliadas.
* O `metrics.csv` deve manter todos os campos da Table 10 e manter os números de artigo originais nas linhas.

---

## 7. Requisitos não funcionais

* Reutilizar componentes de métricas, runners e saída existentes sempre que possível.
* Evitar lógica duplicada entre `googlenet` e `inception_v3`.
* Documentar no código ou em comentários a diferença entre `299x299` Inception e o resto do pipeline ImageNet.
* Não introduzir dependências externas desnecessárias além das já presentes no repositório.

---

## 8. Critérios de aceite

- [ ] O `configs/experiments.yaml` contém `table_10_inception_v3` com `kind: table_10_group`.
- [ ] O experimento `table_10_inception_v3` é executável por `python scripts/run_experiment.py --experiment table_10_inception_v3`.
- [ ] O diretório `results/experiments/table_10/inception_v3/` é criado com `metrics.csv` e `metrics.json`.
- [ ] O `manifest.json` é produzido e contém `blocked_reason` para linhas não executadas.
- [ ] O schema da Table 10 mantém os campos `no`, `attack_model`, `dataset`, `num_failures`, `tp`, `fn`, `fp`, `rtp`, `rtp_percent`, `recall`, `precision`, `f1`.
- [ ] O modelo Inception v3 usa entrada `299x299` e o pré-processamento descrito no repositório original.
- [ ] As linhas 14, 15, 16, 17, 18 e 20 preservam seus números originais do artigo.
- [ ] O grupo `inception_v3` é integrado sem necessidade de um script específico fora do runner da Table 10.

---

## 9. Observações de implementação

* A configuração atual de `image_size: 224` em `table_10_inception_v3` está incorreta para Inception v3 e deve ser ajustada para `299`.
* O wrapper de modelo deve compatibilizar o fluxo de CW com o grafo Inception e com os ataques existentes do projeto.
* O repositório original usa `modified_setup_inception.py` para importar `softmax/logits:0`; essa mesma abstração deve ser reproduzida de forma limpa aqui.
* O original `Test_CWLi_ImageNet.py` carrega as classes zebra/panda/cab e usa alvos one-hot com 1008 dimensões; o suporte a esse tipo de rótulo deve ser preservado na implementação.
