# Detecção de Exemplos Adversários com Redução Adaptativa de Ruído
> **Referência:** *B. Liang, H. Li, M. Su, X. Li, W. Shi and X. Wang, "Detecting Adversarial Image Examples in Deep Neural Networks with Adaptive Noise Reduction"*

## 📝 Visão Geral
Este projeto implementa uma defesa para Redes Neurais Profundas (DNNs) contra ataques adversários. A técnica proposta utiliza **entropia de imagem** para aplicar, de forma adaptativa, **quantização escalar** e **filtros de suavização espacial**. O sistema detecta ataques comparando a predição da imagem original com sua versão processada, sem necessidade de re-treinar o modelo.

## 🛠️ Tecnologias e Frameworks
* **Linguagens:** Python (Scripts de teste/treino) e Matlab (DeepFool).
* **Framework de Deep Learning:** Caffe (Modelos originais: CaffeNet, GoogLeNet).
* **Processamento de Imagem:** OpenCV, NumPy, Scipy.
* **Ataques Suportados:** FGSM, C&W (L2 e L_inf), DeepFool.

## ⚙️ Configuração do Ambiente
O código original do DeepDetector usa uma pilha antiga: **Python 2**, **Caffe**, **TensorFlow 1.x**, **Keras 2.x** e **CleverHans 2.x**. Para evitar conflitos com o sistema local, a replicação deve ser executada via Docker.

### 1. Construir a imagem
```bash
./docker/build_docker.sh
```
Esse script executa:
```bash
docker compose build legacy
```

### 2. Abrir um shell no ambiente
Para abrir um container descartável:
```bash
./docker/run_docker.sh
```
Também é possível passar um comando diretamente:
```bash
./docker/run_docker.sh python scripts/check_legacy_environment.py
```
Para iniciar ou reentrar em um container persistente chamado `adversarialimage-ids-legacy`:
```bash
./docker/start_docker.sh
```
### 3. Verificar dependências
Dentro do container:
```bash
python scripts/check_legacy_environment.py
```
Se os imports de `caffe`, `tensorflow`, `cleverhans`, `keras`, `scipy`, `skimage`, `pandas` e `matplotlib` passarem, o ambiente está pronto para a etapa de replicação.

## 📂 Estrutura do Repositório
```text
.
├── src/
│   ├── replication/      # Código para reproduzir os resultados do artigo original
│   │   ├── mnist/        # Testes com dataset MNIST
│   │   └── imagenet/     # Testes com dataset ImageNet
│   ├── new_dataset/      # Implementação do sistema em um novo dataset (ex: GTSRB)
│   └── improvement/      # Implementação da proposta de melhoria da equipe
├── data/                 # Datasets e exemplos gerados (não versionados)
├── docker/               # Ambiente legado com Caffe/TensorFlow/CleverHans
├── models/               # Arquivos .prototxt e .caffemodel
├── analysis/             # Scripts para geração de gráficos e métricas (Pandas/Matplotlib)
├── docs/                 # Documentação
└── requirements.txt      # Lista de dependências
```

## 🚀 Roadmap de Execução
### Fase 1: Replicação (Até 15/05)
- [x] Configurar ambiente (Docker com Caffe e Organizar datasets utilizados).
- [ ] Gerar exemplos adversários (FGSM, CW, DeepFool) conforme o artigo.
- [ ] Executar o pipeline de detecção adaptativa e validar resultados.

### Fase 2: Novo Dataset (Até 25/05)
- [ ] Escolha e justificativa do dataset.
- [ ] Adaptação dos scripts de extração de entropia e filtros.

### Fase 3: Proposta de Melhoria (Até 05/06)
- [ ] Implementação de filtros alternativos (ex: Non-Local Means).
- [ ] Comparação de performance (Acurácia vs. Taxa de Detecção).

### Fase 4: Finalização (Até 10/06)
- [ ] Redação do relatório no formato IEEE.
- [ ] Preparação dos slides de apresentação.

## 📋 Requisitos da Disciplina
* **Resultados:** Obter valores próximos aos do artigo original.
* **Relatório:** Deve responder às perguntas de motivação, modelo de ameaça e metodologia.
* **Apresentação:** 15 minutos (obrigatória para quem realizar melhorias).
* **Prazo Final:** 10 de Junho de 2026.
