# 使用 NVIDIA NeMo 進行大型語言模型 (LLM) 的訓練 🤖

本研究專案旨在系統性地闡述如何利用 [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) 平台完成大型語言模型（LLM）的轉換 🔄、預訓練 📚 及微調 🛠️ 流程。以下內容涵蓋的四個主要階段包括：

1. **模型轉換：從 Huggingface 格式匯入至 NeMo 格式** 🔄
2. **預訓練（Pretraining）** 📖
3. **微調（Finetuning）** ✨
4. **模型轉換：從 NeMo 格式匯出至 Huggingface 格式** 🔃

每個階段均對應獨立的 Python 腳本 🐍，確保功能模組化。

---

## 📂 目錄 📖

- [⚙️ 環境設定](#⚙️-環境設定)
- [📥 取得腳本](#📥-取得腳本)
- [🛠️ 操作流程](#🛠️-操作流程)
  - [1️⃣ 模型轉換：從 Huggingface 匯入至 NeMo](#1️⃣-模型轉換從-huggingface-匯入至-nemo)
  - [2️⃣ 預訓練（Pretraining）](#2️⃣-預訓練pretraining)
  - [3️⃣ 微調（Finetuning）](#3️⃣-微調finetuning)
  - [4️⃣ 模型轉換：從 NeMo 匯出至 Huggingface](#4️⃣-模型轉換從-nemo-匯出至-huggingface)
- [📚 參考資料](#📚-參考資料)

---

## ⚙️ 環境設定 🖥️

NVIDIA NeMo 容器會隨 NeMo 版本更新同步發布，您可以在 [NeMo 版本發布頁面](https://github.com/NVIDIA/NeMo/releases) 查詢更多資訊。

### 使用預構建容器（推薦）

此方式適用於希望使用穩定版本 NeMo 的大多數使用者。請按照以下步驟操作：

1. 登錄或免費註冊一個帳戶： [NVIDIA GPU Cloud (NGC)](https://ngc.nvidia.com/signin)。

2. 登錄後，您可以在 [NVIDIA NGC NeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) 查看所有容器版本。

3. 在終端中執行以下指令啟動容器：

```bash
docker run \
    --gpus all -it --rm --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:$PWD -w $PWD -p 8888:8888 \
    nvcr.io/nvidia/nemo:25.02
```

此容器包含所有所需的核心依賴套件，包括 NeMo、PyTorch 和其他相關工具。請確保您的腳本和資料已掛載到容器內以進行後續操作。

---

## 📥 取得腳本

在操作流程之前，請先執行以下命令以下載本專案的程式庫。

```bash
git clone https://github.com/wcks13589/NeMo-Tutorial.git
cd NeMo-Tutorial
```

---

## 🛠️ 操作流程 ⚙️

### 1️⃣ 模型轉換：從 Huggingface 匯入至 NeMo 🔄

在此步驟中，需將 Huggingface 格式的模型權重轉換為 NeMo 格式。

在執行轉換腳本之前，請先確保已經登入 Huggingface 帳戶並下載模型。

#### 登入 Huggingface
執行以下命令以完成登入：

```bash
huggingface-cli login --token <HF_TOKEN>
```

#### 下載模型
以 `meta-llama/Llama-3.1-8B-Instruct` 為例，執行以下命令以下載模型：

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir Llama-3.1-8B-Instruct --local-dir-use-symlinks=False
```

#### 模型轉換
登入並下載模型後，您即可進行模型轉換。執行以下命令：

Option 1: 透過Python
```bash
HF_MODEL_ID=Llama-3.1-8B-Instruct
OUTPUT_PATH=nemo_ckpt/Llama-3.1-8B-Instruct

python convert_ckpt/convert_from_hf_to_nemo.py \
  --source ${HF_MODEL_ID} \
  --output_path ${OUTPUT_PATH}
```

Option 2: 透過Cli
```bash
MODEL=llama3_8b
HF_MODEL_ID=Llama-3.1-8B-Instruct
OUTPUT_PATH=nemo_ckpt/Llama-3.1-8B-Instruct
OVERWRITE_EXISTING=false

nemo llm import -y model=${MODEL} source=hf://${HF_MODEL_ID} output_path=${OUTPUT_PATH} overwrite=${OVERWRITE_EXISTING}
```

### 2️⃣ 持續預訓練（Continual Pretraining） 📖

#### 資料下載與處理 🗂️

在進行預訓練之前，需下載並處理相應的語料數據。

下載資料集：

```python
from datasets import load_dataset

dataset = load_dataset('erhwenkuo/wikinews-zhtw')['train']
dataset.to_json('./data/custom_dataset/json/wikinews-zhtw.jsonl', force_ascii=False)
exit()
```

預處理資料以適配 NeMo 格式：

```bash
mkdir -p data/custom_dataset/preprocessed

python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=data/custom_dataset/json/wikinews-zhtw.jsonl \
    --json-keys=text \
    --dataset-impl mmap \
    --tokenizer-library=huggingface \
    --tokenizer-type meta-llama/Llama-3.1-8B-Instruct \
    --output-prefix=data/custom_dataset/preprocessed/wikinews \
    --append-eod
```

#### 預訓練過程

利用大規模語料對模型進行語言建模訓練，以提升其泛化能力。

執行以下命令：

```bash
JOB_NAME=llama31_pretraining

NUM_NODES=1
NUM_GPUS=8

HF_MODEL_ID=Llama-3.1-8B-Instruct
NEMO_MODEL=nemo_ckpt/${HF_MODEL_ID}
HF_TOKEN=<HF_TOKEN>

TP=2
PP=1
CP=1

GBS=2048
MAX_STEPS=100
DATASET_PATH=data/custom_dataset/preprocessed/

python pretraining/pretrain.py \
    --executor local \
    --experiment ${JOB_NAME} \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS} \
    --model_size 8B \
    --hf_model_id meta-llama/${HF_MODEL_ID} \
    --nemo_model ${NEMO_MODEL} \
    --hf_token ${HF_TOKEN} \
    --max_steps ${MAX_STEPS} \
    --global_batch_size ${GBS} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --dataset_path ${DATASET_PATH}
```

更多欲訓練的詳細參數資訊，請查閱`pretraining`資料夾內的說明文件 📄。

### 指令微調 (Instruction Tuning)

##### 資料下載 🗂️

在進行微調之前，需下載並處理相應的語料資料。

下載資料集：

```bash
python finetuning/download_split_data.py
```

針對具體任務或指令語言進行模型微調。執行以下命令：

```bash
JOB_NAME=llama31_finetuning

NUM_NODES=1
NUM_GPUS=8

HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
NEMO_MODEL= # [Optional]
HF_TOKEN=<HF_TOKEN>

TP=2
PP=1
CP=1

MAX_STEPS=100
GBS=128
DATASET_PATH=data/alpaca

python finetuning/finetune.py \
    --executor local \
    --experiment ${JOB_NAME} \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS} \
    --model_size 8B \
    --hf_model_id ${HF_MODEL_ID} \
    --hf_token ${HF_TOKEN} \
    --nemo_model ${NEMO_MODEL} \
    --max_steps ${MAX_STEPS} \
    --global_batch_size ${GBS} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --dataset_path ${DATASET_PATH}
```

更多微調的詳細參數資訊，請查閱`finetuning`資料夾內的說明文件 📄。

### 4️⃣ 模型轉換：從 NeMo 匯出至 Huggingface 🔃

最後，將經過訓練或微調的 NeMo 格式模型轉換為 Huggingface 格式，以便後續的兼容性或部署需求。

執行以下命令：

Option 1: 透過Python
```bash
NEMO_MODEL=results/llama31_finetuning/checkpoints/model_name\=0--val_loss\=1.55-step\=9-consumed_samples\=80.0-last/
OUTPUT_PATH=hf_ckpt

python convert_ckpt/convert_from_nemo_to_hf.py \
  --source ${NEMO_MODEL} \
  --output_path ${OUTPUT_PATH}
```

Option 2: 透過Cli
```bash
NEMO_MODEL=results/llama31_finetuning/checkpoints/model_name\=0--val_loss\=1.38-step\=99-consumed_samples\=1600.0-last/
OUTPUT_PATH=hf_ckpt

nemo llm export -y path=${NEMO_MODEL} output_path=${OUTPUT_PATH} target=hf
```

---

## 📚 參考資料 📘

- [NVIDIA NeMo 官方文件](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) 📄