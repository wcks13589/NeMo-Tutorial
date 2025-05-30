# 🧪 Manual Multi-Node Training with NeMo (No Slurm / No K8s)

本教學說明如何使用手動方式在多個節點上進行 NeMo 的模型訓練，不依賴 Slurm、Kubernetes 等任務調度工具，完全以 NeMo-Run 中的 Local Executor (torchrun) 在容器中啟動跨節點訓練。

適合需要靈活控制、測試環境或小型叢集的使用者。

---

## 📥 1. 取得腳本（所有節點皆需執行）

在操作流程之前，請先執行以下命令以下載本專案的程式庫。

```bash
git clone https://github.com/wcks13589/NeMo-Tutorial.git
cd NeMo-Tutorial
```

---

## 📦 2. 啟動 NeMo 容器（所有節點皆需執行）

請在**每個計算節點**上執行以下 Docker 指令以啟動訓練容器：

```bash
docker run \
    --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:$PWD -w $PWD --network host \
    nvcr.io/nvidia/nemo:25.04.rc2
```

此容器包含所有所需的核心依賴套件，包括 NeMo、PyTorch 和其他相關工具。請確保您的腳本和資料已掛載到容器內以進行後續操作。

---

 
 ## 🔧 3. 更新 NeMo 程式碼與套件（所有節點皆需執行）
 在**每個計算節點**中的容器中執行以下指令以拉取最新版本的 NeMo-Run 程式碼與安裝必要套件：
 
 ```bash
 cd /opt/NeMo-Run/
 git pull origin main
 pip install toml
 ```
 
 ---

## 🔽 4. 模型下載與轉換（僅主節點執行）

### 登入 Hugging Face

```bash
huggingface-cli login --token <HF_TOKEN>
```

### 下載模型

以 `meta-llama/Llama-3.1-8B-Instruct` 為例，執行以下命令下載模型：

```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks=False
```

### 模型轉換

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

請將轉換好的模型放置於所有節點共享可存取的位置（例如NFS）。

---

## 📚 5. 資料下載與預處理（僅主節點執行）

下載範例資料集並進行處理：

```python
from datasets import load_dataset

dataset = load_dataset('erhwenkuo/wikinews-zhtw')['train']
dataset.to_json('./data/custom_dataset/json/wikinews-zhtw.jsonl', force_ascii=False)
exit()
```

資料預處理以適配 NeMo 訓練格式：

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

同樣地，預處理後的資料需放置在所有節點共享可存取的位置。

---

## 🌐 6. 多節點環境變數設置（依節點分別執行）
每台節點皆需設置以下環境變數。注意 `NODE_RANK` 需根據節點編號做調整。

以2個節點為例：

### 🖥️ 節點 1
```bash
export MASTER_ADDR=<節點 1 的主機名稱/IP位址>
export MASTER_PORT=12345
export NODE_RANK=0
```

### 🖥️ 節點 2
```bash
export MASTER_ADDR=<節點 1 的主機名稱/IP位址>
export MASTER_PORT=12345
export NODE_RANK=1
```

📌 若有更多節點，依序增加 `NODE_RANK`（節點 3 設為 2，節點 4 設為 3，以此類推）。
`MASTER_ADDR` 必須是節點 1 的 主機名稱 / IP 位址，所有節點皆需一致。

---

## ⚙️ 7. 執行多節點預訓練 Pre-training（所有節點皆需執行）
設定環境變數後，於每個節點上執行以下指令：

```bash
JOB_NAME=llama31_pretraining

NUM_NODES=2
NUM_GPUS=8

HF_MODEL_ID=Llama-3.1-8B-Instruct
NEMO_MODEL=/mnt/nemo_ckpt/${HF_MODEL_ID}
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

---

## 🔧 8. 多節點微調 Fine-tuning（所有節點皆需執行）

下載微調資料：

```bash
python finetuning/download_split_data.py
```

執行微調：

```bash
JOB_NAME=llama31_finetuning

NUM_NODES=2
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

---

## 📝 附註與建議
- 所有節點應可互相通訊，請確保防火牆與網路設定允許使用`MASTER_PORT`。
- `NEMO_MODEL`, `DATASET_PATH` 路徑需確保為所有節點可存取（建議使用 NFS 或共享磁碟）。
- `HF_TOKEN` 需具備 Hugging Face 權限以存取模型。