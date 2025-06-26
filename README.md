# 使用 NVIDIA NeMo 進行大型語言模型 (LLM) 的訓練 🤖

本專案將帶您完整體驗使用 [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) 進行大型語言模型（LLM）的完整流程，從零開始學會模型轉換、預訓練、微調到部署的實戰技巧。

## 🎯 學習目標

通過本專案，您將學會：

1. **🔄 模型轉換技能**：掌握 Hugging Face 與 NeMo 格式間的轉換
2. **📚 預訓練實踐**：體驗大規模語言模型的持續預訓練
3. **🛠️ 微調技術**：學會針對特定任務進行模型微調、掌握 LoRA 等參數高效微調方法
4. **📊 模型推論**：學會測試模型性能
5. **🚀 模型轉換與評估**：了解模型導出和評估方法

---

## 📂 教學大綱

- [🚀 開始之前：環境設定](#🛠️-環境準備)
  - [📖 詳細環境設定指南](setup/README.md) ⭐
- [📥 專案設置](#📥-專案設置)
- [📖 詳細教學步驟](#📖-詳細教學步驟)
  - [第一章：模型轉換基礎](#第一章模型轉換基礎)
  - [第二章：持續預訓練實戰](#第二章持續預訓練實戰)
  - [第三章：指令微調技術](#第三章指令微調技術)
  - [第四章：Reasoning 資料微調技術](#第四章reasoning-資料微調技術)
  - [第五章：模型評估與測試](#第五章模型評估與測試)
  - [第六章：模型部署與轉換](#第六章模型部署與轉換)
  - [第七章：標準化模型評估](#第七章標準化模型評估)
- [💡 實戰技巧](#💡-實戰技巧)
- [📚 進階學習資源](#📚-進階學習資源)

---

## 🛠️ 環境準備

### 📋 選擇您的環境

在開始訓練之前，您需要設定適合的 GPU 環境。我們提供詳細的環境設定指南：

#### 🐳 本地 Docker 環境

使用官方 NeMo 容器，包含所有必要的依賴套件：

```bash
docker run \
    --gpus all -it --rm --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:$PWD -w $PWD -p 8888:8888 \
    nvcr.io/nvidia/nemo:25.04
```

> 💡 **提示**：您可以在 [NGC NeMo Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) 查看最新版本

---

## 📥 專案設置

### 下載教學專案

```bash
git clone https://github.com/wcks13589/NeMo-Tutorial.git
cd NeMo-Tutorial
```

> 💡 **提示**：請確保在 `NeMo-Tutorial` 專案目錄中執行後續指令。

### 🔑 設定 Hugging Face 權限

申請並設定您的 Hugging Face Token：

1. **申請 Token**：前往 [Hugging Face Settings](https://huggingface.co/settings/tokens) 建立新的 Access Token
2. **設定環境變數**：
   ```bash
   # 替換為您的實際 Token
   export HF_TOKEN="your_hf_token"
   huggingface-cli login --token $HF_TOKEN
   ```

> 📌 **重要**：請先在 [Hugging Face](https://huggingface.co/settings/tokens) 申請 Access Token

## 📖 詳細教學步驟

### 第一章：模型轉換基礎

#### 1.1 下載預訓練模型

```bash
# 下載 Llama 3.1 8B Instruct 模型
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir Llama-3.1-8B-Instruct \
    --exclude original/
```

#### 1.2 轉換為 NeMo 格式

**Option 1:** 使用Python腳本

```bash
# 設定變數
HF_MODEL_ID=Llama-3.1-8B-Instruct
OUTPUT_PATH=nemo_ckpt/Llama-3.1-8B-Instruct

python convert_ckpt/convert_from_hf_to_nemo.py \
  --source ${HF_MODEL_ID} \
  --output_path ${OUTPUT_PATH}
```

**Option 2:** 使用CLI
```bash
# 設定變數
MODEL=llama3_8b
HF_MODEL_ID=Llama-3.1-8B-Instruct
OUTPUT_PATH=nemo_ckpt/Llama-3.1-8B-Instruct
OVERWRITE_EXISTING=false

# 執行轉換
nemo llm import -y \
    model=${MODEL} \
    source=hf://${HF_MODEL_ID} \
    output_path=${OUTPUT_PATH} \
    overwrite=${OVERWRITE_EXISTING}
```

> ✅ **檢查點**：確認 `nemo_ckpt/` 目錄下成功生成了 NeMo 格式的模型檔案

---

### 第二章：持續預訓練實戰

#### 2.1 準備訓練資料

**下載中文資料集**：

```bash
python data_preparation/download_pretrain_data.py \
    --dataset_name erhwenkuo/wikinews-zhtw \
    --output_dir data/custom_dataset/json/wikinews-zhtw.jsonl
```

#### 2.2 資料預處理

```bash
# 建立預處理目錄
mkdir -p data/custom_dataset/preprocessed

# 使用 NeMo 的資料預處理工具
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=data/custom_dataset/json/wikinews-zhtw.jsonl \
    --json-keys=text \
    --dataset-impl mmap \
    --tokenizer-library=huggingface \
    --tokenizer-type meta-llama/Llama-3.1-8B-Instruct \
    --output-prefix=data/custom_dataset/preprocessed/wikinews \
    --append-eod
```

#### 2.3 執行預訓練

##### 前置準備

設定基本參數：

```bash
JOB_NAME=llama31_pretraining
NUM_NODES=1
NUM_GPUS=8
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

# 平行處理參數
TP=2  # Tensor Parallel
PP=1  # Pipeline Parallel  
CP=1  # Context Parallel

# 訓練參數
GBS=4          # Global Batch Size
MAX_STEPS=20  # 最大訓練步數(模型權重更新次數)
DATASET_PATH=data/custom_dataset/preprocessed/
```

##### 方法一：從頭開始預訓練模型

**適用情況**：當您想要從零開始訓練模型時使用。

**特點**：腳本會自動從基礎模型架構進行權重初始化

**執行指令**：
```bash
python pretraining/pretrain.py \
   --executor local \
   --experiment ${JOB_NAME} \
   --num_nodes ${NUM_NODES} \
   --num_gpus ${NUM_GPUS} \
   --model_size 8B \
   --hf_model_id ${HF_MODEL_ID} \
   --hf_token ${HF_TOKEN} \
   --max_steps ${MAX_STEPS} \
   --global_batch_size ${GBS} \
   --tensor_model_parallel_size ${TP} \
   --pipeline_model_parallel_size ${PP} \
   --context_parallel_size ${CP} \
   --dataset_path ${DATASET_PATH}
```

##### 方法二：從預訓練模型開始繼續預訓練

**適用情況**：當您想要從現有的 NeMo 格式模型開始，進行持續預訓練時使用。

**前置條件**：
- 需要先將 Hugging Face 模型轉換為 NeMo 格式
- 確保 `${NEMO_MODEL}` 路徑下存在有效的 NeMo 模型檔案

**執行指令**：
```bash
NEMO_MODEL=nemo_ckpt/Llama-3.1-8B-Instruct

python pretraining/pretrain.py \
   --executor local \
   --experiment ${JOB_NAME} \
   --num_nodes ${NUM_NODES} \
   --num_gpus ${NUM_GPUS} \
   --model_size 8B \
   --hf_model_id ${HF_MODEL_ID} \
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

### 第三章：模型微調 (Fine-tuning)

#### 3.1 準備微調資料

```bash
# 下載並準備 Alpaca 資料集
python data_preparation/download_sft_data.py
```

#### 3.2 執行模型微調

```bash
# 微調參數設定
JOB_NAME=llama31_finetuning
NUM_NODES=1
NUM_GPUS=8
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
# NEMO_MODEL=nemo_ckpt/Llama-3.1-8B-Instruct
NEMO_MODEL=$(find nemo_experiments/llama31_pretraining/checkpoints/ -type d -name "*-last" | sort -r | head -n 1)
HF_TOKEN=$HF_TOKEN

# 平行處理參數
TP=2
PP=1
CP=1

# 微調參數
MAX_STEPS=10
GBS=4
DATASET_PATH=data/alpaca

# 執行 LoRA 微調
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
#    --peft lora
```

> 🎯 **高效參數參數微調**：若要進行高效參數的微調，請新增`--peft lora`

---

### 第四章：Reasoning 資料微調技術
> 🧠 **Reasoning 微調特色**：使用高品質的推理資料集，提升模型的邏輯推理和複雜問題解決能力

#### 4.1 準備 Reasoning 資料集

```bash
# 建立 Reasoning 資料集目錄
mkdir -p data/reasoning_dataset/

# 下載 NVIDIA Llama-Nemotron 後訓練資料集
wget https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/resolve/main/SFT/chat/chat.jsonl -P data/reasoning_dataset/

# 從資料集中選取樣本進行快速訓練
head -n 200 data/reasoning_dataset/chat.jsonl > data/reasoning_dataset/chat_subset.jsonl
```

#### 4.2 資料預處理與策展

```bash
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=tcp

# 執行資料策展與預處理
python data_preparation/curate_reasoning_data.py \
    --input-dir "data/reasoning_dataset" \
    --filename-filter "chat_subset" \
    --remove-columns "category" "generator" "license" "reasoning" "system_prompt" "used_in_training" "version" \
    --json-files-per-partition 16 \
    --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    --max-token-count 16384 \
    --max-completion-token-count 8192 \
    --output-dir data/reasoning_dataset/curated-data \
    --device "gpu" \
    --n-workers 1
```

> 💡 **執行提示**：此程式執行過程中可能會出現一些錯誤訊息，但只要輸出資料夾 `data/reasoning_dataset/curated-data` 內有檔案產生就算執行成功，可以忽略錯誤訊息繼續後續步驟。

#### 4.3 執行 Reasoning 微調

```bash
# Reasoning 微調參數設定
JOB_NAME=llama31_reasoning_finetuning
NUM_NODES=1
NUM_GPUS=8
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
# NEMO_MODEL=nemo_ckpt/Llama-3.1-8B-Instruct
NEMO_MODEL=$(find nemo_experiments/llama31_finetuning/checkpoints/ -type d -name "*-last" | sort -r | head -n 1)
HF_TOKEN=$HF_TOKEN

# 平行處理參數
TP=2
PP=1
CP=1

# 微調參數
MAX_STEPS=10
GBS=4
DATASET_PATH=data/reasoning_dataset/curated-data

# 執行 Reasoning 微調
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
> 🎯 **高效參數參數微調**：若要進行高效參數的微調，請新增`--peft lora`

---

### 第五章：模型評估與測試

#### 5.1 準備測試資料

```bash
# 從測試集中選取樣本進行快速評估
head -n 30 data/alpaca/test.jsonl > data/alpaca/test_subset.jsonl
```

#### 5.2 執行推理測試

```bash
# 使用微調後的模型進行推理
# 找到最新的檢查點資料夾
LATEST_CHECKPOINT=$(find nemo_experiments/llama31_reasoning_finetuning/checkpoints/ -type d -name "*-last" | sort -r | head -n 1)

python evaluation/inference.py \
    --peft_ckpt_path ${LATEST_CHECKPOINT} \
    --input_dataset data/alpaca/test_subset.jsonl \
    --output_path data/alpaca/peft_prediction.jsonl
```

#### 5.3 計算評估指標

```bash
# 計算模型性能指標
python /opt/NeMo/scripts/metric_calculation/peft_metric_calc.py \
    --pred_file data/alpaca/peft_prediction.jsonl \
    --label_field "label" \
    --pred_field "prediction"
```

---

### 第六章：模型轉換

> ⚠️ **重要提醒**：如果您進行的是 LoRA 微調，請先執行步驟 6.1 合併 LoRA 權重，再進行步驟 6.2 的格式轉換。

#### 6.1 合併 LoRA 權重（僅限 LoRA 微調）

如果您使用了 LoRA 進行微調（在微調指令中包含 `--peft lora`），您需要先將 LoRA 權重合併回基底模型，然後再進行格式轉換：

```bash
# 找到最新的 LoRA checkpoint
LATEST_LORA_CHECKPOINT=$(find nemo_experiments/llama31_reasoning_finetuning/checkpoints/ -type d -name "*-last" | sort -r | head -n 1)
NEMO_MODEL=nemo_experiments/llama31_reasoning_finetuning/checkpoints/nemo_ckpt_merged

# 合併 LoRA 權重到基底模型
python finetuning/merge_lora.py \
    --nemo_lora_model ${LATEST_LORA_CHECKPOINT} \
    --output_path ${NEMO_MODEL}
```

> 💡 **說明**：
> - 此步驟會將 LoRA 適配器的權重合併到原始的基底模型中
> - 合併後的模型包含完整的權重，可以獨立使用
> - 如果您進行的是全參數微調，請跳過此步驟

#### 6.2 轉換回 Hugging Face 格式

```bash
# 設定轉換參數
# 如果您進行的是全參數微調，請使用：
NEMO_MODEL=$(find nemo_experiments/llama31_reasoning_finetuning/checkpoints/ -type d -name "*-last" | sort -r | head -n 1)
# 如果您完成了 LoRA 合併，請使用合併後的模型路徑：
# NEMO_MODEL=nemo_experiments/llama31_reasoning_finetuning/checkpoints/nemo_ckpt_merged

OUTPUT_PATH=hf_ckpt/

# 執行轉換
nemo llm export -y \
    path=${NEMO_MODEL} \
    output_path=${OUTPUT_PATH} \
    target=hf
```

---

### 第七章：標準化模型評估

#### 7.1 安裝評估工具

使用 EleutherAI 的 lm-evaluation-harness 工具進行標準化模型評估：

```bash
# 下載並安裝 lm-evaluation-harness
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

#### 7.2 執行標準化評估

使用 LAMBADA OpenAI 任務評估模型的語言建模能力：

```bash
# 切換回主目錄
cd ..

# 執行 LAMBADA OpenAI 評估 (僅使用較少樣本進行快速評估)
lm_eval --model hf \
    --model_args pretrained=hf_ckpt/ \
    --tasks lambada_openai \
    --device cuda:0 \
    --batch_size 8 \
    --limit 100
```

**執行結果範例**：

```
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.7100|±  |0.0456|
|              |       |none  |     0|perplexity|↓  |3.4032|±  |0.5080|
```

**結果指標說明**：
- **acc (準確率)**：模型正確預測句子最後一個詞的比例，本例為 71%
- **perplexity (困惑度)**：衡量模型對文本的不確定性，數值越低越好

#### 7.3 其他評估任務

您也可以嘗試其他常見的評估任務：

```bash
lm_eval --model hf \
    --model_args pretrained=hf_ckpt/ \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 8
```

> 📊 **評估任務說明**：
> - **LAMBADA OpenAI**：測試語言建模和上下文理解能力，評估模型預測句子最後一個詞的準確性
> - **ARC (AI2 Reasoning Challenge)**：測試科學推理能力
> 
> 🔧 **調優提示**：
> - 可根據 GPU 記憶體調整 `batch_size` 參數
> - 使用 `--limit` 參數進行快速測試
> - 詳細的評估結果會顯示準確率和其他相關指標

---

## 📚 進階學習資源

### 官方文檔
- [NeMo 官方文件](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)

### 進階主題
1. **多模態模型訓練**
2. **分散式訓練優化**
3. **模型壓縮與量化**
4. **自定義資料載入器**

---

## 🎉 恭喜完成

通過本教學，您已經掌握了：
- ✅ 大型語言模型的完整訓練流程
- ✅ NeMo 框架的核心功能
- ✅ 實際的 AI 模型開發技能
- ✅ 企業級 AI 應用開發基礎

**下一步建議**：
1. 嘗試使用自己的資料集
2. 探索不同的模型架構
3. 學習模型部署與服務化
4. 參與開源專案貢獻

---

> 💬 **需要幫助？** 歡迎在 [Issues](https://github.com/wcks13589/NeMo-Tutorial/issues) 中提出問題或建議！

**Happy Learning! 🚀**