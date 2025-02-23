# Llama 3.1-8B 微調腳本說明

此文件說明如何使用 `finetuning/finetuning.py` 腳本進行 Llama 3.1-8B 模型的微調。該腳本基於 NVIDIA NeMo 平臺，並利用 PyTorch Lightning 與 NeMo 完成微調工作。

---

## 主要功能

- 自定義資料集的路徑與參數。
- 配置微調的模型參數與分佈式訓練配置。
- 支持使用 `torchrun` 作為啟動分佈式訓練的工具。
- 支持檢查點恢復與自動化。

---

## 內容概要

### 資料集設定

資料集的設定通過 `configure_dataset` 函數完成，包括：

- **資料夾結構**：微調的資料應按照如下結構放置。
- **batch_size**：`global_batch_size` 與 `micro_batch_size`。
- **seq_length**：設定模型輸入的序列長度（預設 8192）。
- **tokenizer**：使用 Huggingface 的標記器，指定為 `meta-llama/Llama-3.1-8B-Instruct`。
- **prompt_template**：提供 prompt 格式化模板。

#### 資料集格式

微調的資料需使用 JSON Lines 格式，每行為一個 JSON 對象，包含 `input` 與 `output` 欄位。

示例資料：

```json
{"input":"請解釋 Python 的主要功能。","output":"Python 是一種高效能的程式語言，具有易於學習、強大的標準庫與多樣化的應用場景。"}
{"input":"如何在資料中搜尋字串？","output":"可以使用 Python 的 `in` 關鍵字或者 `re` 模組進行正則表達式匹配。"}
```

#### 資料夾結構

資料應按以下結構儲存：

```
data/alpaca/
├── gpt4-chinese-zhtw.jsonl
├── test.jsonl
├── training.jsonl
└── validation.jsonl
```

- `training.jsonl`：訓練數據集。
- `validation.jsonl`：驗證數據集。
- `test.jsonl`：測試數據集。

---

## 執行方法

1. 確保資料按照上述格式與結構準備好。

2. 執行 `finetuning.py`：

   ```bash
   JOB_NAME=llama31_finetuning

   NUM_NODES=1
   NUM_GPUS=8

   HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

   HF_TOKEN=<HF_TOKEN>

   GBS=128
   MAX_STEPS=100
   TP=2
   PP=1
   CP=1

   DATASET_PATH=data/alpaca

   python finetune.py \
      --executor local \
      --experiment ${JOB_NAME} \
      --num_nodes ${NUM_NODES} \
      --num_gpus ${NUM_GPUS} \
      --hf_model_id ${HF_MODEL_ID} \
      --hf_token ${HF_TOKEN} \
      --max_steps ${MAX_STEPS} \
      --global_batch_size ${GBS} \
      --tensor_model_parallel_size ${TP} \
      --pipeline_model_parallel_size ${PP} \
      --context_parallel_size ${CP} \
      --dataset_path ${DATASET_PATH}
   ```

---

## 參數說明

以下是 `finetune.py` 訓練腳本的參數說明。你可以透過指令行參數來設定模型訓練的方式，例如選擇執行環境、指定模型、設定批次大小等。

## **基本執行選項**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--executor` | `str` | `local` | 選擇執行方式，可選 `slurm`（使用 Slurm 叢集）或 `local`（單機執行）。 |
| `-E, --experiment` | `str` | `llama31_pretraining` | 設定實驗名稱，這會影響輸出資料夾的命名。 |

## **Slurm 相關設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-a, --account` | `str` | `root` | Slurm 的帳戶名稱，適用於需要多帳戶管理的 HPC 環境。 |
| `-p, --partition` | `str` | `defq` | Slurm 叢集的 Partition 名稱，不同叢集可能有不同的 Partition。 |
| `-i, --container_image` | `str` | `nvcr.io/nvidia/nemo:dev` | 指定要執行的 NeMo Docker 容器映像檔（通常是 NVIDIA NGC 提供的映像）。 |

## **硬體設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-N, --num_nodes` | `int` | `1` | 設定要使用的計算節點數量（適用於多機環境）。 |
| `-G, --num_gpus` | `int` | `8` | 每個計算節點使用的 GPU 數量（如果是單機訓練，通常設定為可用 GPU 數）。 |

## **模型與資料設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--hf_model_id` | `str` | **(必填)** | 指定要使用的 Hugging Face 模型 ID，例如 `"meta-llama/Llama-3-8B"`。 |
| `-n, --nemo_model` | `str` | `None` | 指定已訓練好的 NeMo 模型權重檔案（通常用於微調）。 |
| `--hf_token` | `str` | **(必填)** | Hugging Face 的 API Token，用以下載 tokenizer。 |
| `--dataset_path` | `str` | **(必填)** | 設定訓練資料夾的路徑，此資料夾應包含 `.bin` 和 `.idx` 檔案。 |

## **訓練超參數**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--max_steps` | `int` | `None` | 設定最大訓練步數（如果不設定，則以資料集大小計算 1 epoch 來決定）。 |
| `-g, --global_batch_size` | `int` | `2048` | 訓練時的全域 batch size，需為 `micro_batch_size * data_parallel_size` 的倍數。 |
| `-m, --micro_batch_size` | `int` | `1` | 設定微批次大小，通常取決於單個 GPU 可承受的記憶體大小。 |

## **模型平行化設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-T, --tensor_model_parallel_size` | `int` | `1` | 設定 Tensor Model Parallelism（張量模型平行化）。 |
| `-P, --pipeline_model_parallel_size` | `int` | `1` | 設定 Pipeline Model Parallelism（管線平行化）。 |
| `-C, --context_parallel_size` | `int` | `1` | 設定 Context Parallelism（上下文平行化）。 |

---

## 檢查點恢復

若訓練中斷，可通過以下參數自動恢復：

- **策略**：`resume_if_exists=True`。
- **恢復路徑**：可透過 `restore_config` 指定檢查點路徑（如 `experiments/llama31_pretraining/checkpoints/...`）。

重新執行訓練腳本時，會自動檢查是否存在檢查點並進行恢復。

---

## 訓練過程輸出與紀錄

訓練過程的輸出會被記錄於實驗目錄 `experiments` 下，並支持即時監控與自動化管理。

---

