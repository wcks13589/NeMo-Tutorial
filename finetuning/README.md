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
   python finetuning/finetuning.py
   ```

---

## 配置參數

- **全局批量大小 (`global_batch_size`)**：定義看過多少訓練樣本後更新模型權重一次。
- **微批量大小 (`micro_batch_size`)**：定義每個 GPU 的批量。
- **序列長度 (`seq_length`)**：設置模型的最大輸入長度。
- **節點數量 (`nodes`)**：分佈式訓練的節點數。
- **每節點 GPU 數量 (`gpus_per_node`)**：每個節點的 GPU 數量。
- **prompt_template**：定義對話範本以適配應用場景。

---

## 檢查點恢復

若訓練中斷，可通過以下參數自動恢復：

- **策略**：`resume_if_exists=True`。
- **恢復路徑**：可透過 `restore_config` 指定檢查點路徑（如 `nemo-experiments/llama31_pretraining/checkpoints/...`）。

重新執行腳本時，會自動檢查是否存在檢查點並進行恢復。

---

## 訓練過程輸出與紀錄

訓練過程的輸出會被記錄於實驗目錄 `nemo-experiments` 下，並支持即時監控與自動化管理。

---

