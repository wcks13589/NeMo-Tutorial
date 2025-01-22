# Llama 3.1-8B 預訓練腳本說明

此文件說明如何使用 `pretraining.py` 腳本進行大型語言模型 Llama 3.1-8B 的預訓練。該腳本培於 NVIDIA NeMo 平臺，並利用 PyTorch Lightning 與 NeMo 進行預訓練工作。

## 主要功能

- 配置資料集的路徑與參數。
- 配置訓練的參數與分佈式訓練參數。
- 支持使用 `torchrun` 作為啟動分佈式訓練的工具。
- 允許恢復中斷的訓練。

---

## 內容概要

### 資料集設定

資料集的設定通過 `configure_dataset` 函數完成，包括：

- **paths**：預處理後的資料儲放位置（預設為 `data/custom_dataset/preprocessed/wikinews_text_document`）。
- **batch_size**：`global_batch_size`與`micro_batch_size`。
- **seq_length**：設定模型輸入的序列長度（預設 8192）。
- **分割比例**：分割訓練、驗證、測試資料的比例。
- **tokenizer**：使用 Huggingface 提供的標記器。

#### 資料集格式

預訓練需要使用格式为 JSON Lines (每行一個 JSON 對象)的資料集。每個 JSON 對象應包含一個字段 `text`，用來儲存文本內容。

以 Wikinews 為例，資料格式如下：

```
{"text":"中華民國...."}
{"text":"日近有程式設計師...."}
```

若要替換成其他資料集，可以導入相同格式的 JSON Lines。例如：
##### 客製化資料集

```
{"text":"Python 是一個強大而簡潔的程式語言，它擁有很多工具和平台支援...."}
{"text":"在測試過程中，我們採用了優化的模型構廻以提高準確性...."}
```

### 資料夾結構

- **資料處理**：使用預處理工具對原始資料進行Tokenize的動作。
- **資料儲存**：前處理完成後，資料應儲存在 `.bin` 和 `.idx` 格式中，模型訓練前需確認資料夾結構。

```
data
└─ custom_dataset
    ├─ json
    │   └─ wikinews-zhtw.jsonl
    └─ preprocessed
        ├─ wikinews_text_document.bin
        └─ wikinews_text_document.idx
```

---

## 執行方法

1. 確保資料已下載並預處。

   預處完畢後，資料目錄中必須包含 `.bin` 與 `.idx` 檔案，結構如上。

2. 執行 `pretraining.py`：

   ```bash
   python pretraining.py
   ```

---

## 配置參數

- **全局批量大小 (`global_batch_size`)**：定義看過多少訓練檔案更新一次模型權重。
- **微批量大小 (`micro_batch_size`)**：定義每個 GPU 的批量。
- **序列長度 (`seq_length`)**：設置模型的最大輸入長度。
- **節點數量 (`nodes`)**：設定分佈式訓練的節點數。
- **每節點 GPU 數量 (`gpus_per_node`)**：定義每個節點的 GPU 數量。

---

## 檢查點恢復

若訓練中斷，可通過下列參數自動恢復：

- **策略**：`resume_if_exists=True`。

重新執行腳本，會自動檢查是否存在檢查點並進行恢復。

---

## 訓練過程輸出與紀錄

訓練過程中的輸出會被記錄於實驗目錄`nemo-experiments`下，並支持即時監控。

---