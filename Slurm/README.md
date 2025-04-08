# 🚀 使用 Slurm 執行大型語言模型（LLM）訓練任務

以下將說明如何透過 Slurm 作業管理系統執行大型語言模型（LLM）的完整訓練流程。內容涵蓋資料準備、模型轉換、預訓練與微調，使用腳本可直接提交至 GPU 叢集運行。

---

## 📁 腳本目錄

| 檔案名稱                        | 功能說明                                 |
|-------------------------------|----------------------------------------|
| `data_preparation.sh`         | 預訓練語料的資料準備（資料下載與格式轉換）    |
| `download_finetuning_data.sh` | 微調資料下載與切分                            |
| `import_model.sh`             | 將 HuggingFace 模型轉換為 NeMo 格式            |
| `run_pretraining.sh`          | 執行 NeMo 模型的持續預訓練                    |
| `run_finetuning.sh`           | 執行 NeMo 模型的指令微調                     |
| `export_model.sh`             | 將 NeMo 格式模型匯出為 HuggingFace 格式         |

---

## ⚙️ 環境與依賴

- 作業系統：HPC 環境（支援 Slurm）
- GPU：支援 H100, H200, B200（8x GPU 建議）
- Pyxis / enroot：Slurm 需具備容器化插件

---

## 🧱 操作流程

---

### 1️⃣ 模型轉換：從 Huggingface 匯入至 NeMo 🔄

將 Huggingface 上的 模型轉換成 NeMo 2.0 所支援的格式，以便後續進行訓練：

```bash
bash import_model.sh
```
請先確保腳本內已設定完成欲下載的 Huggingface 模型名稱與路徑，確保順利轉換。

### 2️⃣ 預訓練（Pretraining）📚
#### 🔧 資料準備
預訓練前需準備乾淨、格式正確的語料資料：

```bash
bash data_preparation.sh
```
該腳本會進行以下操作：
- 下載範例資料集
- 資料前處理(.jsonl → .bin 與建立索引與映射)

#### 🚀 執行預訓練任務
準備完成後，即可提交 Slurm 任務進行預訓練：

```bash
run_pretraining.sh
```

你可以在`run_pretraining.sh`中調整：
- GPU / Node 數量
- 訓練模型的規模
- 訓練步數 (MAX_STEPS)
- 全域批次大小 (GBS)
- 資料集路徑 (DATASET_PATH)

## 3️⃣ 微調（Finetuning）🛠️
### 🔧 資料下載與準備
微調前需下載並切分資料
```bash
bash download_finetuning_data.sh
```
此腳本將下載 `erhwenkuo/alpaca-data-gpt4-chinese-zhtw` 資料集，並切分成
- 訓練集 (`training.jsonl`)
- 驗證集 (`validation.jsonl`)
- 測試集(`test.jsonl`)

### ✨ 執行微調任務
資料準備好後，執行：
```bash
bash run_finetuning.sh
```
`run_finetuning.sh`內你可以調整：
- GPU / Node 數量
- 訓練模型的規模
- 訓練步數 (MAX_STEPS)
- 全域批次大小 (GBS)
- 資料集路徑 (DATASET_PATH)

## 4️⃣ 模型轉換：從 NeMo 匯出至 Huggingface 格式 🔃
當模型訓練完成後，若需部署或進一步應用，可將其轉為 Huggingface 格式：
```bash
bash export_model.sh
```
此腳本將 NeMo 格式模型檔案匯出成 Huggingface 格式權重檔。