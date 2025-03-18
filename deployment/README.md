# TensorRT-LLM 教學筆記

此教學筆記旨在展示如何使用 TensorRT-LLM 對 Llama-3.1-8B-Instruct 模型進行優化，執行推理，並運用各種先進的優化技術。

## 環境設定

本教學適用於 NVIDIA Triton 容器環境：  
`nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3`

請確保您已安裝並設定好 Docker。

### 啟動 JupyterLab

可以透過以下 `docker run` 指令啟動包含 JupyterLab 的容器環境：

```bash
docker run --gpus all --rm -it \
  --shm-size=2g \
  -p 8888:8888 \
  -v ${PWD}:/workspace \
  nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 \
  bash -c "pip3 install jupyterlab && jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser"
```

啟動容器後，複製終端機中帶有 token 的 URL，並在瀏覽器中開啟以存取 JupyterLab。

---

## TensorRT-LLM 簡介

TensorRT-LLM 提供了一個易於使用的 Python API，用於定義和優化大型語言模型（LLMs），以在 NVIDIA GPU 上高效地執行推理。其主要特性包括：

- 支援多種數據精度優化：FP16、FP8、INT8 和 INT4。
- 提供 SmoothQuant 和群組式量化（AWQ/GPTQ）技術。
- 支援單 GPU 或多 GPU 的張量並行化（Tensor Parallelism）。
- 與 NVIDIA Triton 推理伺服器的整合。

---

## 教學步驟

### 1. 下載模型

透過 Hugging Face CLI 下載 Llama-3.1-8B-Instruct 模型至本地資料夾。

### 2. 建立 TensorRT 引擎

筆記中提供了詳細的示例，演示如何基於 Hugging Face 模型檢查點建構各種數據精度（FP16、INT8、FP8 等）的 TensorRT 引擎。

---

### 3. 啟動推理伺服器

完成 TensorRT 引擎構建後，可以透過`trtllm-serve `啟動推理伺服器。

---

## 附註

- 教學筆記展示了如何透過 `trtllm-build` 指令使用不同量化模式進行優化。
- 如需更多技術細節，請參考官方 [TensorRT-LLM 文件](https://github.com/NVIDIA/TensorRT-LLM)。

---