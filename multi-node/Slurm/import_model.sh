#!/bin/bash

ml slurm

## 參數設定
PARTITION=defq
CONTAINER=nvcr.io/nvidia/nemo:25.07
NUM_GPUS=8

NEMO_PATH=${PWD}

MODEL=llama31_8b
# MODEL=llama31_70b
# MODEL=qwen3_30b_a3b

export HF_TOKEN=<HF_TOKEN>
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
# HF_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
# HF_MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507

OUTPUT_PATH=${NEMO_PATH}/nemo_ckpt/${HF_MODEL_ID}
OVERWRITE_EXISTING=false

if [ ! -d ${NEMO_PATH} ]; then
  mkdir ${NEMO_PATH}
fi

## 下載模型權重與執行模型轉換(to nemo)
srun -p ${PARTITION} -G ${NUM_GPUS} \
--container-image ${CONTAINER} \
--container-env HF_TOKEN \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty bash -c \
"huggingface-cli download ${HF_MODEL_ID} --local-dir ${NEMO_PATH}/${HF_MODEL_ID} && \
 nemo llm import -y model=${MODEL} source=hf://${NEMO_PATH}/${HF_MODEL_ID} output_path=${OUTPUT_PATH} overwrite=${OVERWRITE_EXISTING}"