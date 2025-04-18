#!/bin/bash

ml slurm

## 參數設定
PARTITION=defq
NEMO_PATH=${PWD}
CONTAINER=nvcr.io/nvidia/nemo:dev

MODEL=llama3_8b
# MODEL=llama3_70b

HF_MODEL_ID=Llama-3.1-8B-Instruct
# HF_MODEL_ID=Llama-3.3-70B-Instruct

OUTPUT_PATH=${NEMO_PATH}/nemo_ckpt/${HF_MODEL_ID}
OVERWRITE_EXISTING=false

export HF_TOKEN=<HF_TOKEN>

if [ ! -d ${NEMO_PATH} ]; then
  mkdir ${NEMO_PATH}
fi

## 下載模型權重
huggingface-cli download meta-llama/$HF_MODEL_ID --local-dir ${NEMO_PATH}/$HF_MODEL_ID --local-dir-use-symlinks=False 

## 執行模型轉換(to nemo)
srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty bash -c \
"nemo llm import -y model=${MODEL} source=hf://${NEMO_PATH}/${HF_MODEL_ID} output_path=${OUTPUT_PATH} overwrite=${OVERWRITE_EXISTING}"