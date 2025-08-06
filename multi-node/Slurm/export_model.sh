#!/bin/bash

ml slurm

## 參數設定
PARTITION=defq
CONTAINER=nvcr.io/nvidia/nemo:25.07
NUM_GPUS=8

NEMO_PATH=${PWD}

LATEST_CHECKPOINT=$(find ${NEMO_PATH}/experiments/<JOB_NAME>/<JOB_ID>/pretraining/code/nemo_experiments/<JOB_NAME>/ -type d -name "*-last" | sort -r | head -n 1)
OUTPUT_PATH=${NEMO_PATH}/hf_ckpt

## 執行模型轉換(to huggingface)
srun -p ${PARTITION} -G ${NUM_GPUS} \
--container-image ${CONTAINER} \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty bash -c \
"nemo llm export -y path=${LATEST_CHECKPOINT} output_path=${OUTPUT_PATH} target=hf"