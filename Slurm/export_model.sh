#!/bin/bash

ml slurm

## 參數設定
PARTITION=defq
NEMO_PATH=${PWD}/NeMo
CONTAINER=nvcr.io/nvidia/nemo:dev
NEMO_MODEL=${NEMO_PATH}/experiments/llama31_pretraining/llama31_pretraining_1740317067/pretraining/code/results/llama31_finetuning/checkpoints/model_name\=0--val_loss\=1.38-step\=99-consumed_samples\=1600.0-last/
OUTPUT_PATH=${NEMO_PATH}/hf_ckpt

## 執行模型轉換(to nemo)
srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty bash -c \
"nemo llm export -y path=${NEMO_MODEL} output_path=${OUTPUT_PATH} target=hf"