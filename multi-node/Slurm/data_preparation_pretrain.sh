#!/bin/bash

ml slurm

PARTITION=defq
CONTAINER=nvcr.io/nvidia/nemo:25.07
NUM_GPUS=8

NEMO_PATH=${PWD}

export HF_TOKEN=<HF_TOKEN>
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
# HF_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
# HF_MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507

## 下載資料集
mkdir -p data/custom_dataset/json/

srun -p ${PARTITION} -G ${NUM_GPUS} \
--container-image ${CONTAINER} \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty bash -c \
"python data_preparation/download_pretrain_data.py \
    --dataset_name erhwenkuo/wikinews-zhtw \
    --output_dir data/custom_dataset/json/wikinews-zhtw.jsonl"

## 資料前處理
mkdir -p data/custom_dataset/preprocessed

srun -p ${PARTITION} -G ${NUM_GPUS} \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-env HF_TOKEN \
--container-writable \
--no-container-mount-home \
--pty bash -c \
"python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=data/custom_dataset/json/wikinews-zhtw.jsonl \
    --json-keys=text \
    --dataset-impl mmap \
    --tokenizer-library=huggingface \
    --tokenizer-type ${HF_MODEL_ID} \
    --output-prefix=data/custom_dataset/preprocessed/wikinews \
    --append-eod"