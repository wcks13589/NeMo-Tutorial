#!/bin/bash

ml slurm

PARTITION=defq
NEMO_PATH=${PWD}/NeMo
CONTAINER=nvcr.io/nvidia/nemo:dev
HF_MODEL_ID=Llama-3.1-8B-Instruct
# HF_MODEL_ID=Llama-3.3-70B-Instruct
HF_TOKEN=<HF_TOKEN>

cd ${NEMO_PATH}

## 下載資料集
mkdir -p data/custom_dataset/json/
CMD="from datasets import load_dataset; dataset = load_dataset('erhwenkuo/wikinews-zhtw', '20231001')['train']; dataset.to_json('data/custom_dataset/json/wikinews-zhtw.jsonl', num_proc=112, force_ascii=False)"

srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty python -c "$CMD"

## 資料前處理
mkdir -p data/custom_dataset/preprocessed

srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-env HF_TOKEN=${HF_TOKEN} \
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