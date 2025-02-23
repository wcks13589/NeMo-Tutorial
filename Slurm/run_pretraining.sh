#!/bin/bash

ml slurm

NEMO_PATH=${PWD}/NeMo

PARTITION=defq
CONTAINER=nvcr.io/nvidia/nemo:dev
JOB_NAME=llama31_pretraining

NUM_NODES=1
NUM_GPUS=8

HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
NEMO_MODEL=${NEMO_PATH}/nemo_ckpt/Llama-3.1-8B-Instruct

HF_TOKEN=<HF_TOKEN>

MAX_STEPS=100
GBS=2048
TP=4
PP=1
CP=1

DATASET_PATH=data/custom_dataset/preprocessed/

experiment_id=$(srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
bash -c \
"python pretrain.py \
    --executor slurm \
    --account ${USER} \
    --partition ${PARTITION} \
    --container_image ${CONTAINER} \
    --experiment ${JOB_NAME} \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS} \
    --hf_model_id ${HF_MODEL_ID} \
    --nemo_model ${NEMO_MODEL} \
    --hf_token ${HF_TOKEN} \
    --max_steps ${MAX_STEPS} \
    --global_batch_size ${GBS} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --dataset_path ${DATASET_PATH}" | grep "Experiment" | sed -n 's/.*\[[^]]*\] Experiment \([^ ]*\).*/\1/p')

echo "Your NeMo Experiment is submitted with job name: ${experiment_id}"
sbatch --requeue ${NEMO_PATH}/experiments/${JOB_NAME}/${experiment_id}/pretraining_sbatch.sh