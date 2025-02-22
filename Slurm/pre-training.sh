#!/bin/bash

ml slurm

PARTITION=p1
NEMO_PATH=${PWD}/NeMo
CONTAINER=/mnt/nemo2502.sqsh

NUM_NODES=1
NUM_GPUS=8

HF_MODEL_ID=Llama-3.1-8B-Instruct
NEMO_MODEL=${NEMO_PATH}/nemo_ckpt/Llama-3.1-8B-Instruct

experiment_id=$(srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty bash -c \
"python pre-training.py \
    -p ${PARTITION} \
    -N ${NUM_NODES} \
    -G ${NUM_GPUS} \
    -i ${CONTAINER} \
    --hf_model_id ${HF_MODEL_ID} \
    -n ${NEMO_MODEL}" | grep "id: " | sed -E 's/.*id: ([^ ]+).*/\1/')

echo ${experiment_id}
# sbatch --requeue --parsable ${NEMO_PATH}/experiments/llama31-8b-pretraining/${experiment_id}/pretraining_sbatch.sh
