#!/bin/bash

ml slurm

NEMO_PATH=${PWD}/NeMo

PARTITION=defq
CONTAINER=nvcr.io/nvidia/nemo:dev
JOB_NAME=llama31_finetuning

NUM_NODES=4
NUM_GPUS=8

# Manually set the model size ("8B" or "70B")
MODEL_SIZE="8B"  # Change to "70B" to switch the model
if [[ "$MODEL_SIZE" == "8B" ]]; then
    HF_MODEL_ID=Llama-3.1-8B-Instruct
    TP=2
    PP=1
    CP=1
elif [[ "$MODEL_SIZE" == "70B" ]]; then
    HF_MODEL_ID=Llama-3.3-70B-Instruct
    TP=8
    PP=4
    CP=1
else
    echo "Error: MODEL_SIZE must be '8B' or '70B'."
    exit 1
fi

NEMO_MODEL=
HF_TOKEN=<HF_TOKEN>

MAX_STEPS=100
GBS=128
DATASET_PATH=data/alpaca

experiment_id=$(srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
bash -c \
"python finetune.py \
    --executor slurm \
    --account ${USER} \
    --partition ${PARTITION} \
    --container_image ${CONTAINER} \
    --experiment ${JOB_NAME} \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS} \
    --hf_model_id meta-llama/${HF_MODEL_ID} \
    --nemo_model ${NEMO_MODEL} \
    --hf_token ${HF_TOKEN} \
    --max_steps ${MAX_STEPS} \
    --global_batch_size ${GBS} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --dataset_path ${DATASET_PATH}" | grep "Experiment" | sed -n 's/.*\[[^]]*\] Experiment \([^ ]*\).*/\1/p')

echo "Your NeMo Experiment is submitted with job name: ${experiment_id}"
sbatch --requeue ${NEMO_PATH}/experiments/${JOB_NAME}/${experiment_id}/finetuning_sbatch.sh