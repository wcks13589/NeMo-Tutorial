#!/bin/bash

ml slurm

PARTITION=defq
CONTAINER=nvcr.io/nvidia/nemo:25.07
NUM_GPUS=8

NEMO_PATH=${PWD}

JOB_NAME=model_finetuning

NUM_NODES=1
NUM_GPUS=8

# Manually set the model size ("8B" or "70B")
MODEL="llama31_8b"  # Change to "70B" to switch the model
if [[ "$MODEL" == "llama31_8b" ]]; then
    HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
    TP=2
    PP=1
    CP=1
elif [[ "$MODEL" == "llama31_70b" ]]; then
    HF_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
    TP=8
    PP=4
    CP=1
elif [[ "$MODEL" == "qwen3_30b_a3b" ]]; then
    HF_MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507
    TP=4
    PP=2
    CP=1
else
    echo "Error: MODEL must be 'llama31_8b', 'llama31_70b', or 'qwen3_30b_a3b'."
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
"python ../../finetuning/finetune.py \
    --executor slurm \
    --account ${USER} \
    --partition ${PARTITION} \
    --container_image ${CONTAINER} \
    --experiment ${JOB_NAME} \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS} \
    --model ${MODEL} \
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
sbatch --requeue ${NEMO_PATH}/experiments/${JOB_NAME}/${experiment_id}/finetuning_sbatch.sh