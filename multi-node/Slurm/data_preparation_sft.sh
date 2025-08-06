#!/bin/bash

ml slurm

PARTITION=defq
CONTAINER=nvcr.io/nvidia/nemo:25.07
NUM_GPUS=8

NEMO_PATH=${PWD}

srun -p ${PARTITION} -G ${NUM_GPUS} \
--container-image ${CONTAINER} \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty bash -c \
"python data_preparation/download_sft_data.py"