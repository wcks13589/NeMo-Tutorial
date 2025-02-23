#!/bin/bash

ml slurm

PARTITION=defq
NEMO_PATH=${PWD}/NeMo
CONTAINER=nvcr.io/nvidia/nemo:dev
HF_MODEL_ID=Llama-3.1-8B-Instruct

srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty python download_split_data.py