#!/bin/bash

ml slurm

PARTITION=defq
NEMO_PATH=${PWD}
CONTAINER=nvcr.io/nvidia/nemo:dev

srun -p ${PARTITION} -G 8 \
--container-image $CONTAINER \
--container-mounts ${NEMO_PATH}:${NEMO_PATH} \
--container-workdir ${NEMO_PATH} \
--container-writable \
--no-container-mount-home \
--pty python download_split_data.py