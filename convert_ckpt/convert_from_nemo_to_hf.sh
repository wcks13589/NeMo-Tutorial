#!/bin/bash

NEMO_MODEL=nemo-experiments/llama31_finetuning/checkpoints/model_name\=0--val_loss\=1.55-step\=9-consumed_samples\=80.0-last/
OUTPUT_PATH=hf_ckpt

nemo llm export -y path=${NEMO_MODEL} output_path=${OUTPUT_PATH} target=hf