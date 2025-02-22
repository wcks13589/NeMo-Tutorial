import os
from typing import Optional
import pytorch_lightning as pl

import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer

from nemo_run.core.tunnel.client import LocalTunnel

USERNAME = "clchiu"
NUM_NODES = 1
NUM_GPUS = 8
TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
MODEL = "nemo_ckpt/Llama-3.1-8B-Instruct"
NEMO_IMAGE = "/mnt/nemo2502.sqsh"

def configure_dataset(
    gbs: int = 8,
    mbs: int = 1,
    seq_length: int = 8192,
) -> run.Config[pl.LightningDataModule]:

    dataset_paths = [
        "data/custom_dataset/preprocessed/wikinews_text_document"
    ]

    return run.Config(
        llm.PreTrainingDataModule,
        paths=dataset_paths,
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=TOKENIZER),
        split="9998,2,0",
        num_workers=0,
        index_mapping_dir="data/custom_dataset/index_mapping",
    )

def configure_recipe(nodes: int = 1, gpus_per_node: int = 8):
    recipe = llm.llama31_8b.pretrain_recipe(
        dir="nemo-experiments",
        name="llama31_pretraining",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )
    recipe.data = configure_dataset()
    recipe.trainer.devices = gpus_per_node
    
    recipe.trainer.max_steps = 30
    recipe.trainer.val_check_interval = 10
    recipe.trainer.num_sanity_val_steps = 0
    
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.pipeline_model_parallel_size = 1
    recipe.trainer.strategy.context_parallel_size = 1
    
    recipe.log.ckpt.save_optim_on_train_end = True
    
    return recipe

def slurm_executor(
    account: str,
    nodes: int,
    devices: int,
    time: str = "30-00:00:00",
    container_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
) -> run.SlurmExecutor:

    mounts = []
    # Custom mounts are defined here.
    if container_mounts:
        mounts.extend(container_mounts)

    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
    executor = run.SlurmExecutor(
        account=account,
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
        packager=run.Packager(),
        tunnel=LocalTunnel(job_dir=os.getcwd())
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor

def run_pretraining():
    recipe = configure_recipe(nodes=NUM_NODES, gpus_per_node=NUM_GPUS)
    executor = slurm_executor(
        account=USERNAME,
        container_image=NEMO_IMAGE, # TODO: Set the container image you want to use for your job
        nodes=recipe.trainer.num_nodes,
        devices=recipe.trainer.devices,
    )

    if MODEL:
        restore_config = run.Config(nl.RestoreConfig, path=MODEL)
    else:
        restore_config = None
        
    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=restore_config,
        resume_if_exists=True
    )

    with run.Experiment("llama31-8b-pretraining", base_dir=os.getcwd()) as exp:
        exp.add(recipe, executor=executor, name="pretraining")
        exp.dryrun()

if __name__ == "__main__":
    run_pretraining()