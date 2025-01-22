import pytorch_lightning as pl

import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer

def configure_dataset(
    gbs: int = 8,
    mbs: int = 1,
    seq_length: int = 8192,
) -> run.Config[pl.LightningDataModule]:

    return run.Config(
        llm.PreTrainingDataModule,
        paths=["data/custom_dataset/preprocessed/wikinews_text_document"],
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name="meta-llama/Llama-3.1-8B-Instruct"),
        split="9998,2,0",
        num_workers=0,
        index_mapping_dir="data/index_mapping",
    )

def configure_recipe(nodes: int = 1, gpus_per_node: int = 4):

    recipe = llm.llama31_8b.pretrain_recipe(
        dir="nemo-experiments", # Path to store checkpoints
        name="llama31_pretraining",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    recipe.data = configure_dataset()
    recipe.trainer.max_steps = 10
    recipe.trainer.num_sanity_val_steps = 0

    # Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.val_check_interval = 5

    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 8) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def run_pretraining():
    recipe = configure_recipe()

    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    # Change executor params
    executor.ntasks_per_node = 4
    executor.env_vars["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    executor.env_vars["HF_TOKEN_PATH"] = "/tokens/huggingface"

    # Change recipe params
    # We also need to set TP to 1, since we had used 2 for 2 GPUs.
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    # Lastly, we need to set devices to 1 in the trainer.
    recipe.trainer.devices = 4
    recipe.log.ckpt.save_optim_on_train_end=True
    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig, path="nemo_ckpt/Llama-3.1-8B-Instruct"),
        resume_if_exists=True
    )

    with run.Experiment("llama31-8b-pretraining") as exp:
        exp.add(recipe, executor=executor, name="pretraining")
        exp.run(sequential=True, tail_logs=True) # This will run the tasks sequentially and stream the logs

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()