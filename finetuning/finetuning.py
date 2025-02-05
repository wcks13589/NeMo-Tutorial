import os
import glob

import pytorch_lightning as pl

import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer

NUM_GPUS = 8
TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
MODEL = None

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant trained to provide accurate and helpful information. "
    "Please respond to the user's queries promptly and politely."
)

PROMPT_TEMPLATE = f"""\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{{input}}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{output}}\
"""

def find_latest_checkpoint(directory="nemo-experiments/llama31_pretraining/checkpoints"):
    checkpoint_files = glob.glob(os.path.join(directory, "**", "*last*"), recursive=True)
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

def configure_dataset(
    gbs: int = 8,
    mbs: int = 1,
    seq_length: int = 8192,
) -> run.Config[pl.LightningDataModule]:

    return run.Config(
        llm.FineTuningDataModule,
        dataset_root="data/alpaca",
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=TOKENIZER),
        seq_length=seq_length,
        dataset_kwargs={"prompt_template": PROMPT_TEMPLATE}
    )

def configure_recipe(nodes: int = 1, gpus_per_node: int = 8):
    recipe = llm.llama31_8b.pretrain_recipe(
        dir="nemo-experiments",
        name="llama31_finetuning",
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

def local_executor_torchrun(devices: int = 8) -> run.LocalExecutor:
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "HF_TOKEN_PATH": "/tokens/huggingface",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(devices)))
    }
    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)
    return executor

def run_finetuning(num_gpus=8):
    recipe = configure_recipe(gpus_per_node=NUM_GPUS)
    executor = local_executor_torchrun(devices=NUM_GPUS)

    if MODEL:
        checkpoint = MODEL
    else
        checkpoint = find_latest_checkpoint()
        
    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig, path=checkpoint),
        resume_if_exists=True
    )

    with run.Experiment("llama31-8b-finetuning") as exp:
        exp.add(recipe, executor=executor, name="finetuning")
        exp.run(sequential=True, tail_logs=True)

if __name__ == "__main__":
    run_finetuning()