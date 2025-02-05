import pytorch_lightning as pl

import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer

NUM_GPUS = 8
TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
MODEL = "nemo_ckpt/Llama-3.1-8B-Instruct"

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

def run_pretraining():
    recipe = configure_recipe(gpus_per_node=NUM_GPUS)
    executor = local_executor_torchrun(devices=NUM_GPUS)

    if MODEL:
        restore_config = run.Config(nl.RestoreConfig, path=MODEL)
    else
        restore_config = None
        
    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=restore_config,
        resume_if_exists=True
    )

    with run.Experiment("llama31-8b-pretraining") as exp:
        exp.add(recipe, executor=executor, name="pretraining")
        exp.run(sequential=True, tail_logs=True) # This will run the tasks sequentially and stream the logs

if __name__ == "__main__":
    run_pretraining()