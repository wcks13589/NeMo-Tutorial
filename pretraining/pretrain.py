import os
import glob
import math
import argparse
from typing import Tuple

import pytorch_lightning as pl

import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer

from nemo_run.core.tunnel.client import LocalTunnel

WORK_PATH = os.getcwd()

def calculate_training_steps(
    path: str, 
    seq_length: int = 8192,
    global_batch_size: int = 2048
)-> Tuple[str, int]:
    """
    計算資料集之比例與訓練一個 epoch 所需的步數。

    Args:
        path (str): 資料集所在的資料夾。
        seq_length (int): 序列長度 (sequence length)。
        global_batch_size (int): 全域 batch size。

    Returns:
        tuple: (dataset_paths, total_steps)
    """
    total_steps = 0
    dataset_paths = []
    
    # 找出所有 .bin 檔案
    bin_files = glob.glob(os.path.join(path, "*.bin"))
    
    for file_path in bin_files:
        # 取得檔案大小（以 byte 為單位）
        size = os.path.getsize(file_path)
        # 計算該檔案可產生的 steps 數量
        num_steps = math.ceil(size / (2 * seq_length * global_batch_size))  # 無條件進位
        # 取得檔案名稱（去除副檔名）
        text_document = os.path.splitext(os.path.basename(file_path))[0]
        dataset_path_without_extension = os.path.join(os.path.dirname(file_path), text_document)
        # 儲存檔案比例
        dataset_paths.append(dataset_path_without_extension)
        # 累計總步數
        total_steps += num_steps
        
    return dataset_paths, total_steps

def configure_dataset(
    args,
    seq_length: int = 8192,
) -> run.Config[pl.LightningDataModule]:

    preprocessed_data_path = os.path.join(WORK_PATH, args.dataset_path)
    dataset_paths, total_steps = calculate_training_steps(
        path=preprocessed_data_path,
        seq_length=seq_length, 
        global_batch_size=args.global_batch_size
    )

    dataset = run.Config(
        llm.PreTrainingDataModule,
        paths=dataset_paths,
        seq_length=seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=args.hf_model_id),
        split="9998,2,0",
        num_workers=0,
        index_mapping_dir=os.path.join(WORK_PATH, "data/custom_dataset/index_mapping"),
    )

    return dataset, total_steps

def configure_recipe(args):
    recipe = llm.llama31_8b.pretrain_recipe(
        dir="experiments",
        name=args.experiment,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus,
    )
    recipe.data, one_epoch_steps = configure_dataset(args, seq_length=recipe.data.seq_length)
    recipe.trainer.devices = args.num_gpus
    
    recipe.trainer.max_steps = args.max_steps or one_epoch_steps
    recipe.trainer.val_check_interval = one_epoch_steps // 5 if one_epoch_steps > 1000 else recipe.trainer.max_steps
    recipe.trainer.num_sanity_val_steps = 0
    
    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_model_parallel_size
    recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    recipe.trainer.strategy.context_parallel_size = args.context_parallel_size
    recipe.trainer.strategy.sequence_parallel=True
    
    recipe.log.ckpt.save_optim_on_train_end = True
    
    return recipe

def slurm_executor(
    args, 
    env_vars
) -> run.SlurmExecutor:

    # Custom mounts are defined here.
    container_mounts = [f"{WORK_PATH}:{WORK_PATH}"]
    srun_args = ["--container-writable"]

    tunnel = LocalTunnel(job_dir=os.path.join(WORK_PATH, "experiments"))

    # This defines the slurm executor.
    executor = run.SlurmExecutor(
        packager=run.Packager(),
        env_vars=env_vars,
        account=args.account,
        partition=args.partition,
        time="30-00:00:00",
        nodes=args.num_nodes,
        ntasks_per_node=args.num_gpus,
        gpus_per_node=args.num_gpus,
        mem="0",
        gres="gpu:8",
        exclusive=True,
        container_image=args.container_image,
        container_mounts=container_mounts,
        srun_args=srun_args,
        tunnel=tunnel,
    )

    return executor

def run_pretraining(args):
    recipe = configure_recipe(args)

    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "HF_TOKEN": args.hf_token,
    }

    if args.executor == "slurm":
        executor = slurm_executor(args, env_vars)
    else:
        executor = run.LocalExecutor(
            launcher="torchrun", 
            ntasks_per_node=args.num_gpus, 
            env_vars=env_vars
        )

    if args.nemo_model:
        restore_config = run.Config(nl.RestoreConfig, path=args.nemo_model)
    else:
        restore_config = None

    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=restore_config,
        resume_if_exists=True
    )

    with run.Experiment(args.experiment, base_dir=WORK_PATH) as exp:
        exp.add(recipe, executor=executor, name="pretraining")
        exp.dryrun() if args.executor == "slurm" else exp.run(sequential=True, tail_logs=True)

def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Pretraining Arguments")
    
    parser.add_argument("--executor", type=str, choices=["slurm", "local"], default="local",
                        help="選擇執行方式: 'slurm'（使用 Slurm）或 'local'（單機執行）")
    
    parser.add_argument("-a", "--account", type=str, default="root", help="Slurm partition name")
    parser.add_argument("-p", "--partition", type=str, default="defq", help="Slurm partition name")
    parser.add_argument("-i", "--container_image", type=str, default="nvcr.io/nvidia/nemo:dev", help="NEMO image path")
    
    parser.add_argument("-E", "--experiment", type=str, default="llama31_pretraining", help="Name of experiment")
    parser.add_argument("-N", "--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("-G", "--num_gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--hf_model_id", type=str, required=True, help="Huggingface Model ID")
    parser.add_argument("-n", "--nemo_model", type=str, help="Pretrained NeMo Model path")
    parser.add_argument("--hf_token", type=str, required=True, help="Huggingface Token for downloading tokenizer")

    # 訓練參數
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("-g", "--global_batch_size", type=int, default=2048, help="Global batch size (must be multiple of micro_batch_size * data parallel size)")
    parser.add_argument("-m", "--micro_batch_size", type=int, default=1, help="Micro batch size per data parallel group")

    # 模型平行化參數
    parser.add_argument("-T", "--tensor_model_parallel_size", type=int, default=1,
                        help="Tensor model parallelism size")
    parser.add_argument("-P", "--pipeline_model_parallel_size", type=int, default=1,
                        help="Pipeline model parallelism size")
    parser.add_argument("-C", "--context_parallel_size", type=int, default=1,
                        help="Context parallelism size (usually 1, unless using advanced parallelism)")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to the folder containing the preprocessed dataset. "
                        "This folder should include files named in the format: "
                        "'<dataset_name>_text_document.bin' and '<dataset_name>_text_document.idx'.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretraining(args)