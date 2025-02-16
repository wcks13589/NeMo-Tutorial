import os
from nemo.collections.llm import deploy

# Set environment variables
os.environ["OUTPUT_GENERATION_LOGITS"] = "False"
os.environ["OPENAI_FORMAT_RESPONSE"] = "True"

NEMO_MODEL = "nemo_ckpt/Llama-3.1-8B-Instruct/"

if __name__ == "__main__":
    deploy(
        nemo_checkpoint=NEMO_MODEL,
        model_type="llama",
        triton_model_name="triton_model",
        num_gpus=2,
        tensor_parallelism_size=2,
        pipeline_parallelism_size=1,
        dtype="bfloat16",
        max_input_len=8192,
        max_output_len=2048,
        max_batch_size=64,
    )