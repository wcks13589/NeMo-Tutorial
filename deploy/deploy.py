import os
from nemo.collections.llm import deploy

# Set environment variables
os.environ["OUTPUT_GENERATION_LOGITS"] = "True"
os.environ["OPENAI_FORMAT_RESPONSE"] = "True"

NEMO_MODEL = "nemo_ckpt/Llama-3.1-8B-Instruct/"

if __name__ == "__main__":
    deploy(
        nemo_checkpoint=NEMO_MODEL,
        model_type="llama",
        triton_model_name="triton_model",
        num_gpus=1,
        tensor_parallelism_size=1,
        pipeline_parallelism_size=1,
        dtype="bfloat16",
        max_input_len=8192,
        max_output_len=8192,
        max_batch_size=16,
    )