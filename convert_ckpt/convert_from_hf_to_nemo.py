from pathlib import Path
from nemo.collections import llm

# Constants for configuration
HF_MODEL_ID = "Llama-3.1-8B-Instruct"
OUTPUT_PATH = "nemo_ckpt/Llama-3.1-8B-Instruct"
OVERWRITE_EXISTING = False

def import_checkpoint():
    """
    Imports a checkpoint from Hugging Face to NeMo format.
    """
    # Step 1: Initialize configuration and model
    cfg = llm.Llama31Config8B()
    model = llm.LlamaModel(config=cfg)

    # Step 2: Log the process
    print(f"Initializing model with HF model ID: {HF_MODEL_ID}")
    print(f"Output will be saved to: {OUTPUT_PATH}")

    # Step 3: Import the checkpoint
    try:
        llm.import_ckpt(
            model=model,
            source=f"hf://{HF_MODEL_ID}",
            output_path=Path(OUTPUT_PATH),
            overwrite=OVERWRITE_EXISTING,
        )
    except Exception as e:
        print(f"Error during checkpoint conversion: {e}")
        raise

if __name__ == "__main__":
    import_checkpoint()