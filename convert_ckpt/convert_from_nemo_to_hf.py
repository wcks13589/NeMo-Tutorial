from pathlib import Path
from nemo.collections.llm import export_ckpt

# Constants for configuration
NEMO_MODEL = "nemo-experiments/llama31_finetuning/checkpoints/model_name=0--val_loss=1.55-step=9-consumed_samples=80.0-last"
OUTPUT_PATH = "hf_ckpt"

def export_checkpoint():
    """
    Imports a checkpoint from NeMo to Huggingface format.
    """
    print(f"Huggingface weight will be saved to: {OUTPUT_PATH}")

    # Export the checkpoint
    try:
        export_ckpt(
            path=NEMO_MODEL,
            output_path=Path(OUTPUT_PATH),
            target="hf",
        )
    except Exception as e:
        print(f"Error during checkpoint conversion: {e}")
        raise

if __name__ == "__main__":
    export_checkpoint()
