from nemo.collections.llm import evaluate

NEMO_MODEL = "nemo_ckpt/Llama-3.1-8B-Instruct/"
evaluate(
    nemo_checkpoint_path=NEMO_MODEL,
    url="http://0.0.0.0:8080/",
    model_name="triton_model",
    eval_task="tmmluplus"
)