from nemo.collections.llm import evaluate
from nemo.collections.llm.evaluation.api import EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams

NEMO_MODEL = "nemo_ckpt/Llama-3.1-8B-Instruct/"

api_endpoint = ApiEndpoint(
    url="http://localhost:8000/",
    model_id="triton_model",
    nemo_checkpoint_path=NEMO_MODEL,
    nemo_triton_http_port=8080,
)

eval_task = "gsm8k"
eval_cfg = ConfigParams(
    top_p=0.7,
    temperature=1e-07,
    max_new_tokens=256,
    batch_size=16,
)

evaluate(
    target_cfg=EvaluationTarget(api_endpoint=api_endpoint),
    eval_cfg=EvaluationConfig(
        type=eval_task,
        params=eval_cfg
    )
)