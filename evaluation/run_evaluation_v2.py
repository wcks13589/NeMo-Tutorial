from typing import Dict, List

from lm_eval import evaluator

from nemo.lightning import io
from nemo.collections.llm.evaluation.base import NeMoFWLMEval
from nemo.collections.llm.evaluation.api import EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams

NEMO_MODEL = "nemo_ckpt/Llama-3.1-8B-Instruct/"
eval_task = "gsm8k"

class NeMoFWLMEvalWithTokenizer(NeMoFWLMEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def tokenizer_name(self):
        return self.tokenizer.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated


api_endpoint = ApiEndpoint(
    url="http://localhost:8000/",
    model_id="triton_model",
    nemo_checkpoint_path=NEMO_MODEL,
    nemo_triton_http_port=8080,
)

params = ConfigParams(
    top_p=0.9,
    temperature=0.6,
    max_new_tokens=256,
    batch_size=64,
    num_fewshot=5,
)

tokenizer = io.load_context(api_endpoint.nemo_checkpoint_path + "/context", subpath="model.tokenizer")

model = NeMoFWLMEvalWithTokenizer(
    model_name=api_endpoint.model_id,
    api_url=api_endpoint.url,
    tokenizer=tokenizer,
    batch_size=params.batch_size,
    max_tokens_to_generate=params.max_new_tokens,
    temperature=params.temperature,
    top_p=params.top_p,
    top_k=params.top_k,
    add_bos=params.add_bos,
)

results = evaluator.simple_evaluate(
    model=model,
    tasks=eval_task,
    limit=params.limit_samples,
    num_fewshot=params.num_fewshot,
    bootstrap_iters=params.bootstrap_iters,
    apply_chat_template=True,
    fewshot_as_multiturn=True,
)

print("score", results["results"][eval_task])
