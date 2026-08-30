"""Small lm-evaluation-harness compatibility patches for local-completions."""

import time


def response_has_logprobs(response) -> bool:
    outputs = response if isinstance(response, list) else [response]
    for output in outputs:
        if not isinstance(output, dict):
            return False
        choices = output.get("choices") or []
        if not choices or any(
            not isinstance(choice.get("logprobs"), dict)
            or not isinstance(choice["logprobs"].get("token_logprobs"), list)
            for choice in choices
        ):
            return False
    return True


def install() -> None:
    """Retry intermittent null-logprobs responses before harness parsing."""
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    original_model_call = LocalCompletionsAPI.model_call

    def resilient_model_call(self, *args, **kwargs):
        if kwargs.get("generate", True):
            return original_model_call(self, *args, **kwargs)
        for attempt in range(3):
            response = original_model_call(self, *args, **kwargs)
            if response_has_logprobs(response):
                return response
            if attempt < 2:
                print(
                    "lm-eval received a response without logprobs; "
                    f"retrying ({attempt + 1}/2)",
                    flush=True,
                )
                time.sleep(0.5)
        raise RuntimeError(
            "local-completions returned logprobs=null after 3 attempts; "
            "the server cannot score this request reliably"
        )

    LocalCompletionsAPI.model_call = resilient_model_call
