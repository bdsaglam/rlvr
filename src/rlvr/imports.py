"""Central import handling for platform-specific dependencies."""
from typing import Any, Dict, List, Optional, Protocol, Union

from vllm import CompletionOutput, RequestOutput, SamplingParams

__all__ = ["ILLM", "LLM", "CompletionOutput", "RequestOutput", "SamplingParams"]


class ILLM(Protocol):
    def chat(
        messages: Union[List[Dict], List[List[Dict]]],
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Any] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: str = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[RequestOutput]: ...


def LLM(model: str, **kwargs) -> ILLM:
    if model.startswith("openai/"):
        from verifiers.api_llm import LLM as APILLM
        model = model.replace("openai/", "")
        return APILLM(model, **kwargs)
    else:
        from vllm import LLM as VLLM

        return VLLM(model, **kwargs)
