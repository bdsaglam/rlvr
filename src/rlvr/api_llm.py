from typing import Dict, List, Optional, Union

import tiktoken
from openai import OpenAI

from verifiers.imports import CompletionOutput, RequestOutput, SamplingParams

import logfire

logfire.configure()
logfire.instrument_openai()

class LLM:
    """API-based LLM implementation matching vLLM's interface"""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        # Get the encoding for the model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _encode_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        """Encode a sequence of messages into token IDs in the same way OpenAI would"""
        # Join messages in OpenAI chat format
        formatted_text = ""
        for msg in messages:
            formatted_text += f"{msg['role']}: {msg['content']}\n"
        return self.encoding.encode(formatted_text)

    def chat(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
        **kwargs,
    ) -> List[RequestOutput]:
        """Generate chat completions using the OpenAI API"""

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Handle both single conversation and batch of conversations
        if not isinstance(messages[0], list):
            messages = [messages]

        outputs = []
        for i, conversation in enumerate(messages):
            # First encode the prompt messages to get prompt token IDs
            prompt_token_ids = self._encode_messages(conversation)

            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in conversation],
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                max_tokens=sampling_params.max_tokens,
            )

            # Get completion text and tokens
            completion_text = response.choices[0].message.content
            completion_tokens = self.encoding.encode(completion_text)

            # Convert API response to vLLM-style output
            stop_reason = next((token for token in sampling_params.stop if token in completion_text[-16:]), None)
            completion = CompletionOutput(
                index=0,
                text=completion_text,
                token_ids=completion_tokens,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=response.choices[0].finish_reason,
                stop_reason=stop_reason,
            )

            # Construct full prompt text
            prompt_text = "\n".join(msg["content"] for msg in conversation)

            outputs.append(
                RequestOutput(
                    request_id=str(i),
                    prompt=prompt_text,
                    prompt_token_ids=prompt_token_ids,
                    outputs=[completion],
                    prompt_logprobs=None,
                    finished=True,
                )
            )

        return outputs
