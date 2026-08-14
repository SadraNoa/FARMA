"""
Provider عمومی برای هر endpoint سازگار با OpenAI Chat Completions API.
vLLM، OpenRouter و GapGPT هر سه از این پروتکل پیروی می‌کنند، پس یک کلاس
برای هر سه کافی است و فقط base_url/api_key فرق می‌کند.
"""

from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, GenerationResult


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, model_id: str, base_url: str, api_key: str,
                 temperature: float = 0.2, max_tokens: int = 2048,
                 retry_attempts: int = 3, retry_backoff_seconds: int = 2):
        super().__init__(model_id, base_url, api_key, temperature, max_tokens)
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._retry_attempts = retry_attempts
        self._retry_backoff_seconds = retry_backoff_seconds

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> GenerationResult:
        @retry(stop=stop_after_attempt(self._retry_attempts),
               wait=wait_exponential(multiplier=self._retry_backoff_seconds))
        def _call():
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return resp

        resp = _call()
        choice = resp.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            raw_response=resp.model_dump(),
            model_id=self.model_id,
            finish_reason=choice.finish_reason,
        )

    async def agenerate(self, system_prompt: str, user_prompt: str, **kwargs) -> GenerationResult:
        @retry(stop=stop_after_attempt(self._retry_attempts),
               wait=wait_exponential(multiplier=self._retry_backoff_seconds))
        async def _call():
            resp = await self._async_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return resp

        resp = await _call()
        choice = resp.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            raw_response=resp.model_dump(),
            model_id=self.model_id,
            finish_reason=choice.finish_reason,
        )
