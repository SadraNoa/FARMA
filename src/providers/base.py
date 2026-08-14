"""
کلاس پایه‌ی انتزاعی برای providerهای مدل (چه مدل تحت آزمون چه مدل داور).
هر provider باید متد generate را پیاده‌سازی کند.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    raw_response: dict
    model_id: str
    finish_reason: str | None = None


class BaseProvider(ABC):
    def __init__(self, model_id: str, base_url: str, api_key: str,
                 temperature: float = 0.2, max_tokens: int = 2048):
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> GenerationResult:
        """یک تولید متن از مدل انجام می‌دهد و GenerationResult برمی‌گرداند."""
        raise NotImplementedError

    @abstractmethod
    async def agenerate(self, system_prompt: str, user_prompt: str, **kwargs) -> GenerationResult:
        """نسخه‌ی async همان generate، برای اجرای موازی."""
        raise NotImplementedError
