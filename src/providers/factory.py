"""
از روی configs/models.yaml، شیء provider مناسب (vLLM / OpenRouter / GapGPT)
را می‌سازد. چون هر سه سازگار با OpenAI API هستند، همه از یک کلاس
OpenAICompatibleProvider استفاده می‌کنند و فقط تنظیمات فرق می‌کند.
"""

import os
import yaml
from dotenv import load_dotenv

from .openai_compatible import OpenAICompatibleProvider

load_dotenv()

_SUPPORTED_PROVIDER_TYPES = {"vllm", "openrouter", "gapgpt"}


def _resolve_env(var_name: str) -> str:
    value = os.environ.get(var_name, "")
    return value


def load_models_config(config_path: str = "configs/models.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_provider_from_entry(entry: dict, runtime_cfg: dict | None = None) -> OpenAICompatibleProvider:
    """یک entry از candidate_models یا judge_models را به شیء provider تبدیل می‌کند."""
    provider_type = entry["provider"]
    if provider_type not in _SUPPORTED_PROVIDER_TYPES:
        raise ValueError(
            f"provider ناشناخته: {provider_type}. مقادیر مجاز: {_SUPPORTED_PROVIDER_TYPES}"
        )

    base_url = _resolve_env(entry["base_url_env"])
    api_key = _resolve_env(entry["api_key_env"])
    runtime_cfg = runtime_cfg or {}

    return OpenAICompatibleProvider(
        model_id=entry["model_id"],
        base_url=base_url,
        api_key=api_key,
        temperature=entry.get("temperature", 0.2),
        max_tokens=entry.get("max_tokens", 2048),
        retry_attempts=runtime_cfg.get("retry_attempts", 3),
        retry_backoff_seconds=runtime_cfg.get("retry_backoff_seconds", 2),
    )


def get_candidate_provider(model_name: str, config_path: str = "configs/models.yaml") -> OpenAICompatibleProvider:
    cfg = load_models_config(config_path)
    for entry in cfg["candidate_models"]:
        if entry["name"] == model_name:
            return build_provider_from_entry(entry, cfg.get("runtime"))
    raise ValueError(f"مدل کاندید با نام '{model_name}' در configs/models.yaml پیدا نشد.")


def get_judge_provider(judge_name: str = "judge-primary", config_path: str = "configs/models.yaml") -> OpenAICompatibleProvider:
    cfg = load_models_config(config_path)
    for entry in cfg["judge_models"]:
        if entry["name"] == judge_name:
            return build_provider_from_entry(entry, cfg.get("runtime"))
    raise ValueError(f"مدل داور با نام '{judge_name}' در configs/models.yaml پیدا نشد.")


def list_candidate_models(config_path: str = "configs/models.yaml") -> list[dict]:
    cfg = load_models_config(config_path)
    return cfg["candidate_models"]
