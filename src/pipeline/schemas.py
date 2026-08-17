"""
اسکیمای مشترک داده در طول پایپ‌لاین: از نمونه‌ی خام تسک، تا خروجی مدل،
تا نتیجه‌ی نهایی امتیازدهی‌شده. هر تسک (ریاضی، BBH، تقویم جلالی و...) یک
فایل jsonl مطابق TaskSample تولید می‌کند و از همین پایپ‌لاین مشترک عبور
می‌دهد.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class TaskSample(BaseModel):
    """فرمت عمومی هر نمونه‌ی ورودی، مستقل از نوع تسک.
    فیلد extra برای نگه‌داشتن فیلدهای اختصاصی هر تسک (مثلاً constraints
    برای IFEval یا gold_meter_pattern برای عروض) استفاده می‌شود."""

    sample_id: str
    task_type: str
    problem_fa: str
    gold_answer: Optional[str] = None
    requires_cot_judging: bool = False
    apply_verifiable_reasoning_rubric: bool = False
    apply_persian_stability_rubric: bool = False
    apply_deep_brainstorm_rubric: bool = False
    apply_persian_controllability_rubric: bool = False
    system_prompt_fa: Optional[str] = None
    answer_extraction_regex: str = r"پاسخ نهایی:\s*(.+)"
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelGeneration(BaseModel):
    sample_id: str
    model_name: str
    raw_output: str
    extracted_final_answer: Optional[str] = None
    finish_reason: Optional[str] = None


class ScoredResult(BaseModel):
    sample_id: str
    model_name: str
    task_type: str

    correctness: Optional[int] = None  # 0/1، فقط برای تسک‌های دارای gold ساده

    verifiable_reasoning_total: Optional[int] = None
    verifiable_reasoning_breakdown: Optional[dict[str, Any]] = None

    persian_stability_llm_total: Optional[int] = None
    code_switch_rate: Optional[float] = None
    persian_script_ratio: Optional[float] = None
    persian_stability_breakdown: Optional[dict[str, Any]] = None

    deep_brainstorm_total: Optional[int] = None
    deep_brainstorm_breakdown: Optional[dict[str, Any]] = None

    controllability_hard_fail: Optional[bool] = None
    controllability_satisfaction_rate: Optional[float] = None
    controllability_semantic_fidelity: Optional[int] = None
    controllability_breakdown: Optional[dict[str, Any]] = None

    raw_output: str = ""
    notes: Optional[str] = None
