"""
ارزیابی پایداری زبان فارسی در طول یک زنجیره استدلال.
بخشی از این ارزیابی کاملاً rule-based است (بدون هزینه‌ی LLM) و بخشی
نیازمند مدل داور است (روانی دستوری، ثبات اصطلاحات، افت کیفیت در طول متن).
"""

import json
from dataclasses import dataclass, field

from src.providers.base import BaseProvider
from src.utils.text_utils import (
    code_switch_rate,
    persian_script_ratio,
    split_first_last_third,
    try_parse_json,
)


@dataclass
class PersianStabilityScore:
    # بخش rule-based
    code_switch_rate_per_100_words: float
    persian_script_ratio: float
    # بخش LLM-judge
    grammatical_fluency: int
    terminology_consistency: int
    degradation_over_length: int
    llm_part_total: int
    brief_reason_fa: str
    raw_judge_output: str
    parse_success: bool


class PersianStabilityJudge:
    def __init__(self, judge_provider: BaseProvider, rubric_path: str = "rubrics/persian_stability.json"):
        self.provider = judge_provider
        with open(rubric_path, "r", encoding="utf-8") as f:
            self.rubric = json.load(f)

    def _rule_based_part(self, cot: str) -> dict:
        whitelist_cfg = next(
            m for m in self.rubric["rule_based_metrics"] if m["id"] == "code_switching_rate"
        )["whitelist"]
        whitelist = set(whitelist_cfg)
        return {
            "code_switch_rate_per_100_words": code_switch_rate(cot, whitelist),
            "persian_script_ratio": persian_script_ratio(cot),
        }

    def _build_criteria_list(self) -> str:
        lines = []
        for c in self.rubric["llm_judge_metrics"]:
            lines.append(f"- {c['id']} (۰ تا {c['scale_max']}): {c['question_fa']}")
        return "\n".join(lines)

    def _llm_based_part(self, cot: str) -> dict:
        system_prompt = self.rubric["judge_system_prompt_fa"]
        template = self.rubric["judge_user_prompt_template_fa"]
        user_prompt = template.format(cot=cot, criteria_list=self._build_criteria_list())

        result = self.provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        parsed = try_parse_json(result.text)

        if parsed is None:
            return {
                "grammatical_fluency": 0,
                "terminology_consistency": 0,
                "degradation_over_length": 0,
                "brief_reason_fa": "خطا: خروجی داور قابل parse نبود.",
                "raw_judge_output": result.text,
                "parse_success": False,
            }

        return {
            "grammatical_fluency": int(parsed.get("grammatical_fluency", 0)),
            "terminology_consistency": int(parsed.get("terminology_consistency", 0)),
            "degradation_over_length": int(parsed.get("degradation_over_length", 0)),
            "brief_reason_fa": parsed.get("brief_reason_fa", ""),
            "raw_judge_output": result.text,
            "parse_success": True,
        }

    def score(self, cot: str) -> PersianStabilityScore:
        rule_based = self._rule_based_part(cot)
        llm_based = self._llm_based_part(cot)

        llm_total = (
            llm_based["grammatical_fluency"]
            + llm_based["terminology_consistency"]
            + llm_based["degradation_over_length"]
        )

        return PersianStabilityScore(
            code_switch_rate_per_100_words=rule_based["code_switch_rate_per_100_words"],
            persian_script_ratio=rule_based["persian_script_ratio"],
            grammatical_fluency=llm_based["grammatical_fluency"],
            terminology_consistency=llm_based["terminology_consistency"],
            degradation_over_length=llm_based["degradation_over_length"],
            llm_part_total=llm_total,
            brief_reason_fa=llm_based["brief_reason_fa"],
            raw_judge_output=llm_based["raw_judge_output"],
            parse_success=llm_based["parse_success"],
        )
