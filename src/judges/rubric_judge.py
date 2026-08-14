"""
داور عمومی برای اعمال rubric «استدلال قابل‌راستی‌آزمایی» روی خروجی هر تسکی
که در applies_to این rubric قرار دارد (ریاضی، BBH، تناقض/سازگاری، دوگانه
حداقلی، abstention، برین‌استورم عمیق).

این ماژول عمداً به هیچ تسک خاصی وابسته نیست تا در همه‌ی فازها قابل استفاده
مجدد باشد.
"""

import json
from dataclasses import dataclass

from src.providers.base import BaseProvider
from src.utils.text_utils import try_parse_json


@dataclass
class VerifiableReasoningScore:
    step_support: int
    conclusion_validity: int
    no_logical_jump: int
    answer_consistency: int
    total: int
    brief_reason_fa: str
    raw_judge_output: str
    parse_success: bool


class VerifiableReasoningJudge:
    def __init__(self, judge_provider: BaseProvider, rubric_path: str = "rubrics/verifiable_reasoning.json"):
        self.provider = judge_provider
        with open(rubric_path, "r", encoding="utf-8") as f:
            self.rubric = json.load(f)

    def _build_criteria_list(self) -> str:
        lines = []
        for c in self.rubric["criteria"]:
            lines.append(f"- {c['id']} (۰ تا {c['scale_max']}): {c['question_fa']}")
        return "\n".join(lines)

    def _build_prompt(self, problem: str, cot: str, final_answer: str) -> str:
        template = self.rubric["judge_user_prompt_template_fa"]
        return template.format(
            problem=problem,
            cot=cot,
            final_answer=final_answer,
            criteria_list=self._build_criteria_list(),
        )

    def score(self, problem: str, cot: str, final_answer: str) -> VerifiableReasoningScore:
        system_prompt = self.rubric["judge_system_prompt_fa"]
        user_prompt = self._build_prompt(problem, cot, final_answer)

        result = self.provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        parsed = try_parse_json(result.text)

        if parsed is None:
            return VerifiableReasoningScore(
                step_support=0, conclusion_validity=0, no_logical_jump=0,
                answer_consistency=0, total=0,
                brief_reason_fa="خطا: خروجی داور قابل parse نبود.",
                raw_judge_output=result.text, parse_success=False,
            )

        step_support = int(parsed.get("step_support", 0))
        conclusion_validity = int(parsed.get("conclusion_validity", 0))
        no_logical_jump = int(parsed.get("no_logical_jump", 0))
        answer_consistency = int(parsed.get("answer_consistency", 0))
        total = step_support + conclusion_validity + no_logical_jump + answer_consistency

        return VerifiableReasoningScore(
            step_support=step_support,
            conclusion_validity=conclusion_validity,
            no_logical_jump=no_logical_jump,
            answer_consistency=answer_consistency,
            total=total,
            brief_reason_fa=parsed.get("brief_reason_fa", ""),
            raw_judge_output=result.text,
            parse_success=True,
        )
