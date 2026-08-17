"""
Judge for the persian_controllability rubric. Unlike the other judges in
this codebase, this one builds its prompt DYNAMICALLY per sample: only the
judge-checkable constraints actually present on that sample (lexical,
register, dialect) are included in the prompt, since which constraints
apply varies per item. Semantic fidelity is always scored (independent of
which constraints are present).

Rule-checkable constraints (paragraph_count, sentence_structure) are NOT
handled here -- see src/utils/controllability_utils.py, called directly
from runner.py.
"""

import json
from dataclasses import dataclass

from src.providers.base import BaseProvider
from src.utils.text_utils import try_parse_json


@dataclass
class ConstraintJudgeResult:
    index: int
    type: str
    satisfied: bool
    arabic_origin_words: list[str]


@dataclass
class PersianControllabilityScore:
    constraint_results: list[ConstraintJudgeResult]
    semantic_fidelity: int
    brief_reason_fa: str
    raw_judge_output: str
    parse_success: bool


class PersianControllabilityJudge:
    def __init__(self, judge_provider: BaseProvider, rubric_path: str = "rubrics/persian_controllability.json"):
        self.provider = judge_provider
        with open(rubric_path, "r", encoding="utf-8") as f:
            self.rubric = json.load(f)
        self.judge_checkable_types = set(self.rubric["judge_checkable_types"])

    def judge_checkable_constraints(self, constraints: list[dict]) -> list[dict]:
        """Filter a sample's full constraint list down to only the ones
        this judge is responsible for."""
        return [c for c in constraints if c.get("type") in self.judge_checkable_types]

    def _build_constraint_list(self, constraints: list[dict]) -> str:
        lines = []
        templates = self.rubric["constraint_prompts_fa"]
        for i, c in enumerate(constraints):
            template = templates.get(c["type"], "قید ناشناخته: {value}")
            lines.append(f"{i}. [{c['type']}] " + template.format(value=c.get("value", "")))
        return "\n".join(lines)

    def score(self, prompt: str, response: str, constraints: list[dict]) -> PersianControllabilityScore | None:
        """
        constraints should already be filtered to judge_checkable_constraints()
        by the caller. Returns None if there is nothing for this judge to do
        (no judge-checkable constraints on this sample) -- caller should still
        call this for semantic_fidelity even if constraints is empty, since
        semantic_fidelity is scored independent of constraint presence.
        """
        system_prompt = self.rubric["judge_system_prompt_fa"]
        constraint_list_text = (
            self._build_constraint_list(constraints) if constraints else "(هیچ قید judge-checkable‌ای در این نمونه نیست)"
        )
        user_prompt = self.rubric["judge_user_prompt_template_fa"].format(
            prompt=prompt, response=response, constraint_list=constraint_list_text
        )

        result = self.provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        parsed = try_parse_json(result.text)

        if parsed is None:
            return PersianControllabilityScore(
                constraint_results=[],
                semantic_fidelity=0,
                brief_reason_fa="خطا: خروجی داور قابل parse نبود.",
                raw_judge_output=result.text,
                parse_success=False,
            )

        constraint_results = [
            ConstraintJudgeResult(
                index=int(cr.get("index", i)),
                type=cr.get("type", ""),
                satisfied=bool(cr.get("satisfied", False)),
                arabic_origin_words=list(cr.get("arabic_origin_words", [])),
            )
            for i, cr in enumerate(parsed.get("constraint_results", []))
        ]

        return PersianControllabilityScore(
            constraint_results=constraint_results,
            semantic_fidelity=int(parsed.get("semantic_fidelity", 0)),
            brief_reason_fa=parsed.get("brief_reason_fa", ""),
            raw_judge_output=result.text,
            parse_success=True,
        )
