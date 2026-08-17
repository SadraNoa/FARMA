"""
Judge for the deep_brainstorm rubric: idea diversity, self-correction, and
cultural grounding in open-ended brainstorming chains-of-thought. This is
deliberately structured identically to VerifiableReasoningJudge in
rubric_judge.py -- same single-JSON-call pattern, same file layout -- so it
plugs into runner.py the same way.

This rubric is a companion to verifiable_reasoning (already listed under
deep_brainstorm in its applies_to), not a replacement -- logical-quality
dimensions (step_support, no_logical_jump, answer_consistency) stay with
verifiable_reasoning; this covers what that rubric doesn't.
"""

import json
from dataclasses import dataclass

from src.providers.base import BaseProvider
from src.utils.text_utils import try_parse_json


@dataclass
class DeepBrainstormScore:
    idea_diversity: int
    self_correction: int
    cultural_grounding: int
    total: int
    brief_reason_fa: str
    raw_judge_output: str
    parse_success: bool


class DeepBrainstormJudge:
    def __init__(self, judge_provider: BaseProvider, rubric_path: str = "rubrics/deep_brainstorm.json"):
        self.provider = judge_provider
        with open(rubric_path, "r", encoding="utf-8") as f:
            self.rubric = json.load(f)

    def _build_criteria_list(self) -> str:
        lines = []
        for c in self.rubric["criteria"]:
            lines.append(f"- {c['id']} (۰ تا {c['scale_max']}): {c['question_fa']}")
        return "\n".join(lines)

    def _build_prompt(self, problem: str, cot: str) -> str:
        template = self.rubric["judge_user_prompt_template_fa"]
        return template.format(
            problem=problem,
            cot=cot,
            criteria_list=self._build_criteria_list(),
        )

    def score(self, problem: str, cot: str) -> DeepBrainstormScore:
        system_prompt = self.rubric["judge_system_prompt_fa"]
        user_prompt = self._build_prompt(problem, cot)

        result = self.provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        parsed = try_parse_json(result.text)

        if parsed is None:
            return DeepBrainstormScore(
                idea_diversity=0, self_correction=0, cultural_grounding=0, total=0,
                brief_reason_fa="خطا: خروجی داور قابل parse نبود.",
                raw_judge_output=result.text, parse_success=False,
            )

        idea_diversity = int(parsed.get("idea_diversity", 0))
        self_correction = int(parsed.get("self_correction", 0))
        cultural_grounding = int(parsed.get("cultural_grounding", 0))
        total = idea_diversity + self_correction + cultural_grounding

        return DeepBrainstormScore(
            idea_diversity=idea_diversity,
            self_correction=self_correction,
            cultural_grounding=cultural_grounding,
            total=total,
            brief_reason_fa=parsed.get("brief_reason_fa", ""),
            raw_judge_output=result.text,
            parse_success=True,
        )
