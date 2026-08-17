"""
Rule-based checkers for the Persian Controllability task's rule-checkable
constraint types (paragraph_count, sentence_structure). Judge-checkable
types (lexical, register, dialect) are NOT handled here -- see
rubrics/persian_controllability.json and
src/judges/persian_controllability_judge.py.

Follows the same (passed, reason_fa) return convention as
src/utils/instruction_utils.py so the two can be read side by side.
"""

import re

from src.utils.text_utils import split_sentences

RULE_CHECKABLE_TYPES = {"paragraph_count", "sentence_structure"}


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def check_paragraph_count(text: str, value: str) -> tuple[bool, str]:
    paragraphs = _split_paragraphs(text)
    n = len(paragraphs)
    try:
        expected = int(value)
    except ValueError:
        return False, f"مقدار paragraph_count نامعتبر: {value}"
    passed = n == expected
    return passed, f"تعداد پاراگراف‌ها: {n} (هدف: دقیقاً {expected})"


# Heuristic only -- not a real parser. See docs/tasks/persian_controllability_task.md
# for known limitations of this check.
_KE_CLAUSE_MARKER = "که"


def check_sentence_structure(text: str, value: str) -> tuple[bool, str]:
    if value == "no_ke_clauses":
        passed = _KE_CLAUSE_MARKER not in text
        return passed, f"وجود «که»: {'خیر' if passed else 'بله'} (هدف: عدم استفاده)"

    if value == "verb_final":
        sentences = split_sentences(text)
        if not sentences:
            return False, "هیچ جمله‌ای یافت نشد"
        obviously_bad = 0
        for s in sentences:
            last_token = s.split()[-1] if s.split() else ""
            if last_token.isdigit() or last_token.endswith((",", "،")):
                obviously_bad += 1
        passed = obviously_bad == 0
        return passed, f"جملات با پایان مشکوک به غیرفعل: {obviously_bad} از {len(sentences)}"

    return False, f"مقدار sentence_structure ناشناخته: {value}"


_RULE_CHECKERS = {
    "paragraph_count": check_paragraph_count,
    "sentence_structure": check_sentence_structure,
}


def check_rule_constraint(text: str, constraint: dict) -> tuple[bool, str]:
    """Dispatches to the right rule-based checker based on constraint['type'].
    Only call this for constraint types in RULE_CHECKABLE_TYPES -- judge-checkable
    types should go through PersianControllabilityJudge instead."""
    ctype = constraint.get("type")
    checker = _RULE_CHECKERS.get(ctype)
    if checker is None:
        return False, f"نوع قید rule-based ناشناخته: {ctype}"
    return checker(text, constraint.get("value", ""))
