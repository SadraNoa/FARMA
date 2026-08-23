"""
Unit tests for scripts/prepare_multi_constraint_task.py and the
rule_constraints scoring branch it feeds into runner.py.

Unlike test_prepare_deep_brainstorm_task.py / test_prepare_persian_
controllability_task.py, this file deliberately never imports
src.pipeline.schemas / TaskSample, so it does NOT require pydantic and can
run in fully offline sandboxes where pydantic isn't installable — matching
the style of every Phase 2/3 test file (test_prepare_script_
disambiguation_task.py, test_prepare_ieham_task.py, etc.).

Run: pytest tests/ -v
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from prepare_multi_constraint_task import convert
from src.utils.instruction_utils import check_constraint

RAW_SAMPLES = [
    {
        "id": 1, "category": "length_keyword", "topic": "موضوع یک",
        "question": "سوال یک با دو محدودیت.",
        "constraints": [
            {"type": "sentence_count", "mode": "exact", "value": 3},
            {"type": "must_include", "word": "کتاب", "count": 1},
        ],
    },
    {
        "id": 2, "category": "triple_combo", "topic": "موضوع دو",
        "question": "سوال دو با سه محدودیت.",
        "constraints": [
            {"type": "paragraph_count", "mode": "exact", "value": 2},
            {"type": "must_exclude", "word": "متأسفانه"},
            {"type": "must_start_with", "word": "امروز"},
        ],
    },
    {
        # فقط یک قید — باید skip شود چون تسک چندمحدودیتی حداقل ۲ قید لازم دارد
        "id": 3, "category": "length_keyword", "topic": "موضوع سه",
        "question": "سوال ناقص با فقط یک محدودیت.",
        "constraints": [{"type": "sentence_count", "mode": "exact", "value": 2}],
    },
    {
        # category نامعتبر — باید skip شود
        "id": 4, "category": "unknown_category", "topic": "موضوع چهار",
        "question": "سوال با category نامعتبر.",
        "constraints": [
            {"type": "must_include", "word": "الف", "count": 1},
            {"type": "must_exclude", "word": "ب"},
        ],
    },
    {
        # فیلد question ناقص — باید skip شود
        "id": 5, "category": "length_position", "topic": "موضوع پنج",
        "constraints": [
            {"type": "must_include", "word": "الف", "count": 1},
            {"type": "must_exclude", "word": "ب"},
        ],
    },
]


def _run_convert(tmp_path):
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "task.jsonl"
    input_path.write_text(json.dumps(RAW_SAMPLES, ensure_ascii=False), encoding="utf-8")
    convert(str(input_path), str(output_path))
    text = output_path.read_text(encoding="utf-8").strip()
    lines = text.split("\n") if text else []
    return [json.loads(line) for line in lines]


def test_convert_skips_underfilled_and_malformed_samples(tmp_path):
    samples = _run_convert(tmp_path)
    ids = {s["extra"]["raw_id"] for s in samples}
    assert ids == {1, 2}


def test_convert_carries_all_constraints_into_rule_constraints_key(tmp_path):
    samples = _run_convert(tmp_path)
    by_id = {s["extra"]["raw_id"]: s for s in samples}
    assert by_id[1]["extra"]["rule_constraints"] == RAW_SAMPLES[0]["constraints"]
    assert by_id[2]["extra"]["rule_constraints"] == RAW_SAMPLES[1]["constraints"]


def test_convert_sets_no_gold_answer(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        assert s["gold_answer"] is None


def test_convert_uses_full_output_extraction_regex(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        assert s["answer_extraction_regex"] == r"(?s)(.*)"


def test_convert_num_constraints_matches_list_length(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        assert s["extra"]["num_constraints"] == len(s["extra"]["rule_constraints"])


# --- Feasibility & scoring-branch tests (mirrors runner.py's rule_constraints logic) ---

def _score_rule_constraints(constraints: list[dict], raw_output: str) -> tuple[int, str]:
    """Reimplements the exact scoring logic added to runner.py's
    score_sample() for the rule_constraints branch, without needing
    TaskSample/pydantic, so it can be exercised in this offline sandbox."""
    per_constraint_reasons = []
    all_passed = True
    for c in constraints:
        passed, reason_fa = check_constraint(raw_output, c)
        per_constraint_reasons.append(reason_fa)
        if not passed:
            all_passed = False
    return int(all_passed), " | ".join(per_constraint_reasons)


def test_no_raw_sample_has_contradictory_include_exclude_words():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_sources", "multi_constraint_fa_raw.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        cs = item["constraints"]
        include_words = {c["word"] for c in cs if c["type"] in ("must_include", "must_include_exact_count")}
        exclude_words = {c["word"] for c in cs if c["type"] == "must_exclude"}
        assert not (include_words & exclude_words), f"id={item['id']} has overlapping include/exclude words"


def test_no_raw_sample_has_more_than_one_length_constraint():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_sources", "multi_constraint_fa_raw.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    length_types = {"sentence_count", "word_count", "paragraph_count"}
    for item in data:
        n = sum(1 for c in item["constraints"] if c["type"] in length_types)
        assert n <= 1, f"id={item['id']} has {n} length constraints"


def test_fully_compliant_response_scores_correct():
    constraints = [
        {"type": "paragraph_count", "mode": "exact", "value": 2},
        {"type": "must_include", "word": "کتاب", "count": 1},
        {"type": "must_start_with", "word": "واقعیت"},
    ]
    response = (
        "واقعیت این است که موضوع بسیار مهمی است. کتاب‌های زیادی درباره‌ی آن نوشته شده‌اند.\n\n"
        "این پاراگراف دوم است و ادامه‌ی بحث را دنبال می‌کند."
    )
    correctness, _ = _score_rule_constraints(constraints, response)
    assert correctness == 1


def test_response_violating_one_of_several_constraints_scores_incorrect():
    constraints = [
        {"type": "paragraph_count", "mode": "exact", "value": 2},
        {"type": "must_include", "word": "کتاب", "count": 1},
        {"type": "must_start_with", "word": "واقعیت"},
    ]
    # همان پاسخ قبلی ولی با کلمه‌ی شروعِ غلط
    response = (
        "در حقیقت این است که موضوع بسیار مهمی است. کتاب‌های زیادی درباره‌ی آن نوشته شده‌اند.\n\n"
        "این پاراگراف دوم است و ادامه‌ی بحث را دنبال می‌کند."
    )
    correctness, notes = _score_rule_constraints(constraints, response)
    assert correctness == 0
    assert "واقعیت" in notes  # هنوز باید در notes گزارش بشه که این قید رد شده


def test_response_violating_all_constraints_scores_incorrect():
    constraints = [
        {"type": "must_include", "word": "کتاب", "count": 1},
        {"type": "must_exclude", "word": "متأسفانه"},
    ]
    response = "متأسفانه امروز هیچ صحبتی درباره‌ی آن موضوع نمی‌کنیم."
    correctness, _ = _score_rule_constraints(constraints, response)
    assert correctness == 0
