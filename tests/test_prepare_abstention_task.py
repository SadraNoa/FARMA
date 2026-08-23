"""
Unit tests for scripts/prepare_abstention_task.py.

Run: pytest tests/ -v
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import pytest

from prepare_abstention_task import (
    convert,
    OPTION_LETTERS,
    INSUFFICIENT_INFO_TEXT,
    FALSE_PREMISE_TEXT,
    VALID_SUBTYPES,
)

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw_sources", "abstention_fa_raw.json"
)


def _load_raw():
    with open(RAW_PATH, encoding="utf-8") as f:
        return json.load(f)


def _run_convert(raw_samples, seed=42):
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "raw.json")
        out_path = os.path.join(tmp, "out.jsonl")
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(raw_samples, f, ensure_ascii=False)
        convert(in_path, out_path, seed=seed)
        with open(out_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]


FAKE_RAW = [
    {"id": 1, "task_type": "abstention", "subtype": "answerable", "family_id": 1,
     "passage": "کتابخانه از ساعت ۹ باز است.", "question": "کتابخانه از چه ساعتی باز است؟",
     "real_answer": "۹", "foil_answer": "۸"},
    {"id": 2, "task_type": "abstention", "subtype": "insufficient_info", "family_id": 1,
     "passage": "کتابخانه از ساعت ۹ باز است.", "question": "کتابخانه چند طبقه دارد؟",
     "real_answer": "۹", "foil_answer": "۸"},
    {"id": 3, "task_type": "abstention", "subtype": "false_premise", "family_id": 1,
     "passage": "کتابخانه از ساعت ۹ باز است.", "question": "چرا کتابخانه ساعت ۷ باز می‌شود؟",
     "real_answer": "۹", "foil_answer": "۸"},
    # نمونه‌ی ناقص (real_answer == foil_answer) که باید رد شود
    {"id": 4, "task_type": "abstention", "subtype": "answerable", "family_id": 2,
     "passage": "چیزی.", "question": "چیزی؟", "real_answer": "یک", "foil_answer": "یک"},
    # نمونه‌ی ناقص (subtype نامعتبر) که باید رد شود
    {"id": 5, "task_type": "abstention", "subtype": "unknown_type", "family_id": 3,
     "passage": "چیزی.", "question": "چیزی؟", "real_answer": "یک", "foil_answer": "دو"},
]


def test_full_dataset_converts_with_no_skips():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    assert len(task_samples) == len(raw_samples) == 54


def test_full_dataset_is_balanced_across_subtypes():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    from collections import Counter
    counts = Counter(s["extra"]["subtype"] for s in task_samples)
    assert counts == {"answerable": 18, "insufficient_info": 18, "false_premise": 18}


def test_full_dataset_has_eighteen_families_of_three():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    from collections import Counter
    counts = Counter(s["extra"]["family_id"] for s in task_samples)
    assert len(counts) == 18
    assert all(v == 3 for v in counts.values())


def test_skips_malformed_samples():
    task_samples = _run_convert(FAKE_RAW)
    assert len(task_samples) == 3
    assert {s["extra"]["raw_id"] for s in task_samples} == {1, 2, 3}


def test_gold_answer_is_valid_option_letter():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["gold_answer"] in OPTION_LETTERS


def test_all_four_options_present_in_every_problem():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["extra"]["real_answer"] in s["problem_fa"]
        assert s["extra"]["foil_answer"] in s["problem_fa"]
        assert INSUFFICIENT_INFO_TEXT in s["problem_fa"]
        assert FALSE_PREMISE_TEXT in s["problem_fa"]


def test_answerable_gold_points_to_real_answer():
    task_samples = _run_convert(FAKE_RAW)
    s = next(x for x in task_samples if x["extra"]["subtype"] == "answerable")
    gold_letter = s["gold_answer"]
    expected_line = f"{gold_letter}) {s['extra']['real_answer']}"
    assert expected_line in s["problem_fa"]


def test_insufficient_info_gold_points_to_fixed_text():
    task_samples = _run_convert(FAKE_RAW)
    s = next(x for x in task_samples if x["extra"]["subtype"] == "insufficient_info")
    gold_letter = s["gold_answer"]
    expected_line = f"{gold_letter}) {INSUFFICIENT_INFO_TEXT}"
    assert expected_line in s["problem_fa"]


def test_false_premise_gold_points_to_fixed_text():
    task_samples = _run_convert(FAKE_RAW)
    s = next(x for x in task_samples if x["extra"]["subtype"] == "false_premise")
    gold_letter = s["gold_answer"]
    expected_line = f"{gold_letter}) {FALSE_PREMISE_TEXT}"
    assert expected_line in s["problem_fa"]


def test_deterministic_given_same_seed():
    a = _run_convert(FAKE_RAW, seed=7)
    b = _run_convert(FAKE_RAW, seed=7)
    assert [s["gold_answer"] for s in a] == [s["gold_answer"] for s in b]


def test_subtype_is_valid():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["extra"]["subtype"] in VALID_SUBTYPES


def test_sample_ids_are_unique_and_formatted():
    task_samples = _run_convert(FAKE_RAW)
    ids = [s["sample_id"] for s in task_samples]
    assert len(ids) == len(set(ids))
    assert all(sid.startswith("abstention_") for sid in ids)


def test_task_sample_shape():
    task_samples = _run_convert(FAKE_RAW)
    s = task_samples[0]
    assert s["task_type"] == "abstention"
    assert s["apply_persian_stability_rubric"] is True
    assert s["apply_verifiable_reasoning_rubric"] is False
    assert s["requires_cot_judging"] is False
    assert "پاسخ نهایی" in s["system_prompt_fa"]
