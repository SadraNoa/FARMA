"""
Unit tests for scripts/prepare_paraphrase_robustness_task.py.

Run: pytest tests/ -v
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import pytest

from prepare_paraphrase_robustness_task import convert, OPTION_LETTERS, VALID_SUBTYPES

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw_sources", "paraphrase_robustness_fa_raw.json"
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
    {"id": 1, "task_type": "paraphrase_robustness", "subtype": "paraphrase_variant",
     "family_id": 1, "paraphrase_index": 1, "passage": "متن نمونه با عدد ۱۰.",
     "question": "عدد نمونه چند است؟", "correct_answer": "۱۰", "foils": ["۵", "۱۵", "۲۰"]},
    {"id": 2, "task_type": "paraphrase_robustness", "subtype": "paraphrase_variant",
     "family_id": 1, "paraphrase_index": 2, "passage": "متن نمونه با عدد ۱۰.",
     "question": "مقدار عدد در متن نمونه چقدر است؟", "correct_answer": "۱۰", "foils": ["۵", "۱۵", "۲۰"]},
    # نمونه‌ی ناقص (تعداد foils اشتباه) که باید رد شود
    {"id": 3, "task_type": "paraphrase_robustness", "subtype": "paraphrase_variant",
     "family_id": 2, "paraphrase_index": 1, "passage": "چیزی.", "question": "چیزی؟",
     "correct_answer": "الف", "foils": ["ب", "ج"]},
    # نمونه‌ی ناقص (correct_answer در foils تکرار شده) که باید رد شود
    {"id": 4, "task_type": "paraphrase_robustness", "subtype": "paraphrase_variant",
     "family_id": 3, "paraphrase_index": 1, "passage": "چیزی.", "question": "چیزی؟",
     "correct_answer": "الف", "foils": ["الف", "ب", "ج"]},
]


def test_full_dataset_converts_with_no_skips():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    assert len(task_samples) == len(raw_samples) == 60


def test_full_dataset_has_fifteen_families_of_four():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    from collections import Counter
    counts = Counter(s["extra"]["family_id"] for s in task_samples)
    assert len(counts) == 15
    assert all(v == 4 for v in counts.values())


def test_skips_malformed_samples():
    task_samples = _run_convert(FAKE_RAW)
    assert len(task_samples) == 2
    assert {s["extra"]["raw_id"] for s in task_samples} == {1, 2}


def test_gold_answer_is_valid_option_letter():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["gold_answer"] in OPTION_LETTERS


def test_gold_answer_points_to_correct_answer_text():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        gold_letter = s["gold_answer"]
        expected_line = f"{gold_letter}) {s['extra']['correct_answer']}"
        assert expected_line in s["problem_fa"]


def test_all_options_present_in_problem_text():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["extra"]["correct_answer"] in s["problem_fa"]
        for foil in s["extra"]["foils"]:
            assert foil in s["problem_fa"]


def test_same_family_shares_passage_and_options_but_different_question():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    family_1 = [s for s in task_samples if s["extra"]["family_id"] == 1]
    assert len(family_1) == 4
    passages = {s["extra"]["passage"] for s in family_1}
    assert len(passages) == 1  # همه‌ی بازنویسی‌ها یک متن مشترک دارند
    correct_answers = {s["extra"]["correct_answer"] for s in family_1}
    assert len(correct_answers) == 1  # پاسخ درست باید در کل خانواده ثابت بماند
    questions = {s["extra"]["question"] for s in family_1}
    assert len(questions) == 4  # ولی متن سؤال باید در هر بازنویسی متفاوت باشد


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
    assert all(sid.startswith("paraphrase_robustness_") for sid in ids)


def test_task_sample_shape():
    task_samples = _run_convert(FAKE_RAW)
    s = task_samples[0]
    assert s["task_type"] == "paraphrase_robustness"
    assert s["apply_persian_stability_rubric"] is True
    assert s["apply_verifiable_reasoning_rubric"] is False
    assert s["requires_cot_judging"] is False
    assert "پاسخ نهایی" in s["system_prompt_fa"]
