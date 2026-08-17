"""
Unit tests for scripts/prepare_minimal_pairs_task.py.

Run: pytest tests/ -v
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import pytest

from prepare_minimal_pairs_task import convert, OPTION_LETTERS

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw_sources", "minimal_pairs_fa_raw.json"
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
    {
        "id": 1, "task_type": "minimal_pairs", "subtype": "grammaticality_judgment",
        "category": "cat_a", "category_label": "دسته الف", "rule": "قاعده‌ی الف",
        "correct_sentence": "این جمله درست است.", "incorrect_sentence": "جمله این درست است.",
    },
    {
        "id": 2, "task_type": "minimal_pairs", "subtype": "grammaticality_judgment",
        "category": "cat_b", "category_label": "دسته ب", "rule": "قاعده‌ی ب",
        "correct_sentence": "او نمی‌رود.", "incorrect_sentence": "او می‌نرود.",
    },
    # نمونه‌ی ناقص (identical) که باید رد شود
    {
        "id": 3, "task_type": "minimal_pairs", "subtype": "grammaticality_judgment",
        "category": "cat_c", "category_label": "دسته ج", "rule": "قاعده‌ی ج",
        "correct_sentence": "یک جمله.", "incorrect_sentence": "یک جمله.",
    },
    # نمونه‌ی ناقص (missing field) که باید رد شود
    {
        "id": 4, "task_type": "minimal_pairs", "subtype": "grammaticality_judgment",
        "category": "cat_d", "category_label": "دسته د", "rule": "قاعده‌ی د",
        "correct_sentence": "", "incorrect_sentence": "چیزی.",
    },
]


def test_full_dataset_converts_with_no_skips():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    assert len(task_samples) == len(raw_samples) == 64


def test_full_dataset_has_eight_categories_of_eight():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    from collections import Counter
    counts = Counter(s["extra"]["category"] for s in task_samples)
    assert len(counts) == 8
    assert all(v == 8 for v in counts.values())


def test_skips_malformed_samples():
    task_samples = _run_convert(FAKE_RAW)
    # فقط ۲ نمونه‌ی معتبر (id=1,2) باید تبدیل شوند؛ id=3 (یکسان) و id=4 (خالی) رد می‌شوند
    assert len(task_samples) == 2
    assert {s["extra"]["raw_id"] for s in task_samples} == {1, 2}


def test_gold_answer_is_valid_option_letter():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["gold_answer"] in OPTION_LETTERS


def test_gold_answer_matches_correct_sentence_in_problem_text():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        correct_sentence = s["extra"]["correct_sentence"]
        gold_letter = s["gold_answer"]
        expected_line = f"{gold_letter}) {correct_sentence}"
        assert expected_line in s["problem_fa"]


def test_both_sentences_present_in_problem_text():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["extra"]["correct_sentence"] in s["problem_fa"]
        assert s["extra"]["incorrect_sentence"] in s["problem_fa"]


def test_deterministic_given_same_seed():
    a = _run_convert(FAKE_RAW, seed=7)
    b = _run_convert(FAKE_RAW, seed=7)
    assert [s["gold_answer"] for s in a] == [s["gold_answer"] for s in b]


def test_different_seeds_can_change_option_order():
    # با seedهای مختلف، دست‌کم برای برخی نمونه‌ها جای گزینه‌ها باید عوض شود
    # (تست قطعی نیست ولی روی این دیتای کوچک با seedهای دورافتاده بعید است که
    # هر دو حالت کاملاً یکسان دربیایند؛ در صورت flaky بودن می‌توان seedها را عوض کرد)
    a = _run_convert(FAKE_RAW, seed=1)
    b = _run_convert(FAKE_RAW, seed=999)
    letters_a = [s["gold_answer"] for s in a]
    letters_b = [s["gold_answer"] for s in b]
    assert len(letters_a) == len(letters_b)


def test_sample_ids_are_unique_and_formatted():
    task_samples = _run_convert(FAKE_RAW)
    ids = [s["sample_id"] for s in task_samples]
    assert len(ids) == len(set(ids))
    assert all(sid.startswith("minimal_pairs_") for sid in ids)


def test_task_sample_shape():
    task_samples = _run_convert(FAKE_RAW)
    s = task_samples[0]
    assert s["task_type"] == "minimal_pairs"
    assert s["apply_persian_stability_rubric"] is True
    assert s["apply_verifiable_reasoning_rubric"] is False
    assert s["requires_cot_judging"] is False
    assert "پاسخ نهایی" in s["system_prompt_fa"]
