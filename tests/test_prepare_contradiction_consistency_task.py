"""
Unit tests for scripts/prepare_contradiction_consistency_task.py.

Run: pytest tests/ -v
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import pytest

from prepare_contradiction_consistency_task import convert, OPTION_LETTERS, VALID_RELATIONS

RAW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw_sources", "contradiction_consistency_fa_raw.json"
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
    {"id": 1, "task_type": "contradiction_consistency", "subtype": "nli_relation",
     "premise_group": 1, "premise": "علی می‌دود.", "hypothesis": "علی ورزش می‌کند.",
     "gold_relation": "entailment"},
    {"id": 2, "task_type": "contradiction_consistency", "subtype": "nli_relation",
     "premise_group": 1, "premise": "علی می‌دود.", "hypothesis": "علی هرگز ورزش نمی‌کند.",
     "gold_relation": "contradiction"},
    {"id": 3, "task_type": "contradiction_consistency", "subtype": "nli_relation",
     "premise_group": 1, "premise": "علی می‌دود.", "hypothesis": "علی صبحانه می‌خورد.",
     "gold_relation": "neutral"},
    # نمونه‌ی ناقص (gold_relation نامعتبر) که باید رد شود
    {"id": 4, "task_type": "contradiction_consistency", "subtype": "nli_relation",
     "premise_group": 2, "premise": "چیزی.", "hypothesis": "چیز دیگری.",
     "gold_relation": "unknown_label"},
    # نمونه‌ی ناقص (hypothesis خالی) که باید رد شود
    {"id": 5, "task_type": "contradiction_consistency", "subtype": "nli_relation",
     "premise_group": 3, "premise": "چیزی.", "hypothesis": "",
     "gold_relation": "neutral"},
]


def test_full_dataset_converts_with_no_skips():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    assert len(task_samples) == len(raw_samples) == 60


def test_full_dataset_is_balanced_across_relations():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    from collections import Counter
    counts = Counter(s["extra"]["gold_relation"] for s in task_samples)
    assert counts == {"entailment": 20, "contradiction": 20, "neutral": 20}


def test_full_dataset_has_twenty_premise_groups_of_three():
    raw_samples = _load_raw()
    task_samples = _run_convert(raw_samples)
    from collections import Counter
    counts = Counter(s["extra"]["premise_group"] for s in task_samples)
    assert len(counts) == 20
    assert all(v == 3 for v in counts.values())


def test_skips_malformed_samples():
    task_samples = _run_convert(FAKE_RAW)
    # فقط ۳ نمونه‌ی معتبر (id=1,2,3)؛ id=4 (رابطه‌ی نامعتبر) و id=5 (hypothesis خالی) رد می‌شوند
    assert len(task_samples) == 3
    assert {s["extra"]["raw_id"] for s in task_samples} == {1, 2, 3}


def test_gold_answer_is_valid_option_letter():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["gold_answer"] in OPTION_LETTERS


def test_gold_relation_is_valid():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["extra"]["gold_relation"] in VALID_RELATIONS


def test_premise_and_hypothesis_present_in_problem_text():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert s["extra"]["premise"] in s["problem_fa"]
        assert s["extra"]["hypothesis"] in s["problem_fa"]


def test_all_three_relation_labels_present_in_every_problem():
    task_samples = _run_convert(FAKE_RAW)
    for s in task_samples:
        assert "استلزام" in s["problem_fa"]
        assert "تناقض" in s["problem_fa"]
        assert "خنثی" in s["problem_fa"]


def test_deterministic_given_same_seed():
    a = _run_convert(FAKE_RAW, seed=7)
    b = _run_convert(FAKE_RAW, seed=7)
    assert [s["gold_answer"] for s in a] == [s["gold_answer"] for s in b]


def test_sample_ids_are_unique_and_formatted():
    task_samples = _run_convert(FAKE_RAW)
    ids = [s["sample_id"] for s in task_samples]
    assert len(ids) == len(set(ids))
    assert all(sid.startswith("contradiction_consistency_") for sid in ids)


def test_task_sample_shape():
    task_samples = _run_convert(FAKE_RAW)
    s = task_samples[0]
    assert s["task_type"] == "contradiction_consistency"
    assert s["apply_persian_stability_rubric"] is True
    assert s["apply_verifiable_reasoning_rubric"] is False
    assert s["requires_cot_judging"] is False
    assert "پاسخ نهایی" in s["system_prompt_fa"]
