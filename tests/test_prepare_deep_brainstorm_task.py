"""
Unit tests for scripts/prepare_deep_brainstorm_task.py.
Run: pytest tests/ -v
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from prepare_deep_brainstorm_task import convert
from src.pipeline.schemas import TaskSample

RAW_SAMPLES = [
    {"id": "db_001", "prompt_fa": "سوال یک", "category": "policy_or_social"},
    {"id": "db_002", "prompt_fa": "سوال دو", "category": "creative_ideation"},
    {"id": "bad_no_category", "prompt_fa": "سوال بد"},  # missing category -> skipped
    {"id": "bad_no_prompt", "category": "policy_or_social"},  # missing prompt -> skipped
]


def _run_convert(raw_samples):
    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "raw.json")
        output_path = os.path.join(tmp, "out.jsonl")
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(raw_samples, f, ensure_ascii=False)
        convert(input_path, output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def test_skips_malformed_samples():
    results = _run_convert(RAW_SAMPLES)
    assert len(results) == 2
    ids = {r["sample_id"] for r in results}
    assert ids == {"db_001", "db_002"}


def test_output_matches_task_sample_schema():
    results = _run_convert(RAW_SAMPLES)
    for r in results:
        sample = TaskSample(**r)  # raises if shape is wrong
        assert sample.task_type == "deep_brainstorm"
        assert sample.gold_answer is None


def test_all_three_rubrics_enabled():
    results = _run_convert(RAW_SAMPLES)
    for r in results:
        assert r["apply_verifiable_reasoning_rubric"] is True
        assert r["apply_persian_stability_rubric"] is True
        assert r["apply_deep_brainstorm_rubric"] is True


def test_category_preserved_in_extra():
    results = _run_convert(RAW_SAMPLES)
    by_id = {r["sample_id"]: r for r in results}
    assert by_id["db_001"]["extra"]["category"] == "policy_or_social"
    assert by_id["db_002"]["extra"]["category"] == "creative_ideation"


def test_rejects_unknown_category():
    raw = [{"id": "db_x", "prompt_fa": "سوال", "category": "not_a_real_category"}]
    results = _run_convert(raw)
    assert len(results) == 0


def test_sample_ids_are_unique():
    results = _run_convert(RAW_SAMPLES)
    ids = [r["sample_id"] for r in results]
    assert len(ids) == len(set(ids))


def test_answer_extraction_regex_present():
    results = _run_convert(RAW_SAMPLES)
    for r in results:
        assert r["answer_extraction_regex"] == r"پاسخ نهایی:\s*(.+)"
