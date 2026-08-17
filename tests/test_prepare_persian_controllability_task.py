"""
Unit tests for scripts/prepare_persian_controllability_task.py.
Run: pytest tests/ -v
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from prepare_persian_controllability_task import convert
from src.pipeline.schemas import TaskSample

RAW_SAMPLES = [
    {
        "id": "pc_001", "prompt_fa": "سوال یک",
        "constraints": [{"type": "paragraph_count", "value": "2", "hard": True}],
        "difficulty": "easy",
    },
    {
        "id": "pc_002", "prompt_fa": "سوال دو",
        "constraints": [
            {"type": "lexical", "value": "pure_persian_only", "hard": True},
            {"type": "register", "value": "formal_official", "hard": False},
        ],
        "difficulty": "hard",
    },
    {"id": "bad_no_constraints", "prompt_fa": "سوال بد", "constraints": [], "difficulty": "easy"},
    {
        "id": "bad_unknown_type", "prompt_fa": "سوال بد دو",
        "constraints": [{"type": "not_a_real_type", "value": "x", "hard": False}],
        "difficulty": "easy",
    },
    {
        "id": "bad_bad_difficulty", "prompt_fa": "سوال بد سه",
        "constraints": [{"type": "paragraph_count", "value": "1", "hard": True}],
        "difficulty": "impossible",
    },
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
    assert ids == {"pc_001", "pc_002"}


def test_output_matches_task_sample_schema():
    results = _run_convert(RAW_SAMPLES)
    for r in results:
        sample = TaskSample(**r)
        assert sample.task_type == "persian_controllability"
        assert sample.gold_answer is None
        assert sample.apply_persian_controllability_rubric is True


def test_constraints_preserved_in_extra():
    results = _run_convert(RAW_SAMPLES)
    by_id = {r["sample_id"]: r for r in results}
    assert len(by_id["pc_001"]["extra"]["constraints"]) == 1
    assert len(by_id["pc_002"]["extra"]["constraints"]) == 2
    assert by_id["pc_002"]["extra"]["constraints"][0]["type"] == "lexical"


def test_difficulty_preserved():
    results = _run_convert(RAW_SAMPLES)
    by_id = {r["sample_id"]: r for r in results}
    assert by_id["pc_001"]["extra"]["difficulty"] == "easy"
    assert by_id["pc_002"]["extra"]["difficulty"] == "hard"


def test_rejects_empty_constraints():
    raw = [{"id": "x", "prompt_fa": "q", "constraints": [], "difficulty": "easy"}]
    assert len(_run_convert(raw)) == 0


def test_rejects_unknown_constraint_type():
    raw = [{
        "id": "x", "prompt_fa": "q",
        "constraints": [{"type": "nonexistent", "value": "v", "hard": False}],
        "difficulty": "easy",
    }]
    assert len(_run_convert(raw)) == 0


def test_no_verifiable_reasoning_or_stability_rubric():
    # This task doesn't use the two generic CoT rubrics -- responses are
    # typically short and single-purpose, not long reasoning chains.
    results = _run_convert(RAW_SAMPLES)
    for r in results:
        assert r["apply_verifiable_reasoning_rubric"] is False
        assert r["apply_persian_stability_rubric"] is False
