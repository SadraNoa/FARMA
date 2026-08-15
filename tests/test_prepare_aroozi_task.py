"""
Unit tests for scripts/prepare_aroozi_task.py (Aroozi multiple-choice
option generation).
Run: pytest tests/ -v
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from prepare_aroozi_task import build_meter_pool, convert
from src.utils.text_utils import normalize_fa_text

RAW_SAMPLES = [
    {
        "id": 1, "task_type": "aroozi", "subtype": "meter_identification",
        "verse": "بیت یک", "gold_meter_pattern": "پترن یک", "gold_meter_name": "وزن یک",
        "poet": "شاعر یک", "source": "منبع یک",
    },
    {
        "id": 2, "task_type": "aroozi", "subtype": "meter_identification",
        "verse": "بیت دو", "gold_meter_pattern": "پترن دو", "gold_meter_name": "وزن دو",
        "poet": "شاعر دو", "source": "منبع دو",
    },
    {
        "id": 3, "task_type": "aroozi", "subtype": "meter_identification",
        "verse": "بیت سه", "gold_meter_pattern": "پترن سه", "gold_meter_name": "وزن سه",
        "poet": "شاعر سه", "source": "منبع سه",
    },
    {
        "id": 4, "task_type": "aroozi", "subtype": "meter_identification",
        "verse": "بیت چهار", "gold_meter_pattern": "پترن چهار", "gold_meter_name": "وزن چهار",
        "poet": "شاعر چهار", "source": "منبع چهار",
    },
]


def test_build_meter_pool_collects_unique_meters():
    pool = build_meter_pool(RAW_SAMPLES)
    assert pool == {
        "وزن یک": "پترن یک",
        "وزن دو": "پترن دو",
        "وزن سه": "پترن سه",
        "وزن چهار": "پترن چهار",
    }


def _run_convert(tmp_path, num_options=4, seed=42):
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "task.jsonl"
    input_path.write_text(json.dumps(RAW_SAMPLES, ensure_ascii=False), encoding="utf-8")
    convert(str(input_path), str(output_path), num_options=num_options, seed=seed)
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(line) for line in lines]


def test_convert_produces_one_sample_per_input(tmp_path):
    samples = _run_convert(tmp_path)
    assert len(samples) == len(RAW_SAMPLES)


def test_convert_each_sample_has_correct_number_of_options(tmp_path):
    samples = _run_convert(tmp_path, num_options=4)
    for s in samples:
        assert len(s["extra"]["options"]) == 4


def test_convert_gold_answer_matches_correct_option(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        correct_letter = s["gold_answer"]
        option = s["extra"]["options"][correct_letter]
        assert option["meter_name"] == s["extra"]["gold_meter_name"]
        assert option["meter_pattern"] == s["extra"]["gold_meter_pattern"]


def test_convert_options_contain_no_duplicate_meters(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        names = [opt["meter_name"] for opt in s["extra"]["options"].values()]
        assert len(names) == len(set(names))


def test_convert_is_deterministic_given_same_seed(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    samples_a = _run_convert(tmp_path / "a", seed=7)
    # اجرای دوباره با همان seed روی مسیر دیگر باید گزینه‌های یکسانی تولید کند
    samples_b = _run_convert(tmp_path / "b", seed=7)
    for a, b in zip(samples_a, samples_b):
        assert a["extra"]["options"] == b["extra"]["options"]
        assert a["gold_answer"] == b["gold_answer"]


def test_exact_match_scoring_matches_bare_letter():
    # شبیه‌سازی مسیر امتیازدهی عمومی runner.score_sample برای gold_answer
    gold_answer = "ب"
    assert normalize_fa_text("ب") == normalize_fa_text(gold_answer)
    assert normalize_fa_text("ب.") == normalize_fa_text(gold_answer)


def test_convert_rejects_malformed_sample(tmp_path, capsys):
    bad_samples = RAW_SAMPLES + [{"id": 5, "task_type": "aroozi", "subtype": "meter_identification"}]
    input_path = tmp_path / "raw_bad.json"
    output_path = tmp_path / "task_bad.jsonl"
    input_path.write_text(json.dumps(bad_samples, ensure_ascii=False), encoding="utf-8")
    convert(str(input_path), str(output_path), num_options=4, seed=42)
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(RAW_SAMPLES)  # نمونه ناقص رد می‌شود
