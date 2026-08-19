"""
Unit tests for scripts/prepare_script_disambiguation_task.py
(Script Disambiguation same-word-only option generation).
Run: pytest tests/ -v
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from prepare_script_disambiguation_task import build_word_pool, convert
from src.utils.text_utils import normalize_fa_text

RAW_SAMPLES = [
    {
        "id": 1, "task_type": "script_disambiguation", "subtype": "homograph_disambiguation",
        "sentence": "جمله یک با کلمه الف.", "ambiguous_word": "الف",
        "context_hint": "راهنما یک", "gold_reading": "اَلَف", "gold_meaning": "معنی یک کامل",
        "manual_distractors": [{"reading": "اِلِف", "meaning": "معنی دو"}],
        "poet_or_source": None,
    },
    {
        "id": 2, "task_type": "script_disambiguation", "subtype": "homograph_disambiguation",
        "sentence": "جمله دو با کلمه الف.", "ambiguous_word": "الف",
        "context_hint": "راهنما دو", "gold_reading": "اِلِف", "gold_meaning": "معنی دو کوتاه",
        "manual_distractors": [{"reading": "اَلَف", "meaning": "معنی یک"}],
        "poet_or_source": None,
    },
    {
        "id": 3, "task_type": "script_disambiguation", "subtype": "homograph_disambiguation",
        "sentence": "جمله سه با کلمه ب.", "ambiguous_word": "ب",
        "context_hint": "راهنما سه", "gold_reading": "بَ", "gold_meaning": "معنی سه",
        "manual_distractors": [
            {"reading": "بِ", "meaning": "معنی چهار"},
            {"reading": "بُ", "meaning": "معنی پنج"},
        ],
        "poet_or_source": None,
    },
    {
        # کلمه‌ای بدون هیچ manual_distractor و بدون تکرار در جای دیگر — باید skip شود
        "id": 4, "task_type": "script_disambiguation", "subtype": "homograph_disambiguation",
        "sentence": "جمله چهار با کلمه تنها.", "ambiguous_word": "تنها",
        "context_hint": "راهنما چهار", "gold_reading": "تَنها", "gold_meaning": "معنی شش",
        "manual_distractors": [],
        "poet_or_source": None,
    },
]


def test_build_word_pool_uses_reading_as_key_not_reading_meaning_pair():
    pool = build_word_pool(RAW_SAMPLES)
    assert set(pool["الف"].keys()) == {"اَلَف", "اِلِف"}
    # معنیِ کانونیِ هر تلفظ باید همان معنیِ نوشته‌شده در جایی باشد که آن
    # تلفظ خودِ gold_reading بوده (نه عبارتِ کوتاه‌ترِ آمده در distractor)
    assert pool["الف"]["اَلَف"] == "معنی یک کامل"
    assert pool["الف"]["اِلِف"] == "معنی دو کوتاه"


def test_build_word_pool_is_scoped_per_word():
    pool = build_word_pool(RAW_SAMPLES)
    assert "بَ" not in pool["الف"]
    assert set(pool["ب"].keys()) == {"بَ", "بِ", "بُ"}


def _run_convert(tmp_path, max_options=4, min_options=2, seed=42):
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "task.jsonl"
    input_path.write_text(json.dumps(RAW_SAMPLES, ensure_ascii=False), encoding="utf-8")
    convert(str(input_path), str(output_path), max_options=max_options, min_options=min_options, seed=seed)
    text = output_path.read_text(encoding="utf-8").strip()
    lines = text.split("\n") if text else []
    return [json.loads(line) for line in lines]


def test_convert_skips_samples_with_no_known_alternate_reading(tmp_path):
    # نمونه id=4 («تنها») باید حذف شود چون هیچ خوانش جایگزینی برایش وجود ندارد
    samples = _run_convert(tmp_path)
    ids = {s["extra"]["raw_id"] for s in samples}
    assert ids == {1, 2, 3}


def test_convert_variable_option_counts(tmp_path):
    samples = _run_convert(tmp_path)
    by_id = {s["extra"]["raw_id"]: s for s in samples}
    assert len(by_id[1]["extra"]["options"]) == 2
    assert len(by_id[3]["extra"]["options"]) == 3


def test_convert_no_duplicate_readings_within_one_question(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        readings = [opt["reading"] for opt in s["extra"]["options"].values()]
        assert len(readings) == len(set(readings))


def test_convert_gold_answer_matches_correct_option(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        correct_letter = s["gold_answer"]
        option = s["extra"]["options"][correct_letter]
        assert option["reading"] == s["extra"]["gold_reading"]


def test_convert_never_pulls_distractor_from_another_word(tmp_path):
    samples = _run_convert(tmp_path)
    by_id = {s["extra"]["raw_id"]: s for s in samples}
    readings_for_alef_q = {opt["reading"] for opt in by_id[1]["extra"]["options"].values()}
    assert readings_for_alef_q.issubset({"اَلَف", "اِلِف"})


def test_convert_is_deterministic_given_same_seed(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    samples_a = _run_convert(tmp_path / "a", seed=7)
    samples_b = _run_convert(tmp_path / "b", seed=7)
    for a, b in zip(samples_a, samples_b):
        assert a["extra"]["options"] == b["extra"]["options"]
        assert a["gold_answer"] == b["gold_answer"]


def test_exact_match_scoring_matches_bare_letter():
    gold_answer = "ب"
    assert normalize_fa_text("ب") == normalize_fa_text(gold_answer)
    assert normalize_fa_text("ب.") == normalize_fa_text(gold_answer)


def test_convert_rejects_malformed_sample(tmp_path):
    bad_samples = RAW_SAMPLES + [{"id": 5, "task_type": "script_disambiguation", "subtype": "homograph_disambiguation"}]
    input_path = tmp_path / "raw_bad.json"
    output_path = tmp_path / "task_bad.jsonl"
    input_path.write_text(json.dumps(bad_samples, ensure_ascii=False), encoding="utf-8")
    convert(str(input_path), str(output_path), max_options=4, min_options=2, seed=42)
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    # id=4 (تنها) skip می‌شود چون خوانش جایگزین ندارد، id=5 هم به‌خاطر ناقص بودن
    assert len(lines) == 3
