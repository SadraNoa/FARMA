"""
Unit tests for scripts/prepare_proverbs_task.py (Proverbs mixed
manual+pool option generation).
Run: pytest tests/ -v
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from prepare_proverbs_task import build_pool, convert
from src.utils.text_utils import normalize_fa_text

RAW_SAMPLES = [
    {
        "id": 1, "task_type": "proverb", "subtype": "meaning_identification",
        "proverb": "مثل یک", "gold_meaning": "معنی یک",
        "manual_distractors": ["دستی یک الف", "دستی یک ب"], "usage_note": None,
    },
    {
        "id": 2, "task_type": "proverb", "subtype": "meaning_identification",
        "proverb": "مثل دو", "gold_meaning": "معنی دو",
        "manual_distractors": ["دستی دو الف", "دستی دو ب", "دستی دو ج"], "usage_note": None,
    },
    {
        "id": 3, "task_type": "proverb", "subtype": "meaning_identification",
        "proverb": "مثل سه", "gold_meaning": "معنی سه",
        "manual_distractors": [], "usage_note": None,
    },
    {
        # نمونه‌ی ناقص — باید skip شود
        "id": 4, "task_type": "proverb", "subtype": "meaning_identification",
        "proverb": "مثل چهار",
    },
]


def test_build_pool_collects_unique_gold_meanings():
    pool = build_pool(RAW_SAMPLES)
    assert set(pool) == {"معنی یک", "معنی دو", "معنی سه"}


def _run_convert(tmp_path, num_options=4, seed=42):
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "task.jsonl"
    input_path.write_text(json.dumps(RAW_SAMPLES, ensure_ascii=False), encoding="utf-8")
    convert(str(input_path), str(output_path), num_options=num_options, seed=seed)
    text = output_path.read_text(encoding="utf-8").strip()
    lines = text.split("\n") if text else []
    return [json.loads(line) for line in lines]


def test_convert_skips_malformed_and_underfilled_samples(tmp_path):
    # id=3 دارای هیچ manual_distractor نیست و pool (فقط 2 معنی دیگر) برای
    # رسیدن به num_options=4 (نیاز به 3 گزینه‌ی غلط) کافی نیست، پس skip
    # می‌شود؛ id=4 هم ناقص است.
    samples = _run_convert(tmp_path)
    ids = {s["extra"]["raw_id"] for s in samples}
    assert ids == {1, 2}


def test_convert_each_sample_has_correct_number_of_options(tmp_path):
    samples = _run_convert(tmp_path, num_options=4)
    for s in samples:
        assert len(s["extra"]["options"]) == 4


def test_convert_gold_answer_matches_correct_option(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        correct_letter = s["gold_answer"]
        option = s["extra"]["options"][correct_letter]
        assert option["meaning"] == s["extra"]["gold_meaning"]


def test_convert_options_contain_no_duplicate_meanings(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        meanings = [opt["meaning"] for opt in s["extra"]["options"].values()]
        assert len(meanings) == len(set(meanings))


def test_convert_prefers_manual_distractors_before_pool_fallback(tmp_path):
    samples = _run_convert(tmp_path)
    by_id = {s["extra"]["raw_id"]: s for s in samples}
    # نمونه id=2 خودش 3 manual_distractor کامل دارد؛ پس هیچ نیازی به
    # pool fallback نیست و همه‌ی گزینه‌های غلط باید دقیقاً همان‌ها باشند.
    meanings = {opt["meaning"] for opt in by_id[2]["extra"]["options"].values()}
    assert meanings == {"معنی دو", "دستی دو الف", "دستی دو ب", "دستی دو ج"}


def test_convert_falls_back_to_pool_when_manual_distractors_insufficient(tmp_path):
    # با num_options=3 (نیاز به فقط 2 گزینه‌ی غلط)، id=3 اکنون کافی می‌شود
    # چون pool شامل 2 معنیِ دیگر (یک و دو) است.
    samples = _run_convert(tmp_path, num_options=3)
    by_id = {s["extra"]["raw_id"]: s for s in samples}
    meanings = {opt["meaning"] for opt in by_id[3]["extra"]["options"].values()}
    assert "معنی سه" in meanings
    assert meanings.issubset({"معنی سه", "معنی یک", "معنی دو"})


def test_convert_is_deterministic_given_same_seed(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    samples_a = _run_convert(tmp_path / "a", seed=7)
    samples_b = _run_convert(tmp_path / "b", seed=7)
    for a, b in zip(samples_a, samples_b):
        assert a["extra"]["options"] == b["extra"]["options"]
        assert a["gold_answer"] == b["gold_answer"]


def test_exact_match_scoring_matches_bare_letter():
    gold_answer = "ج"
    assert normalize_fa_text("ج") == normalize_fa_text(gold_answer)
    assert normalize_fa_text("ج.") == normalize_fa_text(gold_answer)
