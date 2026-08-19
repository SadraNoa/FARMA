"""
Unit tests for scripts/prepare_ieham_task.py
(Ieham/wordplay same-word-only option generation, keyed on far-meaning).
Run: pytest tests/ -v
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from prepare_ieham_task import build_word_pool, convert
from src.utils.text_utils import normalize_fa_text

RAW_SAMPLES = [
    {
        "id": 1, "task_type": "ieham", "subtype": "double_meaning_identification",
        "verse": "بیت یک با کلمه الف.", "ambiguous_word": "الف",
        "gold_meaning_near": "نزدیکِ یک کامل", "gold_meaning_far": "دورِ یک",
        "manual_distractors": [{"near": "نزدیکِ دو کوتاه", "far": "دورِ دو"}],
        "poet": "شاعر یک", "source": None,
    },
    {
        "id": 2, "task_type": "ieham", "subtype": "double_meaning_identification",
        "verse": "بیت دو با کلمه الف.", "ambiguous_word": "الف",
        "gold_meaning_near": "نزدیکِ دو کامل", "gold_meaning_far": "دورِ دو",
        "manual_distractors": [{"near": "نزدیکِ یک", "far": "دورِ یک"}],
        "poet": "شاعر یک", "source": None,
    },
    {
        "id": 3, "task_type": "ieham", "subtype": "double_meaning_identification",
        "verse": "بیت سه با کلمه ب.", "ambiguous_word": "ب",
        "gold_meaning_near": "نزدیکِ سه", "gold_meaning_far": "دورِ سه",
        "manual_distractors": [
            {"near": "نزدیکِ چهار", "far": "دورِ چهار"},
            {"near": "نزدیکِ پنج", "far": "دورِ پنج"},
        ],
        "poet": None, "source": None,
    },
    {
        # کلمه‌ای بدون هیچ manual_distractor و بدون تکرار در جای دیگر — باید skip شود
        "id": 4, "task_type": "ieham", "subtype": "double_meaning_identification",
        "verse": "بیت چهار با کلمه تنها.", "ambiguous_word": "تنها",
        "gold_meaning_near": "نزدیکِ شش", "gold_meaning_far": "دورِ شش",
        "manual_distractors": [], "poet": None, "source": None,
    },
]


def test_build_word_pool_uses_far_as_key_not_near_far_pair():
    pool = build_word_pool(RAW_SAMPLES)
    assert set(pool["الف"].keys()) == {"دورِ یک", "دورِ دو"}
    # نزدیکِ کانونی هر دور باید همان نزدیکِ نوشته‌شده در جایی باشد که آن
    # دور خودِ gold_meaning_far بوده (نه عبارتِ کوتاه‌ترِ آمده در distractor)
    assert pool["الف"]["دورِ یک"] == "نزدیکِ یک کامل"
    assert pool["الف"]["دورِ دو"] == "نزدیکِ دو کامل"


def test_build_word_pool_is_scoped_per_word():
    pool = build_word_pool(RAW_SAMPLES)
    assert "دورِ سه" not in pool["الف"]
    assert set(pool["ب"].keys()) == {"دورِ سه", "دورِ چهار", "دورِ پنج"}


def _run_convert(tmp_path, max_options=4, min_options=2, seed=42):
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "task.jsonl"
    input_path.write_text(json.dumps(RAW_SAMPLES, ensure_ascii=False), encoding="utf-8")
    convert(str(input_path), str(output_path), max_options=max_options, min_options=min_options, seed=seed)
    text = output_path.read_text(encoding="utf-8").strip()
    lines = text.split("\n") if text else []
    return [json.loads(line) for line in lines]


def test_convert_skips_samples_with_no_known_alternate_pair(tmp_path):
    samples = _run_convert(tmp_path)
    ids = {s["extra"]["raw_id"] for s in samples}
    assert ids == {1, 2, 3}


def test_convert_variable_option_counts(tmp_path):
    samples = _run_convert(tmp_path)
    by_id = {s["extra"]["raw_id"]: s for s in samples}
    assert len(by_id[1]["extra"]["options"]) == 2
    assert len(by_id[3]["extra"]["options"]) == 3


def test_convert_no_duplicate_far_meanings_within_one_question(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        fars = [opt["far"] for opt in s["extra"]["options"].values()]
        assert len(fars) == len(set(fars))


def test_convert_gold_answer_matches_correct_option(tmp_path):
    samples = _run_convert(tmp_path)
    for s in samples:
        correct_letter = s["gold_answer"]
        option = s["extra"]["options"][correct_letter]
        assert option["far"] == s["extra"]["gold_meaning_far"]


def test_convert_never_pulls_distractor_from_another_word(tmp_path):
    samples = _run_convert(tmp_path)
    by_id = {s["extra"]["raw_id"]: s for s in samples}
    fars_for_alef_q = {opt["far"] for opt in by_id[1]["extra"]["options"].values()}
    assert fars_for_alef_q.issubset({"دورِ یک", "دورِ دو"})


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
    bad_samples = RAW_SAMPLES + [{"id": 5, "task_type": "ieham", "subtype": "double_meaning_identification"}]
    input_path = tmp_path / "raw_bad.json"
    output_path = tmp_path / "task_bad.jsonl"
    input_path.write_text(json.dumps(bad_samples, ensure_ascii=False), encoding="utf-8")
    convert(str(input_path), str(output_path), max_options=4, min_options=2, seed=42)
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    # id=4 (تنها) skip می‌شود چون جفت معنای جایگزین ندارد، id=5 هم به‌خاطر ناقص بودن
    assert len(lines) == 3
