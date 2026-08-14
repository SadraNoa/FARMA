"""
Unit tests for src/utils/instruction_utils.py (IFEval-lite constraint
checkers).
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.instruction_utils import check_constraint


def test_sentence_count_exact_pass():
    text = "جمله اول است. جمله دوم است. جمله سوم است."
    passed, _ = check_constraint(text, {"type": "sentence_count", "mode": "exact", "value": 3})
    assert passed is True


def test_sentence_count_exact_fail():
    text = "جمله اول است. جمله دوم است."
    passed, _ = check_constraint(text, {"type": "sentence_count", "mode": "exact", "value": 3})
    assert passed is False


def test_word_count_at_least_pass():
    text = "یک دو سه چهار پنج شش"
    passed, _ = check_constraint(text, {"type": "word_count", "mode": "at_least", "value": 5})
    assert passed is True


def test_word_count_at_least_fail():
    text = "یک دو سه"
    passed, _ = check_constraint(text, {"type": "word_count", "mode": "at_least", "value": 5})
    assert passed is False


def test_paragraph_count_exact():
    text = "پاراگراف اول.\n\nپاراگراف دوم.\n\nپاراگراف سوم."
    passed, _ = check_constraint(text, {"type": "paragraph_count", "mode": "exact", "value": 3})
    assert passed is True


def test_must_include_pass():
    text = "این متن شامل کلمه‌ی پرنده است."
    passed, _ = check_constraint(text, {"type": "must_include", "word": "پرنده", "count": 1})
    assert passed is True


def test_must_include_fail():
    text = "این متن آن کلمه را ندارد."
    passed, _ = check_constraint(text, {"type": "must_include", "word": "پرنده", "count": 1})
    assert passed is False


def test_must_include_exact_count_pass():
    text = "باران باران باران می‌بارد."
    passed, _ = check_constraint(text, {"type": "must_include_exact_count", "word": "باران", "count": 3})
    assert passed is True


def test_must_include_exact_count_fail_too_many():
    text = "باران باران باران باران می‌بارد."
    passed, _ = check_constraint(text, {"type": "must_include_exact_count", "word": "باران", "count": 3})
    assert passed is False


def test_must_exclude_pass():
    text = "این متن هیچ کلمه ممنوعه‌ای ندارد."
    passed, _ = check_constraint(text, {"type": "must_exclude", "word": "متأسفانه"})
    assert passed is True


def test_must_exclude_fail():
    text = "متأسفانه این کلمه اینجاست."
    passed, _ = check_constraint(text, {"type": "must_exclude", "word": "متأسفانه"})
    assert passed is False


def test_must_start_with_pass():
    text = "امروز هوا خوب است."
    passed, _ = check_constraint(text, {"type": "must_start_with", "word": "امروز"})
    assert passed is True


def test_must_start_with_fail():
    text = "هوا امروز خوب است."
    passed, _ = check_constraint(text, {"type": "must_start_with", "word": "امروز"})
    assert passed is False


def test_must_end_with_pass():
    text = "هوا امروز خوب است."
    passed, _ = check_constraint(text, {"type": "must_end_with", "word": "است"})
    assert passed is True


def test_must_end_with_fail():
    text = "هوا امروز خوب است واقعا."
    passed, _ = check_constraint(text, {"type": "must_end_with", "word": "است"})
    assert passed is False


def test_unknown_constraint_type_fails_closed():
    passed, reason = check_constraint("متن دلخواه", {"type": "not_a_real_type"})
    assert passed is False
    assert "ناشناخته" in reason
