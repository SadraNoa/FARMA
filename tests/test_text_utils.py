"""
تست‌های واحد ساده برای توابع src/utils/text_utils.py.
اجرا: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.text_utils import (
    code_switch_rate,
    persian_script_ratio,
    split_sentences,
    extract_final_answer,
    normalize_fa_text,
    try_parse_json,
)


def test_code_switch_rate_no_english():
    text = "این یک متن کاملاً فارسی است و هیچ کلمه انگلیسی ندارد."
    assert code_switch_rate(text) == 0.0


def test_code_switch_rate_with_english():
    text = "این جمله شامل کلمه think و reasoning است در میان متن فارسی است."
    rate = code_switch_rate(text)
    assert rate > 0.0


def test_code_switch_rate_whitelist_excluded():
    text = "این مسئله شامل LATEX و JSON است اما بقیه فارسی است."
    assert code_switch_rate(text) == 0.0


def test_persian_script_ratio_pure_persian():
    text = "سلام دنیا این یک تست است."
    assert persian_script_ratio(text) == 1.0


def test_persian_script_ratio_mixed():
    text = "سلام hello دنیا world"
    ratio = persian_script_ratio(text)
    assert 0.0 < ratio < 1.0


def test_extract_final_answer_found():
    text = "چند گام استدلال...\nپاسخ نهایی: ۴۲"
    assert extract_final_answer(text) == "۴۲"


def test_extract_final_answer_not_found():
    text = "متنی بدون فرمت مشخص جواب."
    assert extract_final_answer(text) is None


def test_normalize_fa_text_arabic_chars():
    assert normalize_fa_text("كتاب") == "کتاب"
    assert normalize_fa_text("علي") == "علی"


def test_try_parse_json_plain():
    parsed = try_parse_json('{"a": 1, "b": 2}')
    assert parsed == {"a": 1, "b": 2}


def test_try_parse_json_with_markdown_fence():
    parsed = try_parse_json('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}


def test_try_parse_json_invalid():
    assert try_parse_json("این یک JSON نیست") is None


def test_split_sentences():
    text = "جمله اول. جمله دوم! جمله سوم؟"
    sentences = split_sentences(text)
    assert len(sentences) == 3
