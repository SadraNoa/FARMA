"""
Unit tests for scripts/prepare_mmlu_task.py.

فقط توابع خالص (بدون وابستگی به شبکه/HF) تست می‌شوند: گروه‌بندی بر اساس
مبحث، نمونه‌گیری قطعی، استخراج گزینه‌ها، نگاشت پاسخ به حرف گزینه، و ساخت
TaskSample نهایی. با رکوردهای ساختگی (فرضی) کار می‌کند — به هیچ داده‌ی
واقعی از دیتاست گیت‌شده‌ی Khayyam Challenge نیاز ندارد.

Run: pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import pytest

from prepare_mmlu_task import (
    slugify_topic,
    group_by_topic,
    sample_group,
    extract_choices,
    resolve_gold_letter,
    build_task_sample,
    convert_records,
)

FAKE_RECORDS = [
    {"id": 1, "topic": "ریاضی", "question": "سوال یک", "choices": ["گزینه ۱", "گزینه ۲", "گزینه ۳", "گزینه ۴"], "answer": 1},
    {"id": 2, "topic": "ریاضی", "question": "سوال دو", "choices": ["گزینه ۱", "گزینه ۲", "گزینه ۳", "گزینه ۴"], "answer": "C"},
    {"id": 3, "topic": "ریاضی", "question": "سوال سه", "choices": ["گزینه ۱", "گزینه ۲", "گزینه ۳", "گزینه ۴"], "answer": "2"},
    {"id": 4, "topic": "زیست‌شناسی", "question": "سوال چهار", "choices": ["الف متن", "ب متن", "ج متن"], "answer": "ب متن"},
    {"id": 5, "topic": "زیست‌شناسی", "question": "سوال پنج", "choices": ["الف متن", "ب متن", "ج متن"], "answer": 0},
]


def test_slugify_topic_handles_persian_and_spaces():
    assert slugify_topic("زیست شناسی") == "زیست_شناسی"


def test_group_by_topic_groups_correctly():
    groups = group_by_topic(FAKE_RECORDS, "topic")
    assert set(groups.keys()) == {"ریاضی", "زیست‌شناسی"}
    assert len(groups["ریاضی"]) == 3
    assert len(groups["زیست‌شناسی"]) == 2


def test_sample_group_caps_at_available_count():
    records = group_by_topic(FAKE_RECORDS, "topic")["ریاضی"]
    selected = sample_group(records, n_requested=10, seed_key="42-ریاضی")
    assert len(selected) == 3  # فقط ۳ رکورد موجود است


def test_sample_group_respects_requested_count():
    records = group_by_topic(FAKE_RECORDS, "topic")["ریاضی"]
    selected = sample_group(records, n_requested=2, seed_key="42-ریاضی")
    assert len(selected) == 2


def test_sample_group_deterministic_given_same_seed():
    records = group_by_topic(FAKE_RECORDS, "topic")["ریاضی"]
    a = sample_group(records, n_requested=2, seed_key="7-ریاضی")
    b = sample_group(records, n_requested=2, seed_key="7-ریاضی")
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_extract_choices_from_list_column():
    choices = extract_choices(FAKE_RECORDS[0], choices_col="choices", choices_cols=None)
    assert choices == ["گزینه ۱", "گزینه ۲", "گزینه ۳", "گزینه ۴"]


def test_extract_choices_from_separate_columns():
    record = {"opt_a": "یک", "opt_b": "دو", "opt_c": "سه"}
    choices = extract_choices(record, choices_col=None, choices_cols=["opt_a", "opt_b", "opt_c"])
    assert choices == ["یک", "دو", "سه"]


@pytest.mark.parametrize("answer,expected_letter", [
    (1, "ب"),          # اندیس صفرمبنا
    ("C", "ج"),         # حرف لاتین
    ("2", "ج"),          # رشته‌ی عددی صفرمبنا -> اندیس ۲ -> "ج"
])
def test_resolve_gold_letter_various_encodings(answer, expected_letter):
    choices = ["گزینه ۱", "گزینه ۲", "گزینه ۳", "گزینه ۴"]
    letters = ["الف", "ب", "ج", "د"]
    assert resolve_gold_letter(choices, answer, letters) == expected_letter


def test_resolve_gold_letter_matches_choice_text():
    choices = ["الف متن", "ب متن", "ج متن"]
    letters = ["الف", "ب", "ج"]
    assert resolve_gold_letter(choices, "ب متن", letters) == "ب"


def test_resolve_gold_letter_raises_on_unmappable_answer():
    choices = ["یک", "دو"]
    letters = ["الف", "ب"]
    with pytest.raises(ValueError):
        resolve_gold_letter(choices, "چیز نامرتبط", letters)


def test_build_task_sample_shape():
    sample = build_task_sample(
        FAKE_RECORDS[0], idx=0, topic="ریاضی",
        question_col="question", choices_col="choices", choices_cols=None,
        answer_col="answer", subject_col=None, difficulty_col=None, stage_col=None, id_col="id",
    )
    assert sample["task_type"] == "mmlu_lite"
    assert sample["sample_id"] == "mmlu_lite_ریاضی_0000"
    assert sample["gold_answer"] == "ب"
    assert sample["apply_persian_stability_rubric"] is True
    assert sample["apply_verifiable_reasoning_rubric"] is False
    assert sample["extra"]["options"]["ب"] == "گزینه ۲"
    assert sample["extra"]["raw_id"] == 1


def test_convert_records_applies_uniform_samples_per_topic():
    task_samples, per_topic, skipped = convert_records(
        FAKE_RECORDS, topic_col="topic", question_col="question", answer_col="answer",
        choices_col="choices", samples_per_topic=2, seed=42,
    )
    assert skipped == 0
    assert per_topic["ریاضی"] == 2
    assert per_topic["زیست‌شناسی"] == 2
    assert len(task_samples) == 4


def test_convert_records_applies_per_topic_override():
    task_samples, per_topic, skipped = convert_records(
        FAKE_RECORDS, topic_col="topic", question_col="question", answer_col="answer",
        choices_col="choices", samples_per_topic=1,
        topic_samples_override={"ریاضی": 3}, seed=42,
    )
    assert per_topic["ریاضی"] == 3
    assert per_topic["زیست‌شناسی"] == 1
    assert len(task_samples) == 4


def test_convert_records_skips_malformed_sample():
    bad_records = FAKE_RECORDS + [
        {"id": 99, "topic": "ریاضی", "question": "سوال بد", "choices": ["فقط یک گزینه"], "answer": 0}
    ]
    task_samples, per_topic, skipped = convert_records(
        bad_records, topic_col="topic", question_col="question", answer_col="answer",
        choices_col="choices", samples_per_topic=10, seed=42,
    )
    # نمونه‌ی بد (کمتر از ۲ گزینه) باید رد شود اما بقیه سالم بمانند
    assert skipped == 1
    assert len(task_samples) == 5


def test_convert_records_sample_ids_are_unique():
    task_samples, _, _ = convert_records(
        FAKE_RECORDS, topic_col="topic", question_col="question", answer_col="answer",
        choices_col="choices", samples_per_topic=10, seed=42,
    )
    ids = [s["sample_id"] for s in task_samples]
    assert len(ids) == len(set(ids))
