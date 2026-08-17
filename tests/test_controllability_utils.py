"""
Unit tests for src/utils/controllability_utils.py.
Run: pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.controllability_utils import (
    check_paragraph_count,
    check_sentence_structure,
    check_rule_constraint,
    RULE_CHECKABLE_TYPES,
)


def test_paragraph_count_exact_match():
    passed, _ = check_paragraph_count("بند یک\n\nبند دو\n\nبند سه", "3")
    assert passed is True


def test_paragraph_count_mismatch():
    passed, _ = check_paragraph_count("بند یک\n\nبند دو", "3")
    assert passed is False


def test_paragraph_count_invalid_value():
    passed, reason = check_paragraph_count("بند یک", "not_a_number")
    assert passed is False
    assert "نامعتبر" in reason


def test_no_ke_clauses_passes_without_marker():
    passed, _ = check_sentence_structure("این یک جمله ساده است.", "no_ke_clauses")
    assert passed is True


def test_no_ke_clauses_fails_with_marker():
    passed, _ = check_sentence_structure("من فکر می‌کنم که این خوب است.", "no_ke_clauses")
    assert passed is False


def test_verb_final_heuristic_passes_clean_sentence():
    passed, _ = check_sentence_structure("او به خانه رفت.", "verb_final")
    assert passed is True


def test_verb_final_heuristic_flags_trailing_comma():
    passed, _ = check_sentence_structure("عدد آن پنج،", "verb_final")
    assert passed is False


def test_verb_final_heuristic_empty_text():
    passed, reason = check_sentence_structure("", "verb_final")
    assert passed is False


def test_unknown_sentence_structure_value_fails_closed():
    passed, reason = check_sentence_structure("متن", "not_a_real_rule")
    assert passed is False
    assert "ناشناخته" in reason


def test_check_rule_constraint_dispatches_paragraph_count():
    passed, _ = check_rule_constraint("بند یک\n\nبند دو", {"type": "paragraph_count", "value": "2"})
    assert passed is True


def test_check_rule_constraint_dispatches_sentence_structure():
    passed, _ = check_rule_constraint(
        "این جمله ساده است.", {"type": "sentence_structure", "value": "no_ke_clauses"}
    )
    assert passed is True


def test_check_rule_constraint_unknown_type_fails_closed():
    passed, reason = check_rule_constraint("متن", {"type": "lexical", "value": "x"})
    assert passed is False
    assert "ناشناخته" in reason


def test_rule_checkable_types_set():
    assert RULE_CHECKABLE_TYPES == {"paragraph_count", "sentence_structure"}
