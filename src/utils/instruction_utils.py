"""
Rule-based verifiers for the IFEval-lite task. Each function checks a
single constraint type against a raw model response and returns a
(passed: bool, reason_fa: str) pair. No LLM judge is involved: every
constraint here is checked deterministically so that correctness is cheap
and 100% reproducible.

These checkers operate on the full raw response text (not on a
"final answer" line), since the constraint usually governs the shape of
the entire response (sentence count, starting word, forbidden word, etc.).
"""

import re

from src.utils.text_utils import split_sentences


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _strip_edge_punctuation(word: str) -> str:
    return word.strip(" .،؛:!؟?\u200c")


def check_sentence_count(text: str, value: int, mode: str) -> tuple[bool, str]:
    sentences = split_sentences(text)
    n = len(sentences)
    if mode == "exact":
        passed = n == value
    elif mode == "at_least":
        passed = n >= value
    elif mode == "at_most":
        passed = n <= value
    else:
        return False, f"حالت ناشناخته برای sentence_count: {mode}"
    return passed, f"تعداد جمله‌ها: {n} (هدف: {mode} {value})"


def check_word_count(text: str, value: int, mode: str) -> tuple[bool, str]:
    n = len(text.split())
    if mode == "exact":
        passed = n == value
    elif mode == "at_least":
        passed = n >= value
    elif mode == "at_most":
        passed = n <= value
    else:
        return False, f"حالت ناشناخته برای word_count: {mode}"
    return passed, f"تعداد کلمه‌ها: {n} (هدف: {mode} {value})"


def check_paragraph_count(text: str, value: int, mode: str) -> tuple[bool, str]:
    paragraphs = _split_paragraphs(text)
    n = len(paragraphs)
    if mode == "exact":
        passed = n == value
    elif mode == "at_least":
        passed = n >= value
    elif mode == "at_most":
        passed = n <= value
    else:
        return False, f"حالت ناشناخته برای paragraph_count: {mode}"
    return passed, f"تعداد پاراگراف‌ها: {n} (هدف: {mode} {value})"


def check_must_include(text: str, word: str) -> tuple[bool, str]:
    n = text.count(word)
    passed = n >= 1
    return passed, f"تعداد تکرار «{word}»: {n} (هدف: حداقل ۱)"


def check_must_include_exact_count(text: str, word: str, count: int) -> tuple[bool, str]:
    n = text.count(word)
    passed = n == count
    return passed, f"تعداد تکرار «{word}»: {n} (هدف: دقیقاً {count})"


def check_must_exclude(text: str, word: str) -> tuple[bool, str]:
    n = text.count(word)
    passed = n == 0
    return passed, f"تعداد تکرار «{word}»: {n} (هدف: صفر)"


def check_must_start_with(text: str, word: str) -> tuple[bool, str]:
    words = text.strip().split()
    first = _strip_edge_punctuation(words[0]) if words else ""
    passed = first == word
    return passed, f"اولین کلمه: «{first}» (هدف: «{word}»)"


def check_must_end_with(text: str, word: str) -> tuple[bool, str]:
    words = text.strip().split()
    last = _strip_edge_punctuation(words[-1]) if words else ""
    passed = last == word
    return passed, f"آخرین کلمه: «{last}» (هدف: «{word}»)"


_CHECKERS = {
    "sentence_count": lambda text, c: check_sentence_count(text, c["value"], c["mode"]),
    "word_count": lambda text, c: check_word_count(text, c["value"], c["mode"]),
    "paragraph_count": lambda text, c: check_paragraph_count(text, c["value"], c["mode"]),
    "must_include": lambda text, c: check_must_include(text, c["word"]),
    "must_include_exact_count": lambda text, c: check_must_include_exact_count(text, c["word"], c["count"]),
    "must_exclude": lambda text, c: check_must_exclude(text, c["word"]),
    "must_start_with": lambda text, c: check_must_start_with(text, c["word"]),
    "must_end_with": lambda text, c: check_must_end_with(text, c["word"]),
}


def check_constraint(text: str, constraint: dict) -> tuple[bool, str]:
    """Dispatches to the right rule-based checker based on constraint['type'].
    Returns (passed, reason_fa). Unknown constraint types fail closed."""
    ctype = constraint.get("type")
    checker = _CHECKERS.get(ctype)
    if checker is None:
        return False, f"نوع محدودیت ناشناخته: {ctype}"
    return checker(text, constraint)
