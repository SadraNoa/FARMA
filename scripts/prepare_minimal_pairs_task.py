"""
Converts the raw minimal-pairs source file
(data/raw_sources/minimal_pairs_fa_raw.json) into the executable TaskSample
format consumed by the shared pipeline (src/pipeline/runner.py).

Phase 3 — Minimal pairs (grammaticality judgment): each item is a matched
pair of Persian sentences that differ by one targeted grammatical feature
(subject-verb agreement, verb-final word order, the "را" definite-object
marker, negation-prefix placement, tense/adverb agreement, plural-marking,
the comparative suffix "تر", or preposition compatibility) — one sentence
is grammatical, the other is a minimally-edited ungrammatical variant. The
model is shown both sentences (labeled الف/ب, order randomized per-sample)
and must pick which one is grammatically correct.

Because each pair already has exactly one correct label, this reuses the
framework's generic exact-match `gold_answer` path — no runner or judge
changes are needed, the same way Aroozi/BBH-lite/Jalali/script-disambiguation
work. Unlike those multi-way multiple-choice tasks, minimal pairs is a
2-option (الف/ب) choice by construction — there is no distractor pool to
build, since the "distractor" is always the paired ungrammatical sentence.

Usage:
    python scripts/prepare_minimal_pairs_task.py \
        --input data/raw_sources/minimal_pairs_fa_raw.json \
        --output data/tasks/minimal_pairs_fa.jsonl \
        --seed 42
"""

import argparse
import json
import random

SYSTEM_PROMPT_FA = (
    "تو یک دستیار متخصص در دستور زبان فارسی هستی. دو جمله‌ی زیر تنها در یک "
    "نکته‌ی دستوری با هم تفاوت دارند؛ فقط یکی از آن‌ها از نظر دستور زبان "
    "فارسی درست است. با دقت هر دو جمله را بررسی کن و جمله‌ی درست را مشخص کن. "
    "در پایان، دقیقاً یک خط جداگانه با این قالب بنویس (فقط حرفِ گزینه، بدون "
    "هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

OPTION_LETTERS = ["الف", "ب"]

VALID_SUBTYPES = {"grammaticality_judgment"}


def convert(input_path: str, output_path: str, seed: int = 42):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0
    per_category_count: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in raw_samples:
            subtype = item.get("subtype")
            category = item.get("category")
            correct_sentence = item.get("correct_sentence")
            incorrect_sentence = item.get("incorrect_sentence")
            raw_id = item.get("id")

            if (
                subtype not in VALID_SUBTYPES
                or not category
                or not correct_sentence
                or not incorrect_sentence
                or correct_sentence == incorrect_sentence
                or raw_id is None
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}")
                continue

            # جایگاه دو جمله (الف/ب) به‌صورت قطعی و قابل‌تکرار جابه‌جا می‌شود
            # تا مدل نتواند صرفاً با حدس زدن "همیشه الف" یا "همیشه ب" جواب بدهد.
            rng = random.Random(f"{seed}-{raw_id}")
            sentences = [
                (correct_sentence, True),
                (incorrect_sentence, False),
            ]
            rng.shuffle(sentences)

            options_lines = []
            correct_letter = None
            for letter, (sentence, is_correct) in zip(OPTION_LETTERS, sentences):
                options_lines.append(f"{letter}) {sentence}")
                if is_correct:
                    correct_letter = letter

            question = (
                "کدام‌یک از دو جمله‌ی زیر از نظر دستور زبان فارسی درست است؟\n\n"
                + "\n".join(options_lines)
            )

            task_sample = {
                "sample_id": f"minimal_pairs_{raw_id:04d}",
                "task_type": "minimal_pairs",
                "problem_fa": question,
                "gold_answer": correct_letter,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "subtype": subtype,
                    "category": category,
                    "category_label": item.get("category_label"),
                    "rule": item.get("rule"),
                    "correct_sentence": correct_sentence,
                    "incorrect_sentence": incorrect_sentence,
                    "correct_option": correct_letter,
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            per_category_count[category] = per_category_count.get(category, 0) + 1

    converted = len(raw_samples) - skipped
    print(f"{converted} minimal-pairs samples converted and saved to: {output_path} ({skipped} skipped)")
    for category, count in sorted(per_category_count.items()):
        print(f"  {category}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed برای جابه‌جایی قطعی و قابل‌تکرار جایگاه الف/ب")
    args = parser.parse_args()
    convert(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
