"""
Converts the raw paraphrase-robustness source file
(data/raw_sources/paraphrase_robustness_fa_raw.json) into the executable
TaskSample format consumed by the shared pipeline (src/pipeline/runner.py).

Phase 3 — Paraphrase robustness: tests whether a model's answer to a
factual question stays correct when the *same* question is reworded —
different word order, synonyms, formal/informal phrasing, direct vs.
indirect question form — while the underlying fact being asked about, and
the passage it's answered from, stay identical.

Data is built from 15 hand-authored (passage, correct_answer, 3 foils)
families, each expressed as 4 differently-worded questions asking exactly
the same thing (60 samples total). Every one of the 4 paraphrases in a
family shares the same passage and the same 4 candidate options (the real
answer + the same 3 foils) — only the wording of the question itself
changes. This isolates the variable the task is designed to probe: a model
whose answer flips across paraphrases of an otherwise-identical question is
not robust, regardless of whether any single answer happens to be right.

Because every sample already has one fixed correct option among a small
closed set, this reuses the framework's generic exact-match `gold_answer`
path — no runner changes needed, the same way every other closed-set task
in this project works. Per-family robustness itself is an aggregate
property (do all 4 gold_answer-correct results agree?) that can be read
off `extra.family_id` groupings at analysis/aggregation time; scoring each
sample individually needs no special handling.

Usage:
    python scripts/prepare_paraphrase_robustness_task.py \
        --input data/raw_sources/paraphrase_robustness_fa_raw.json \
        --output data/tasks/paraphrase_robustness_fa.jsonl \
        --seed 42
"""

import argparse
import json
import random

SYSTEM_PROMPT_FA = (
    "تو یک دستیار دقیق هستی. یک متن کوتاه و یک سؤال درباره‌ی آن به تو داده "
    "می‌شود. با توجه به اطلاعات متن، گزینه‌ی درست را از میان چهار گزینه "
    "انتخاب کن. در پایان، دقیقاً یک خط جداگانه با این قالب بنویس (فقط حرفِ "
    "گزینه، بدون هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

OPTION_LETTERS = ["الف", "ب", "ج", "د"]

VALID_SUBTYPES = {"paraphrase_variant"}


def convert(input_path: str, output_path: str, seed: int = 42):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0
    per_family_count: dict[int, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in raw_samples:
            subtype = item.get("subtype")
            passage = item.get("passage")
            question = item.get("question")
            correct_answer = item.get("correct_answer")
            foils = item.get("foils")
            raw_id = item.get("id")
            family_id = item.get("family_id")

            if (
                subtype not in VALID_SUBTYPES
                or not passage
                or not question
                or not correct_answer
                or not foils
                or len(foils) != 3
                or correct_answer in foils
                or raw_id is None
                or family_id is None
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}")
                continue

            # ترتیب نمایش گزینه‌ها به‌صورت قطعی و قابل‌تکرار جابه‌جا می‌شود؛
            # توجه: seed فقط به id نمونه وابسته است، نه به family_id، پس
            # ترتیب گزینه‌ها می‌تواند بین چهار بازنویسی یک خانواده فرق کند —
            # این عمداً است تا سیگنال استحکام صرفاً از خودِ بازنویسیِ سؤال
            # بیاید، نه از یادگیریِ «جای گزینه‌ی درست ثابت است».
            rng = random.Random(f"{seed}-{raw_id}")
            options_content = [correct_answer] + list(foils)
            rng.shuffle(options_content)

            options_lines = []
            correct_letter = None
            for letter, option_text in zip(OPTION_LETTERS, options_content):
                options_lines.append(f"{letter}) {option_text}")
                if option_text == correct_answer:
                    correct_letter = letter

            question_block = (
                f"متن: {passage}\n"
                f"سؤال: {question}\n\n"
                + "\n".join(options_lines)
            )

            task_sample = {
                "sample_id": f"paraphrase_robustness_{raw_id:04d}",
                "task_type": "paraphrase_robustness",
                "problem_fa": question_block,
                "gold_answer": correct_letter,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "subtype": subtype,
                    "family_id": family_id,
                    "paraphrase_index": item.get("paraphrase_index"),
                    "passage": passage,
                    "question": question,
                    "correct_answer": correct_answer,
                    "foils": foils,
                    "correct_option": correct_letter,
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            per_family_count[family_id] = per_family_count.get(family_id, 0) + 1

    converted = len(raw_samples) - skipped
    print(f"{converted} paraphrase-robustness samples converted and saved to: {output_path} ({skipped} skipped)")
    print(f"تعداد خانواده‌ها (family_id): {len(per_family_count)}")
    for family_id, count in sorted(per_family_count.items()):
        print(f"  family {family_id}: {count} paraphrases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed برای جابه‌جایی قطعی و قابل‌تکرار ترتیب گزینه‌ها")
    args = parser.parse_args()
    convert(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
