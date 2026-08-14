"""
Converts the raw Jalali calendar source file
(data/raw_sources/jalali_calendar_fa_raw.json) into the executable
TaskSample format consumed by the shared pipeline (src/pipeline/runner.py).

Unlike the Math task, this dataset is entirely synthetic: every question
and gold_answer was generated deterministically with the `jdatetime`
library (see the sibling generation script referenced in
docs/tasks/jalali_calendar_task.md), so there is no manual translation or
authoring step here beyond the initial generation. This script only
validates and reshapes the already-correct raw samples.

Usage:
    python scripts/prepare_jalali_task.py \
        --input data/raw_sources/jalali_calendar_fa_raw.json \
        --output data/tasks/jalali_calendar_fa.jsonl
"""

import argparse
import json

SYSTEM_PROMPT_FA = (
    "تو یک دستیار دقیق فارسی‌زبان هستی که در محاسبات تقویم جلالی و میلادی مهارت داری. "
    "به سوال زیر با استدلال کوتاه و به زبان فارسی پاسخ بده. "
    "در پایان، دقیقاً یک خط جداگانه با این قالب بنویس:\n"
    "پاسخ نهایی: <جواب کوتاه و دقیق>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

VALID_CATEGORIES = {"date_conversion", "leap_year", "date_arithmetic", "date_distance"}


def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0
    per_category_count: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(raw_samples):
            category = item.get("category")
            question = item.get("question")
            answer = item.get("answer")

            if not category or not question or not answer or category not in VALID_CATEGORIES:
                skipped += 1
                print(f"skipped (malformed sample): index={i}, category={category}")
                continue

            task_sample = {
                "sample_id": f"jalali_{category}_{i:04d}",
                "task_type": "jalali_calendar",
                "problem_fa": question,
                "gold_answer": answer,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "category": category,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            per_category_count[category] = per_category_count.get(category, 0) + 1

    converted = len(raw_samples) - skipped
    print(f"{converted} Jalali calendar samples converted and saved to: {output_path} ({skipped} skipped)")
    for category, count in sorted(per_category_count.items()):
        print(f"  {category}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
