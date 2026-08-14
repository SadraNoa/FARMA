"""
Converts the raw IFEval-lite source file
(data/raw_sources/ifeval_lite_fa_raw.json) into the executable TaskSample
format consumed by the shared pipeline (src/pipeline/runner.py).

Like the Jalali calendar task, this dataset is fully synthetic: 40
hand-picked Persian topics are combined with rule-verifiable instruction
constraints (length, keyword, position) generated deterministically by
scripts/generate_ifeval_lite_data.py.

Unlike every other task so far, there is no gold_answer to match against.
Instead, each sample carries a machine-checkable constraint spec under
extra.constraint, which the runner checks with
src/utils/instruction_utils.check_constraint against the model's raw
output (not an extracted "final answer" line, since the constraint governs
the shape of the whole response).

Usage:
    python scripts/prepare_ifeval_task.py \
        --input data/raw_sources/ifeval_lite_fa_raw.json \
        --output data/tasks/ifeval_lite_fa.jsonl
"""

import argparse
import json

SYSTEM_PROMPT_FA = (
    "تو یک دستیار فارسی‌زبان هستی که دقیقاً طبق دستورالعمل‌های داده‌شده عمل می‌کنی. "
    "دستور زیر را به‌طور کامل و دقیق اجرا کن. فقط متن خواسته‌شده را بنویس، "
    "بدون مقدمه، توضیح اضافه، یا هرگونه خط دیگر پیش یا پس از آن."
)

# The whole response is the object being checked (sentence count, starting
# word, forbidden word, ...), so the extraction regex simply captures the
# full raw output rather than a "پاسخ نهایی:" line.
ANSWER_EXTRACTION_REGEX = r"(?s)(.*)"

VALID_CATEGORIES = {"length_constraint", "keyword_constraint", "position_constraint"}


def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0
    per_category_count: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(raw_samples):
            category = item.get("category")
            question = item.get("question")
            constraint = item.get("constraint")
            topic = item.get("topic")

            if not category or not question or not constraint or category not in VALID_CATEGORIES:
                skipped += 1
                print(f"skipped (malformed sample): index={i}, category={category}")
                continue

            task_sample = {
                "sample_id": f"ifeval_{category}_{i:04d}",
                "task_type": "ifeval_lite",
                "problem_fa": question,
                "gold_answer": None,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "category": category,
                    "topic": topic,
                    "constraint": constraint,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            per_category_count[category] = per_category_count.get(category, 0) + 1

    converted = len(raw_samples) - skipped
    print(f"{converted} IFEval-lite samples converted and saved to: {output_path} ({skipped} skipped)")
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
