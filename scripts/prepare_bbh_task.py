"""
Converts the raw BBH-lite source file (data/raw_sources/bbh_lite_fa_raw.json)
into the executable TaskSample format consumed by the shared pipeline
(src/pipeline/runner.py).

Unlike the math task, every sample here has a short, unambiguous
gold_answer, so correctness is computed automatically by the runner via
exact-match string comparison (after normalization). No judge-model rubric
is required for correctness itself; the persian_stability rubric is still
applied so we can separately track whether the model's short Persian
reasoning stays linguistically stable even on easy, deterministic tasks.

Usage:
    python scripts/prepare_bbh_task.py \
        --input data/raw_sources/bbh_lite_fa_raw.json \
        --output data/tasks/bbh_lite_fa.jsonl
"""

import argparse
import json

SYSTEM_PROMPT_FA = (
    "تو یک دستیار منطقی و دقیق فارسی‌زبان هستی. به سوال زیر با استدلال کوتاه و "
    "به زبان فارسی پاسخ بده. در پایان، دقیقاً یک خط جداگانه با این قالب بنویس:\n"
    "پاسخ نهایی: <جواب کوتاه و دقیق>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

VALID_CATEGORIES = {"object_ordering", "navigate", "arithmetic", "boolean_logic"}


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
                "sample_id": f"bbh_{category}_{i:04d}",
                "task_type": "bbh_lite",
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
    print(f"{converted} BBH-lite samples converted and saved to: {output_path} ({skipped} skipped)")
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
