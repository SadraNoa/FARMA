"""
Converts the raw Deep Brainstorm source file
(data/raw_sources/deep_brainstorm_fa_raw.json) into the executable
TaskSample format consumed by the shared pipeline (src/pipeline/runner.py).

This task has no gold_answer -- it's open-ended by design. Scoring is done
entirely through two rubrics applied together:
  - verifiable_reasoning (already lists "deep_brainstorm" in its
    applies_to): logical-quality dimensions -- step_support,
    conclusion_validity, no_logical_jump, answer_consistency.
  - deep_brainstorm (new, rubrics/deep_brainstorm.json): dimensions not
    covered by verifiable_reasoning -- idea_diversity, self_correction,
    cultural_grounding.
persian_stability is also applied, same as every other CoT task.

Usage:
    python scripts/prepare_deep_brainstorm_task.py \
        --input data/raw_sources/deep_brainstorm_fa_raw.json \
        --output data/tasks/deep_brainstorm_fa.jsonl
"""

import argparse
import json

SYSTEM_PROMPT_FA = (
    "تو یک دستیار خلاق و تحلیل‌گر فارسی‌زبان هستی. به سوال باز زیر با "
    "زنجیره استدلال کامل پاسخ بده: چند مسیر فکری متفاوت را واقعاً بررسی کن، "
    "در صورت لزوم ایده قبلی خودت را نقد یا اصلاح کن، و ایده‌ها را متناسب با "
    "بافت ایرانی/فارسی‌زبان بومی‌سازی کن. در پایان، دقیقاً یک خط جداگانه با "
    "این قالب بنویس:\n"
    "پاسخ نهایی: <جمع‌بندی کوتاه توصیه یا نتیجه نهایی>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

VALID_CATEGORIES = {
    "open_problem_solving",
    "creative_ideation",
    "policy_or_social",
    "product_or_design",
}


def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(raw_samples):
            item_id = item.get("id")
            prompt = item.get("prompt_fa")
            category = item.get("category")

            if not item_id or not prompt or category not in VALID_CATEGORIES:
                skipped += 1
                print(f"skipped (malformed sample): index={i}, id={item_id}, category={category}")
                continue

            task_sample = {
                "sample_id": item_id,
                "task_type": "deep_brainstorm",
                "problem_fa": prompt,
                "gold_answer": None,
                "requires_cot_judging": True,
                "apply_verifiable_reasoning_rubric": True,
                "apply_persian_stability_rubric": True,
                "apply_deep_brainstorm_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {"category": category},
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")

    total = len(raw_samples) - skipped
    print(f"نوشته شد: {total} نمونه در {output_path} (رد شده: {skipped})")


def main():
    parser = argparse.ArgumentParser(description="آماده‌سازی داده تسک Deep Brainstorm")
    parser.add_argument("--input", default="data/raw_sources/deep_brainstorm_fa_raw.json")
    parser.add_argument("--output", default="data/tasks/deep_brainstorm_fa.jsonl")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
