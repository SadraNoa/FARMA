"""
Converts the raw Multi-Constraint source file
(data/raw_sources/multi_constraint_fa_raw.json) into the executable
TaskSample format consumed by the shared pipeline (src/pipeline/runner.py).

This is the fifth and final planned Phase 3 task. Like IFEval-lite
(Phase 1), there is no gold_answer — each sample instead carries a LIST of
machine-checkable rule constraints under extra.rule_constraints, all of
which must be satisfied simultaneously.

IMPORTANT — why a new extra key instead of reusing IFEval-lite's
extra.constraint or Persian Controllability's extra.constraints:
- IFEval-lite's `extra.constraint` (singular) holds exactly ONE constraint
  dict, not a list — doesn't fit multi-constraint's shape.
- Persian Controllability's `extra.constraints` (plural) is already wired
  in runner.py to a judge-based flow (PersianControllabilityJudge) for its
  judge-checkable constraint types (lexical/register/dialect) plus a
  semantic-fidelity score, and populates controllability_*-specific
  ScoredResult fields. Reusing that key here would incorrectly route
  multi-constraint samples through PersianControllabilityJudge, which
  isn't built for these rule-only constraint types (and would also require
  pydantic to instantiate at all, which this task deliberately avoids
  needing since it's 100% rule-based like IFEval-lite).
- `extra.rule_constraints` is therefore a new, independent key: a runner.py
  branch checks every constraint in the list via the exact same
  `check_constraint` dispatcher IFEval-lite already uses (one call per
  constraint), and `correctness` is 1 only if ALL constraints pass — a
  strictly harder bar than IFEval-lite's single-constraint check. No new
  judge, no schema changes (TaskSample.extra is an open dict), no runner
  changes beyond one additive elif branch.

Usage:
    python scripts/prepare_multi_constraint_task.py \
        --input data/raw_sources/multi_constraint_fa_raw.json \
        --output data/tasks/multi_constraint_fa.jsonl
"""

import argparse
import json

SYSTEM_PROMPT_FA = (
    "تو یک دستیار فارسی‌زبان هستی که دقیقاً طبق دستورالعمل‌های داده‌شده عمل می‌کنی. "
    "دستور زیر شامل چند محدودیت هم‌زمان است که باید همه‌ی آن‌ها را به‌طور کامل رعایت کنی. "
    "فقط متن خواسته‌شده را بنویس، بدون مقدمه، توضیح اضافه، یا هرگونه خط دیگر پیش یا پس از آن."
)

# کل پاسخ همان چیزی است که بررسی می‌شود (تعداد جمله/کلمه/پاراگراف، کلمه‌ی
# شروع/پایان، وجود یا عدم وجود کلمه‌ی خاص)، پس extraction regex کل خروجی
# خام را می‌گیرد، نه یک خط «پاسخ نهایی:».
ANSWER_EXTRACTION_REGEX = r"(?s)(.*)"

VALID_CATEGORIES = {"length_keyword", "length_position", "keyword_position", "triple_combo"}


def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0
    per_category_count: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in raw_samples:
            raw_id = item.get("id")
            category = item.get("category")
            topic = item.get("topic")
            question = item.get("question")
            constraints = item.get("constraints")

            if (
                raw_id is None
                or category not in VALID_CATEGORIES
                or not topic
                or not question
                or not constraints
                or not isinstance(constraints, list)
                or len(constraints) < 2
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}, category={category}")
                continue

            task_sample = {
                "sample_id": f"multi_constraint_{raw_id:04d}",
                "task_type": "multi_constraint",
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
                    "rule_constraints": constraints,
                    "num_constraints": len(constraints),
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            per_category_count[category] = per_category_count.get(category, 0) + 1

    converted = len(raw_samples) - skipped
    print(f"{converted} Multi-Constraint samples converted and saved to: {output_path} ({skipped} skipped)")
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
