"""
Converts the raw Persian Controllability source file
(data/raw_sources/persian_controllability_fa_raw.json) into the executable
TaskSample format consumed by the shared pipeline (src/pipeline/runner.py).

Each sample carries a LIST of constraints under extra.constraints (unlike
IFEval-lite's single extra.constraint), since this task tests whether a
model can satisfy several Persian-specific stylistic constraints at once.
Constraint types are split between two checking paths in runner.py:
  - rule-checkable (paragraph_count, sentence_structure) -> checked
    directly via src/utils/controllability_utils.py
  - judge-checkable (lexical, register, dialect) -> checked via
    src/judges/persian_controllability_judge.py, along with an
    independent semantic_fidelity score

This task does NOT use gold_answer or verifiable_reasoning -- its
"correctness" concept is the constraint-satisfaction rate + hard-fail
flag, stored in the controllability_* fields on ScoredResult. It does not
use persian_stability either, since responses here are typically short
and single-purpose rather than long CoT chains; the register/lexical
constraints already cover language-quality concerns specific to this task.

Usage:
    python scripts/prepare_persian_controllability_task.py \
        --input data/raw_sources/persian_controllability_fa_raw.json \
        --output data/tasks/persian_controllability_fa.jsonl
"""

import argparse
import json

SYSTEM_PROMPT_FA = (
    "تو یک دستیار فارسی‌زبان هستی که باید دقیقاً طبق درخواست کاربر و با "
    "رعایت کامل قیدهای ذکرشده در آن پاسخ بدهی. مستقیم پاسخ را بنویس، بدون "
    "خط «پاسخ نهایی:» یا هر قالب اضافه دیگر."
)

VALID_CONSTRAINT_TYPES = {"register", "lexical", "sentence_structure", "paragraph_count", "dialect"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(raw_samples):
            item_id = item.get("id")
            prompt = item.get("prompt_fa")
            constraints = item.get("constraints")
            difficulty = item.get("difficulty", "medium")

            if not item_id or not prompt or not constraints or difficulty not in VALID_DIFFICULTIES:
                skipped += 1
                print(f"skipped (malformed sample): index={i}, id={item_id}")
                continue

            bad_types = [c["type"] for c in constraints if c.get("type") not in VALID_CONSTRAINT_TYPES]
            if bad_types:
                skipped += 1
                print(f"skipped (unknown constraint type {bad_types}): index={i}, id={item_id}")
                continue

            task_sample = {
                "sample_id": item_id,
                "task_type": "persian_controllability",
                "problem_fa": prompt,
                "gold_answer": None,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": False,
                "apply_persian_controllability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "extra": {"constraints": constraints, "difficulty": difficulty},
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")

    total = len(raw_samples) - skipped
    print(f"نوشته شد: {total} نمونه در {output_path} (رد شده: {skipped})")


def main():
    parser = argparse.ArgumentParser(description="آماده‌سازی داده تسک Persian Controllability")
    parser.add_argument("--input", default="data/raw_sources/persian_controllability_fa_raw.json")
    parser.add_argument("--output", default="data/tasks/persian_controllability_fa.jsonl")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
