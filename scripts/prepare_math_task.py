"""
تبدیل فایل خام ریاضی (خروجی ترجمه‌ی شما با تگ‌های [LATEX_N]) به فرمت
اجرایی TaskSample که پایپ‌لاین اصلی (src/pipeline/runner.py) می‌فهمد.

نکته‌ی مهم: این تسک gold_answer ندارد. هدف تسک، سنجش کیفیت خودِ زنجیره‌ی
استدلال است (با rubric verifiable_reasoning) نه تطبیق دقیق جواب نهایی.
پس هیچ امتیاز correctness ساده‌ای برای این تسک محاسبه نمی‌شود.

استفاده:
    python scripts/prepare_math_task.py \
        --input data/raw_sources/openmathreasoning_200_translated.json \
        --output data/tasks/math_fa.jsonl
"""

import argparse
import json
import re


def fill_latex_placeholders(problem_fa: str, latex_list: list[str]) -> str:
    """جای [LATEX_1], [LATEX_2], ... را با فرمول واقعی (بین $...$) پر می‌کند."""
    def replace(match):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(latex_list):
            return f"${latex_list[idx]}$"
        return match.group(0)

    return re.sub(r"\[LATEX_(\d+)\]", replace, problem_fa)


def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(raw_samples):
            if not item.get("problem_fa") or item.get("error"):
                skipped += 1
                print(f"رد شد (نمونه‌ی خراب/بدون ترجمه): index={i}, error={item.get('error')}")
                continue

            full_problem_fa = fill_latex_placeholders(item["problem_fa"], item.get("latex", []))

            task_sample = {
                "sample_id": f"math_{i:04d}",
                "task_type": "math",
                "problem_fa": full_problem_fa,
                "gold_answer": None,
                "requires_cot_judging": True,
                "apply_verifiable_reasoning_rubric": True,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": (
                    "تو یک ریاضی‌دان دقیق و فارسی‌زبان هستی. مسئله‌ی زیر را به فارسی، "
                    "قدم به قدم و با استدلال کامل حل کن. هیچ گامی را حذف نکن. "
                    "در پایان، دقیقاً یک خط جداگانه با این قالب بنویس:\n"
                    "پاسخ نهایی: <جواب نهایی>"
                ),
                "answer_extraction_regex": r"پاسخ نهایی:\s*(.+)",
                "extra": {
                    "problem_source": item.get("problem_source"),
                    "problem_en": item.get("problem_en"),
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")

    converted = len(raw_samples) - skipped
    print(f"{converted} نمونه‌ی ریاضی تبدیل و ذخیره شد در: {output_path} ({skipped} نمونه رد شد)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
