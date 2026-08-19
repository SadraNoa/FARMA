"""
Converts the raw Proverbs source file (data/raw_sources/proverbs_fa_raw.json)
into the executable TaskSample format consumed by the shared pipeline
(src/pipeline/runner.py).

Each sample is a Persian proverb (ضرب‌المثل). The model must pick its
correct meaning/usage out of `--num-options` multiple-choice options.

Distractor generation is *mixed*, unlike Aroozi's pool-only approach:
1. Hand-authored `manual_distractors` from the raw sample itself are used
   first (these are meaningful, plausible-but-wrong meanings).
2. If more distractors are still needed to reach `--num-options`, the
   script falls back to sampling `gold_meaning` values belonging to
   *other* proverbs in the dataset (the "pool"), excluding the current
   sample's own gold meaning.

Because every sample already has exactly one correct meaning, this task
reuses the framework's generic exact-match path (comparing the chosen
option letter against gold_answer) — no judge model or runner change is
required, the same way Aroozi/BBH-lite/Jalali calendar work.

Usage:
    python scripts/prepare_proverbs_task.py \
        --input data/raw_sources/proverbs_fa_raw.json \
        --output data/tasks/proverbs_fa.jsonl \
        --num-options 4 \
        --seed 42
"""

import argparse
import json
import random

SYSTEM_PROMPT_FA = (
    "تو یک دستیار متخصص در ادبیات و فرهنگ عامه‌ی فارسی هستی. برای ضرب‌المثل "
    "داده‌شده، معنی و کاربرد صحیح آن را از میان گزینه‌های زیر پیدا کن. در "
    "صورت نیاز، ابتدا استدلال کوتاهی بنویس و در پایان، دقیقاً یک خط جداگانه "
    "با این قالب بنویس (فقط حرفِ گزینه، بدون هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

# ترتیب حروف گزینه‌ها؛ حداکثر ۶ گزینه پشتیبانی می‌شود
OPTION_LETTERS = ["الف", "ب", "ج", "د", "ه", "و"]

VALID_SUBTYPES = {"meaning_identification"}


def build_pool(raw_samples: list[dict]) -> list[str]:
    """استخر یکتای معناهای درستِ کل دیتاست، برای پرکردن گزینه‌های نادرست
    وقتی manual_distractors کافی نباشند."""
    pool: set[str] = set()
    for item in raw_samples:
        meaning = item.get("gold_meaning")
        if meaning:
            pool.add(meaning)
    return list(pool)


def convert(input_path: str, output_path: str, num_options: int = 4, seed: int = 42):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    if num_options < 2:
        raise ValueError("num_options باید حداقل ۲ باشد.")
    if num_options > len(OPTION_LETTERS):
        raise ValueError(f"num_options نباید بیشتر از {len(OPTION_LETTERS)} باشد.")

    pool = build_pool(raw_samples)

    skipped = 0
    converted = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in raw_samples:
            subtype = item.get("subtype")
            proverb = item.get("proverb")
            gold_meaning = item.get("gold_meaning")
            raw_id = item.get("id")
            manual_distractors = [d for d in (item.get("manual_distractors") or []) if d]

            if (
                subtype not in VALID_SUBTYPES
                or not proverb
                or not gold_meaning
                or raw_id is None
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}")
                continue

            rng = random.Random(f"{seed}-{raw_id}")

            # ۱. اول از manual_distractors استفاده کن (تا سقف num_options - 1)
            manual_meanings = [m for m in manual_distractors if m != gold_meaning]
            rng.shuffle(manual_meanings)
            chosen_distractors = manual_meanings[: num_options - 1]

            # ۲. اگر کافی نبود، از استخر کل دیتاست پر کن
            if len(chosen_distractors) < num_options - 1:
                already = set(chosen_distractors) | {gold_meaning}
                pool_candidates = [m for m in pool if m not in already]
                rng.shuffle(pool_candidates)
                needed = (num_options - 1) - len(chosen_distractors)
                chosen_distractors += pool_candidates[:needed]

            if len(chosen_distractors) < num_options - 1:
                skipped += 1
                print(
                    f"skipped (not enough distractors, manual+pool insufficient): id={raw_id}"
                )
                continue

            options_content = [gold_meaning] + chosen_distractors
            rng.shuffle(options_content)

            letters = OPTION_LETTERS[:num_options]
            options_map = {}
            options_lines = []
            correct_letter = None
            for letter, meaning in zip(letters, options_content):
                options_map[letter] = {"meaning": meaning}
                options_lines.append(f"{letter}) {meaning}")
                if meaning == gold_meaning:
                    correct_letter = letter

            question = (
                f'ضرب‌المثل زیر را در نظر بگیر:\n"{proverb}"\n\n'
                "معنی و کاربرد درست این ضرب‌المثل کدام است؟\n"
                + "\n".join(options_lines)
            )

            task_sample = {
                "sample_id": f"proverb_{raw_id:04d}",
                "task_type": "proverb",
                "problem_fa": question,
                "gold_answer": correct_letter,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "subtype": subtype,
                    "proverb": proverb,
                    "usage_note": item.get("usage_note"),
                    "gold_meaning": gold_meaning,
                    "options": options_map,
                    "correct_option": correct_letter,
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            converted += 1

    print(
        f"{converted} Proverb samples converted and saved to: {output_path} ({skipped} skipped)"
    )
    print(f"تعداد معناهای یکتا در استخر گزینه‌ها: {len(pool)} | تعداد گزینه به ازای هر سوال: {num_options}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-options", type=int, default=4, help="تعداد کل گزینه‌ها (۱ درست + بقیه غلط)")
    parser.add_argument("--seed", type=int, default=42, help="seed برای تولید قطعی و قابل‌تکرار گزینه‌ها")
    args = parser.parse_args()
    convert(args.input, args.output, num_options=args.num_options, seed=args.seed)


if __name__ == "__main__":
    main()
