"""
Converts the raw Aroozi (Persian classical-meter identification) source
file (data/raw_sources/aroozi_meter_fa_raw.json) into the executable
TaskSample format consumed by the shared pipeline (src/pipeline/runner.py).

Each verse is turned into a multiple-choice question: the model must pick
the correct عروضی (prosodic) meter out of `--num-options` options (1
correct + the rest sampled as distractors from the *other* meters present
in the dataset). Because every sample already has exactly one correct
meter drawn from a small closed set, this task reuses the framework's
generic exact-match path (comparing the chosen option letter against
gold_answer) — no judge model or runner change is required, the same way
BBH-lite and Jalali calendar work.

Usage:
    python scripts/prepare_aroozi_task.py \
        --input data/raw_sources/aroozi_meter_fa_raw.json \
        --output data/tasks/aroozi_meter_fa.jsonl \
        --num-options 4 \
        --seed 42
"""

import argparse
import json
import random

SYSTEM_PROMPT_FA = (
    "تو یک دستیار متخصص در عروض کلاسیک فارسی هستی. برای بیت داده‌شده، وزن "
    "عروضی صحیح آن را از میان گزینه‌های زیر پیدا کن. در صورت نیاز، ابتدا با "
    "تقطیع کوتاهی از بیت استدلال کن و در پایان، دقیقاً یک خط جداگانه با این "
    "قالب بنویس (فقط حرفِ گزینه، بدون هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

# ترتیب حروف گزینه‌ها؛ حداکثر ۶ گزینه پشتیبانی می‌شود
OPTION_LETTERS = ["الف", "ب", "ج", "د", "ه", "و"]

VALID_SUBTYPES = {"meter_identification"}


def build_meter_pool(raw_samples: list[dict]) -> dict[str, str]:
    """استخر یکتای {نام وزن: الگوی وزن} را از کل دیتاست جمع می‌کند تا
    گزینه‌های نادرست (distractor) از میان اوزانِ واقعاً موجود در دیتاست
    انتخاب شوند، نه اوزان دلخواه/ساختگی."""
    pool: dict[str, str] = {}
    for item in raw_samples:
        name = item.get("gold_meter_name")
        pattern = item.get("gold_meter_pattern")
        if name and pattern:
            pool[name] = pattern
    return pool


def format_option(name: str, pattern: str) -> str:
    return f"{name} ({pattern})"


def convert(input_path: str, output_path: str, num_options: int = 4, seed: int = 42):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    if num_options < 2:
        raise ValueError("num_options باید حداقل ۲ باشد.")
    if num_options > len(OPTION_LETTERS):
        raise ValueError(f"num_options نباید بیشتر از {len(OPTION_LETTERS)} باشد.")

    meter_pool = build_meter_pool(raw_samples)
    if len(meter_pool) < num_options:
        raise ValueError(
            f"تعداد اوزان یکتای موجود در دیتاست ({len(meter_pool)}) کمتر از "
            f"num_options ({num_options}) است — یا num_options را کم کن یا "
            "دیتای خام را کامل کن."
        )

    skipped = 0
    per_meter_count: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in raw_samples:
            subtype = item.get("subtype")
            verse = item.get("verse")
            gold_name = item.get("gold_meter_name")
            gold_pattern = item.get("gold_meter_pattern")
            raw_id = item.get("id")

            if (
                subtype not in VALID_SUBTYPES
                or not verse
                or not gold_name
                or not gold_pattern
                or raw_id is None
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}")
                continue

            # RNG قطعی و مستقل به ازای هر نمونه (وابسته به seed کلی + id نمونه)
            # تا تولید گزینه‌ها بین اجراهای مختلف پایدار و قابل تکرار بماند.
            rng = random.Random(f"{seed}-{raw_id}")

            distractor_names = [n for n in meter_pool if n != gold_name]
            rng.shuffle(distractor_names)
            chosen_distractors = distractor_names[: num_options - 1]

            options_content = [(gold_name, gold_pattern)] + [
                (n, meter_pool[n]) for n in chosen_distractors
            ]
            rng.shuffle(options_content)

            letters = OPTION_LETTERS[:num_options]
            options_map = {}
            options_lines = []
            correct_letter = None
            for letter, (name, pattern) in zip(letters, options_content):
                options_map[letter] = {"meter_name": name, "meter_pattern": pattern}
                options_lines.append(f"{letter}) {format_option(name, pattern)}")
                if name == gold_name:
                    correct_letter = letter

            question = (
                f'بیت زیر را از نظر وزن عروضی بررسی کن:\n"{verse}"\n\n'
                "وزن این بیت کدام است؟\n" + "\n".join(options_lines)
            )

            task_sample = {
                "sample_id": f"aroozi_meter_{raw_id:04d}",
                "task_type": "aroozi",
                "problem_fa": question,
                "gold_answer": correct_letter,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "subtype": subtype,
                    "verse": verse,
                    "poet": item.get("poet"),
                    "source": item.get("source"),
                    "gold_meter_name": gold_name,
                    "gold_meter_pattern": gold_pattern,
                    "options": options_map,
                    "correct_option": correct_letter,
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            per_meter_count[gold_name] = per_meter_count.get(gold_name, 0) + 1

    converted = len(raw_samples) - skipped
    print(f"{converted} Aroozi samples converted and saved to: {output_path} ({skipped} skipped)")
    print(f"تعداد اوزان یکتا در استخر گزینه‌ها: {len(meter_pool)} | تعداد گزینه به ازای هر سوال: {num_options}")
    for name, count in sorted(per_meter_count.items()):
        print(f"  {name}: {count}")


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
