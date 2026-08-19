"""
Converts the raw Ieham (wordplay/double-entendre) source file
(data/raw_sources/ieham_fa_raw.json) into the executable TaskSample format
consumed by the shared pipeline (src/pipeline/runner.py).

Each sample is a classical Persian verse (بیت) containing ایهام — a word
with two simultaneously valid meanings (a "near"/obvious one and a
"far"/subtle one that carries the poetic point). The model must pick the
correct (near, far) meaning *pair* for the ambiguous word out of
`--num-options` multiple-choice options; each option renders as
"نزدیک: ... / دور: ...".

Distractor generation is *mixed*, unlike Aroozi's pool-only approach:
1. Hand-authored `manual_distractors` from the raw sample itself are used
   first (these are meaningful, deliberately-plausible wrong pairs).
2. If more distractors are still needed to reach `--num-options`, the
   script falls back to sampling (near, far) pairs belonging to *other*
   samples in the dataset (the "pool"), excluding the current sample's own
   gold pair.

Because every sample already has exactly one correct (near, far) pair,
this task reuses the framework's generic exact-match path (comparing the
chosen option letter against gold_answer) — no judge model or runner
change is required, the same way Aroozi/BBH-lite/Jalali calendar work.

Usage:
    python scripts/prepare_ieham_task.py \
        --input data/raw_sources/ieham_fa_raw.json \
        --output data/tasks/ieham_fa.jsonl \
        --num-options 4 \
        --seed 42
"""

import argparse
import json
import random

SYSTEM_PROMPT_FA = (
    "تو یک دستیار متخصص در بلاغت و ادبیات کلاسیک فارسی هستی، به‌ویژه در "
    "شناخت آرایه‌ی ایهام. در بیت داده‌شده یک کلمه به‌کار رفته که دارای دو "
    "معنای هم‌زمان است: یک معنای نزدیک (ظاهری) و یک معنای دور (باریک و "
    "هنری) که نکته‌ی اصلی ایهام را می‌سازد. جفتِ درستِ این دو معنا را از "
    "میان گزینه‌های زیر پیدا کن. در صورت نیاز، ابتدا استدلال کوتاهی بنویس "
    "و در پایان، دقیقاً یک خط جداگانه با این قالب بنویس (فقط حرفِ گزینه، "
    "بدون هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

# ترتیب حروف گزینه‌ها؛ حداکثر ۶ گزینه پشتیبانی می‌شود
OPTION_LETTERS = ["الف", "ب", "ج", "د", "ه", "و"]

VALID_SUBTYPES = {"double_meaning_identification"}


def format_option(near: str, far: str) -> str:
    return f"نزدیک: {near} / دور: {far}"


def build_pool(raw_samples: list[dict]) -> list[tuple[str, str]]:
    """استخر یکتای (معنای نزدیک، معنای دور) از کل دیتاست، برای پرکردن
    گزینه‌های نادرست وقتی manual_distractors کافی نباشند."""
    pool: set[tuple[str, str]] = set()
    for item in raw_samples:
        near = item.get("gold_meaning_near")
        far = item.get("gold_meaning_far")
        if near and far:
            pool.add((near, far))
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
            verse = item.get("verse")
            ambiguous_word = item.get("ambiguous_word")
            gold_near = item.get("gold_meaning_near")
            gold_far = item.get("gold_meaning_far")
            raw_id = item.get("id")
            manual_distractors = item.get("manual_distractors") or []

            if (
                subtype not in VALID_SUBTYPES
                or not verse
                or not ambiguous_word
                or not gold_near
                or not gold_far
                or raw_id is None
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}")
                continue

            rng = random.Random(f"{seed}-{raw_id}")

            gold_pair = (gold_near, gold_far)

            # ۱. اول از manual_distractors استفاده کن (تا سقف num_options - 1)
            manual_pairs = [
                (d["near"], d["far"])
                for d in manual_distractors
                if d.get("near") and d.get("far") and (d["near"], d["far"]) != gold_pair
            ]
            rng.shuffle(manual_pairs)
            chosen_distractors = manual_pairs[: num_options - 1]

            # ۲. اگر کافی نبود، از استخر کل دیتاست پر کن
            if len(chosen_distractors) < num_options - 1:
                already = set(chosen_distractors) | {gold_pair}
                pool_candidates = [p for p in pool if p not in already]
                rng.shuffle(pool_candidates)
                needed = (num_options - 1) - len(chosen_distractors)
                chosen_distractors += pool_candidates[:needed]

            if len(chosen_distractors) < num_options - 1:
                skipped += 1
                print(
                    f"skipped (not enough distractors, manual+pool insufficient): id={raw_id}"
                )
                continue

            options_content = [gold_pair] + chosen_distractors
            rng.shuffle(options_content)

            letters = OPTION_LETTERS[:num_options]
            options_map = {}
            options_lines = []
            correct_letter = None
            for letter, (near, far) in zip(letters, options_content):
                options_map[letter] = {"near": near, "far": far}
                options_lines.append(f"{letter}) {format_option(near, far)}")
                if (near, far) == gold_pair:
                    correct_letter = letter

            question = (
                f'بیت زیر را در نظر بگیر:\n"{verse}"\n\n'
                f'کلمه‌ی ایهامی: «{ambiguous_word}»\n\n'
                "جفتِ درستِ معنای نزدیک و معنای دورِ این کلمه در این بیت کدام است؟\n"
                + "\n".join(options_lines)
            )

            task_sample = {
                "sample_id": f"ieham_{raw_id:04d}",
                "task_type": "ieham",
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
                    "ambiguous_word": ambiguous_word,
                    "poet": item.get("poet"),
                    "source": item.get("source"),
                    "gold_meaning_near": gold_near,
                    "gold_meaning_far": gold_far,
                    "options": options_map,
                    "correct_option": correct_letter,
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            converted += 1

    print(
        f"{converted} Ieham samples converted and saved to: {output_path} ({skipped} skipped)"
    )
    print(f"تعداد جفت‌های یکتا در استخر گزینه‌ها: {len(pool)} | تعداد گزینه به ازای هر سوال: {num_options}")


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
