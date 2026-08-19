"""
Converts the raw Script Disambiguation source file
(data/raw_sources/script_disambiguation_fa_raw.json) into the executable
TaskSample format consumed by the shared pipeline (src/pipeline/runner.py).

Each sample is a Persian sentence containing a homograph (a word written
identically without diacritics but with more than one possible
reading/meaning, e.g. "کرم" = کِرم/کَرَم). The model must pick the correct
(reading, meaning) pair for that word in that specific sentence out of a
small set of multiple-choice options.

IMPORTANT — distractor sourcing is *same-word only*:
Unlike Aroozi/Proverbs (which draw fallback distractors from a dataset-wide
pool), a distractor here is only meaningful if it is an alternate reading
of the *same* ambiguous word. Mixing in an unrelated word's reading (e.g.
offering "دَوْر — نوبت" as a wrong option for the word "سحر") makes the
question trivially easy and doesn't test disambiguation at all. So:

1. A per-word pool is built by grouping every (reading, meaning) pair —
   both each sample's own gold pair and its `manual_distractors` — by
   `ambiguous_word` across the whole raw file.
2. For a given sample, distractor candidates are drawn *only* from that
   word's own pool (excluding the sample's own gold pair).
3. The number of options is therefore **variable per sample**: most words
   in this dataset have exactly 2 known senses (2-option MCQ); a few (e.g.
   "ملک" with 3 senses) yield 3-option MCQ. `--max-options` caps the upper
   bound (default 4) in case future data has more senses for one word;
   `--min-options` (default 2) is the minimum required for a sample to be
   usable — samples whose word has no known alternate reading are skipped.

Because every sample already has exactly one correct (reading, meaning)
pair, this task reuses the framework's generic exact-match path (comparing
the chosen option letter against gold_answer) — no judge model or runner
change is required, the same way Aroozi/BBH-lite/Jalali calendar work.

Usage:
    python scripts/prepare_script_disambiguation_task.py \
        --input data/raw_sources/script_disambiguation_fa_raw.json \
        --output data/tasks/script_disambiguation_fa.jsonl \
        --max-options 4 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict

SYSTEM_PROMPT_FA = (
    "تو یک دستیار متخصص در زبان و ادبیات فارسی هستی. در جمله‌ی داده‌شده یک "
    "کلمه به‌صورت مبهم (بدون اعراب) آمده که بیش از یک تلفظ/معنی ممکن دارد. "
    "با توجه به بافت جمله، تلفظ و معنی درست آن کلمه را از میان گزینه‌های زیر "
    "پیدا کن. در صورت نیاز، ابتدا استدلال کوتاهی بنویس و در پایان، دقیقاً یک "
    "خط جداگانه با این قالب بنویس (فقط حرفِ گزینه، بدون هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

# ترتیب حروف گزینه‌ها؛ حداکثر ۶ گزینه پشتیبانی می‌شود
OPTION_LETTERS = ["الف", "ب", "ج", "د", "ه", "و"]

VALID_SUBTYPES = {"homograph_disambiguation"}


def format_option(reading: str, meaning: str) -> str:
    return f"{reading} — {meaning}"


def build_word_pool(raw_samples: list[dict]) -> dict[str, dict[str, str]]:
    """برای هر ambiguous_word، نگاشتِ یکتای {تلفظ: معنیِ کانونی} را از گلد و
    manual_distractors همه‌ی نمونه‌های همان کلمه در کل دیتاست جمع می‌کند.
    این تنها منبع گزینه‌های نادرست است — هیچ ترکیبی با کلمات دیگر رخ
    نمی‌دهد.

    یکتاسازی روی خودِ «تلفظ» انجام می‌شود، نه روی جفت (تلفظ، معنی):
    چون عبارت‌پردازیِ معنی یک خوانش ممکن است در نمونه‌های مختلف کمی متفاوت
    نوشته شده باشد (مثلاً «داور یا فردی که...» در یک نمونه و «داور» در
    نمونه‌ای دیگر برای همان خوانش «حَکَم»)، دو نگارش متفاوت از یک خوانش
    نباید به‌عنوان دو گزینه‌ی جداگانه در یک سوال ظاهر شوند. برای هر تلفظ،
    اولویت با معنیِ نوشته‌شده در جایی است که آن تلفظ خودِ gold_reading بوده
    (چون معمولاً کامل‌تر/دقیق‌تر نوشته شده)؛ در غیر این صورت از عبارتِ
    آمده در manual_distractors استفاده می‌شود."""
    pool: dict[str, dict[str, str]] = defaultdict(dict)
    gold_readings: dict[str, set[str]] = defaultdict(set)

    # پاس اول: معنیِ کانونی را از جاهایی که تلفظ به‌عنوان gold آمده می‌گیریم
    for item in raw_samples:
        word = item.get("ambiguous_word")
        gold_reading = item.get("gold_reading")
        gold_meaning = item.get("gold_meaning")
        if word and gold_reading and gold_meaning:
            pool[word][gold_reading] = gold_meaning
            gold_readings[word].add(gold_reading)

    # پاس دوم: تلفظ‌هایی که هرگز gold نبوده‌اند را از manual_distractors پر می‌کنیم
    for item in raw_samples:
        word = item.get("ambiguous_word")
        if not word:
            continue
        for d in item.get("manual_distractors") or []:
            reading, meaning = d.get("reading"), d.get("meaning")
            if reading and meaning and reading not in gold_readings[word]:
                pool[word].setdefault(reading, meaning)

    return pool


def convert(
    input_path: str,
    output_path: str,
    max_options: int = 4,
    min_options: int = 2,
    seed: int = 42,
):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    if min_options < 2:
        raise ValueError("min_options باید حداقل ۲ باشد.")
    if max_options > len(OPTION_LETTERS):
        raise ValueError(f"max_options نباید بیشتر از {len(OPTION_LETTERS)} باشد.")
    if max_options < min_options:
        raise ValueError("max_options نباید کمتر از min_options باشد.")

    word_pool = build_word_pool(raw_samples)

    skipped = 0
    converted = 0
    option_count_histogram: dict[int, int] = defaultdict(int)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in raw_samples:
            subtype = item.get("subtype")
            sentence = item.get("sentence")
            ambiguous_word = item.get("ambiguous_word")
            gold_reading = item.get("gold_reading")
            gold_meaning = item.get("gold_meaning")
            raw_id = item.get("id")

            if (
                subtype not in VALID_SUBTYPES
                or not sentence
                or not ambiguous_word
                or not gold_reading
                or not gold_meaning
                or raw_id is None
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}")
                continue

            rng = random.Random(f"{seed}-{raw_id}")

            gold_pair = (gold_reading, gold_meaning)

            # فقط از استخر مخصوصِ همین کلمه (same-word only) — بدون هیچ
            # ترکیبی با کلمات دیگر دیتاست. استخر اکنون بر اساس تلفظِ یکتا
            # ساخته شده، پس معنیِ کانونیِ همان استخر را برای گلدِ همین نمونه
            # هم به‌کار می‌بریم تا با بقیه‌ی گزینه‌ها سازگار بماند.
            word_readings = word_pool.get(ambiguous_word, {})
            canonical_gold_meaning = word_readings.get(gold_reading, gold_meaning)
            gold_pair = (gold_reading, canonical_gold_meaning)
            same_word_candidates = [
                (reading, meaning) for reading, meaning in word_readings.items() if reading != gold_reading
            ]
            rng.shuffle(same_word_candidates)
            chosen_distractors = same_word_candidates[: max_options - 1]

            num_options = 1 + len(chosen_distractors)
            if num_options < min_options:
                skipped += 1
                print(
                    f"skipped (fewer than {min_options} known senses for "
                    f"'{ambiguous_word}'): id={raw_id}"
                )
                continue

            options_content = [gold_pair] + chosen_distractors
            rng.shuffle(options_content)

            letters = OPTION_LETTERS[:num_options]
            options_map = {}
            options_lines = []
            correct_letter = None
            for letter, (reading, meaning) in zip(letters, options_content):
                options_map[letter] = {"reading": reading, "meaning": meaning}
                options_lines.append(f"{letter}) {format_option(reading, meaning)}")
                if (reading, meaning) == gold_pair:
                    correct_letter = letter

            context_hint = item.get("context_hint") or ""
            hint_line = f"\n{context_hint}" if context_hint else ""

            question = (
                f'جمله‌ی زیر را در نظر بگیر:\n"{sentence}"\n\n'
                f'کلمه‌ی مبهم: «{ambiguous_word}»{hint_line}\n\n'
                "تلفظ و معنی درست این کلمه در این جمله کدام است؟\n"
                + "\n".join(options_lines)
            )

            task_sample = {
                "sample_id": f"script_disambiguation_{raw_id:04d}",
                "task_type": "script_disambiguation",
                "problem_fa": question,
                "gold_answer": correct_letter,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "subtype": subtype,
                    "sentence": sentence,
                    "ambiguous_word": ambiguous_word,
                    "context_hint": item.get("context_hint"),
                    "gold_reading": gold_reading,
                    "gold_meaning": gold_meaning,
                    "options": options_map,
                    "correct_option": correct_letter,
                    "num_options": num_options,
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            converted += 1
            option_count_histogram[num_options] += 1

    print(
        f"{converted} Script Disambiguation samples converted and saved to: "
        f"{output_path} ({skipped} skipped)"
    )
    print(f"تعداد کلمات یکتا در استخر: {len(word_pool)}")
    print("توزیع تعداد گزینه‌ها در سوالات تولیدشده:")
    for k in sorted(option_count_histogram):
        print(f"  {k} گزینه: {option_count_histogram[k]} سوال")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-options", type=int, default=4, help="حداکثر تعداد گزینه (۱ درست + حداکثر این تعداد منهای ۱ گزینه‌ی نادرست)"
    )
    parser.add_argument(
        "--min-options", type=int, default=2, help="حداقل تعداد گزینه‌ی لازم برای اینکه نمونه معتبر باشد"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed برای تولید قطعی و قابل‌تکرار گزینه‌ها")
    args = parser.parse_args()
    convert(
        args.input,
        args.output,
        max_options=args.max_options,
        min_options=args.min_options,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
