"""
Converts the raw Ieham (wordplay/double-entendre) source file
(data/raw_sources/ieham_fa_raw.json) into the executable TaskSample format
consumed by the shared pipeline (src/pipeline/runner.py).

Each sample is a classical Persian verse (بیت) containing ایهام — a word
with two simultaneously valid meanings (a "near"/obvious one and a
"far"/subtle one that carries the poetic point). The model must pick the
correct (near, far) meaning *pair* for the ambiguous word out of a small
set of multiple-choice options; each option renders as "نزدیک: ... / دور:
...".

IMPORTANT — distractor sourcing is *same-word only*, same fix applied to
Script Disambiguation:
A wrong (near, far) pair is only meaningful if it is an alternate reading
of the *same* ambiguous word in the dataset. Mixing in another word's
meaning pair (e.g. offering "نزدیک: گورخر / دور: قبر" — the pair for
"گور" — as a wrong option for the word "مشتری") would be trivially
rejectable without any real understanding of *this* verse's wordplay,
since it plainly doesn't fit the word at all. So:

1. A per-word pool is built by grouping every (near, far) pair — both each
   sample's own gold pair and its `manual_distractors` — by
   `ambiguous_word` across the whole raw file.
2. For a given sample, distractor candidates are drawn *only* from that
   word's own pool (excluding the sample's own gold pair).
3. The number of options is therefore **variable per sample**: most words
   in this dataset appear in only one sample (2 manual_distractors + gold
   = 3-option MCQ); a few repeated words (e.g. "شیرین", "مشتری", "مهر")
   can yield more options. `--max-options` caps the upper bound (default
   4); `--min-options` (default 2) is the minimum required for a sample to
   be usable.

Deduplication is keyed on the **far meaning** (not on the (near, far) pair
as a whole), the same fix applied in Script Disambiguation for the
`reading`: the far meaning is the specific, named referent of the wordplay
(e.g. "شیرین، معشوقهٔ فرهاد", "خورشید", "سیارهٔ مشتری") and stays
consistent across repeated samples for a word, while the near meaning is a
descriptive gloss that gets paraphrased slightly differently from one
sample to the next (e.g. "خوشایند و دلپذیر" vs. "دلپذیر و خوشایند" for
شیرین). Without this, two reworded copies of the same underlying sense
could appear as two different options in one question. The canonical near
phrasing for a given far meaning is taken from wherever that far meaning
is a sample's own `gold_meaning_far` (preferring the more deliberate
phrasing over a shorter `manual_distractors` entry).

Because every sample already has exactly one correct (near, far) pair,
this task reuses the framework's generic exact-match path (comparing the
chosen option letter against gold_answer) — no judge model or runner
change is required, the same way Aroozi/Script Disambiguation/Proverbs
work.

Usage:
    python scripts/prepare_ieham_task.py \
        --input data/raw_sources/ieham_fa_raw.json \
        --output data/tasks/ieham_fa.jsonl \
        --max-options 4 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict

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


def build_word_pool(raw_samples: list[dict]) -> dict[str, dict[str, str]]:
    """برای هر ambiguous_word، نگاشتِ یکتای {معنای دور: معنای نزدیکِ کانونی}
    را از گلد و manual_distractors همه‌ی نمونه‌های همان کلمه در کل دیتاست
    جمع می‌کند. این تنها منبع گزینه‌های نادرست است — هیچ ترکیبی با کلمات
    دیگر رخ نمی‌دهد.

    یکتاسازی روی «معنای دور» انجام می‌شود، نه روی جفت (نزدیک، دور): چون
    معنای دور همان مرجعِ خاص و نام‌دارِ ایهام است (مثلاً «شیرین، معشوقهٔ
    فرهاد» یا «خورشید») و در نمونه‌های تکراری یک کلمه ثابت می‌ماند، در
    حالی که معنای نزدیک صرفاً توضیحی است که ممکن است هر بار کمی متفاوت
    عبارت‌پردازی شده باشد. برای هر معنای دور، اولویت با نزدیکِ نوشته‌شده
    در جایی است که آن دور خودِ gold_meaning_far بوده."""
    pool: dict[str, dict[str, str]] = defaultdict(dict)
    gold_fars: dict[str, set[str]] = defaultdict(set)

    # پاس اول: نزدیکِ کانونی را از جاهایی که دور به‌عنوان gold آمده می‌گیریم
    for item in raw_samples:
        word = item.get("ambiguous_word")
        far = item.get("gold_meaning_far")
        near = item.get("gold_meaning_near")
        if word and far and near:
            pool[word][far] = near
            gold_fars[word].add(far)

    # پاس دوم: دورهایی که هرگز gold نبوده‌اند را از manual_distractors پر می‌کنیم
    for item in raw_samples:
        word = item.get("ambiguous_word")
        if not word:
            continue
        for d in item.get("manual_distractors") or []:
            near, far = d.get("near"), d.get("far")
            if near and far and far not in gold_fars[word]:
                pool[word].setdefault(far, near)

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
            verse = item.get("verse")
            ambiguous_word = item.get("ambiguous_word")
            gold_near = item.get("gold_meaning_near")
            gold_far = item.get("gold_meaning_far")
            raw_id = item.get("id")

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

            # فقط از استخر مخصوصِ همین کلمه (same-word only) — بدون هیچ
            # ترکیبی با کلمات دیگر دیتاست. نزدیکِ کانونیِ همان استخر را
            # برای گلدِ همین نمونه هم به‌کار می‌بریم تا با بقیه‌ی گزینه‌ها
            # سازگار بماند.
            word_fars = word_pool.get(ambiguous_word, {})
            canonical_gold_near = word_fars.get(gold_far, gold_near)
            gold_pair = (canonical_gold_near, gold_far)
            same_word_candidates = [
                (near, far) for far, near in word_fars.items() if far != gold_far
            ]
            rng.shuffle(same_word_candidates)
            chosen_distractors = same_word_candidates[: max_options - 1]

            num_options = 1 + len(chosen_distractors)
            if num_options < min_options:
                skipped += 1
                print(
                    f"skipped (fewer than {min_options} known meaning-pairs for "
                    f"'{ambiguous_word}'): id={raw_id}"
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
                    "gold_meaning_near": canonical_gold_near,
                    "gold_meaning_far": gold_far,
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
        f"{converted} Ieham samples converted and saved to: {output_path} ({skipped} skipped)"
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
