"""
Converts the raw abstention source file
(data/raw_sources/abstention_fa_raw.json) into the executable TaskSample
format consumed by the shared pipeline (src/pipeline/runner.py).

Phase 3 — Abstention: tests whether a model correctly recognizes the
*limits* of what it can answer, rather than always confidently producing
some answer. Each of 18 short passages is paired with three questions:

  - answerable            — the passage directly contains the answer.
  - insufficient_info     — the question is on-topic but the passage
                             simply doesn't state the fact asked for.
  - false_premise         — the question presupposes something the
                             passage explicitly contradicts.

For every question (regardless of type), the model is shown the SAME four
candidate options: the real factual answer, a plausible-but-wrong factual
foil, a fixed "insufficient information" statement, and a fixed "false
premise" statement. Which of the four is correct depends only on the
question's type — this is what makes the task a genuine test of
abstention rather than reading comprehension: a model that just pattern
matches "pick the answer-shaped option" will get the insufficient-info and
false-premise questions wrong, and a model that pattern matches "always
hedge" will get the answerable questions wrong.

Because gold_answer is always one of four fixed-position options (letter),
this reuses the framework's generic exact-match path — no runner changes
needed, the same way Aroozi/MMLU-lite/minimal-pairs/contradiction-
consistency work.

Usage:
    python scripts/prepare_abstention_task.py \
        --input data/raw_sources/abstention_fa_raw.json \
        --output data/tasks/abstention_fa.jsonl \
        --seed 42
"""

import argparse
import json
import random

SYSTEM_PROMPT_FA = (
    "تو یک دستیار دقیق و محتاط هستی. یک متن کوتاه و یک سؤال درباره‌ی آن به "
    "تو داده می‌شود. باید تشخیص بدهی که پاسخ صحیح کدام‌یک از حالت‌های زیر "
    "است:\n"
    "۱) اگر متن، پاسخ دقیق سؤال را بیان کرده، همان پاسخ واقعی را انتخاب کن.\n"
    "۲) اگر سؤال به موضوع متن مربوط است ولی متن اطلاعات لازم برای پاسخ را "
    "بیان نکرده، گزینه‌ی «اطلاعات کافی در متن داده نشده است.» را انتخاب کن "
    "— حتی اگر بتوانی حدس بزنی، حدس نزن.\n"
    "۳) اگر سؤال بر پیش‌فرضی نادرست بنا شده که با متن در تضاد است، گزینه‌ی "
    "«پیش‌فرض این سؤال نادرست است.» را انتخاب کن.\n"
    "هرگز بر اساس دانش عمومی یا حدس، به سؤالی که متن پاسخش را نمی‌دهد، پاسخ "
    "قطعی نده. در پایان، دقیقاً یک خط جداگانه با این قالب بنویس (فقط حرفِ "
    "گزینه، بدون هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

OPTION_LETTERS = ["الف", "ب", "ج", "د"]

INSUFFICIENT_INFO_TEXT = "اطلاعات کافی در متن داده نشده است."
FALSE_PREMISE_TEXT = "پیش‌فرض این سؤال نادرست است."

VALID_SUBTYPES = {"answerable", "insufficient_info", "false_premise"}


def convert(input_path: str, output_path: str, seed: int = 42):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0
    per_subtype_count: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in raw_samples:
            subtype = item.get("subtype")
            passage = item.get("passage")
            question = item.get("question")
            real_answer = item.get("real_answer")
            foil_answer = item.get("foil_answer")
            raw_id = item.get("id")

            if (
                subtype not in VALID_SUBTYPES
                or not passage
                or not question
                or not real_answer
                or not foil_answer
                or real_answer == foil_answer
                or raw_id is None
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}")
                continue

            # چهار گزینه همیشه یکسان‌اند؛ فقط اینکه کدام‌یک «درست» است بسته
            # به subtype فرق می‌کند.
            option_contents = {
                "answerable": real_answer,
                "insufficient_info": INSUFFICIENT_INFO_TEXT,
                "false_premise": FALSE_PREMISE_TEXT,
                "foil": foil_answer,
            }

            # ترتیب نمایش گزینه‌ها به‌صورت قطعی و قابل‌تکرار جابه‌جا می‌شود
            rng = random.Random(f"{seed}-{raw_id}")
            option_keys = list(option_contents.keys())
            rng.shuffle(option_keys)

            options_lines = []
            correct_letter = None
            for letter, key in zip(OPTION_LETTERS, option_keys):
                options_lines.append(f"{letter}) {option_contents[key]}")
                if key == subtype:
                    correct_letter = letter

            question_block = (
                f"متن: {passage}\n"
                f"سؤال: {question}\n\n"
                "کدام گزینه پاسخ درستِ این سؤال است؟\n"
                + "\n".join(options_lines)
            )

            task_sample = {
                "sample_id": f"abstention_{raw_id:04d}",
                "task_type": "abstention",
                "problem_fa": question_block,
                "gold_answer": correct_letter,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "subtype": subtype,
                    "family_id": item.get("family_id"),
                    "passage": passage,
                    "question": question,
                    "real_answer": real_answer,
                    "foil_answer": foil_answer,
                    "correct_option": correct_letter,
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            per_subtype_count[subtype] = per_subtype_count.get(subtype, 0) + 1

    converted = len(raw_samples) - skipped
    print(f"{converted} abstention samples converted and saved to: {output_path} ({skipped} skipped)")
    for subtype, count in sorted(per_subtype_count.items()):
        print(f"  {subtype}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed برای جابه‌جایی قطعی و قابل‌تکرار ترتیب گزینه‌ها")
    args = parser.parse_args()
    convert(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
