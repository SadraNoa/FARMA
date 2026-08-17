"""
Converts the raw contradiction & consistency source file
(data/raw_sources/contradiction_consistency_fa_raw.json) into the
executable TaskSample format consumed by the shared pipeline
(src/pipeline/runner.py).

Phase 3 — Contradiction & consistency: standard 3-way Natural Language
Inference (NLI). Each item gives a premise and a hypothesis; the model
must classify their logical relationship as one of:
  - استلزام (entailment)    — the hypothesis must be true if the premise is true
  - تناقض (contradiction)   — the hypothesis and premise cannot both be true
  - خنثی (neutral)          — neither necessarily follows nor conflicts

Data is built from 20 hand-authored premises, each paired with exactly one
entailed, one contradicting, and one neutral hypothesis (60 samples total,
perfectly balanced 20/20/20 across labels) — mirroring the "one family, N
controlled variants" pattern used by script-disambiguation and minimal
pairs, so the label can't be guessed from surface cues of the premise
alone.

Because the label set is small and fixed (unlike Aroozi/MMLU-lite, which
draw distractors from a large pool), the three options are simply the three
relation labels, presented in a per-sample randomized order (so the model
can't exploit a fixed position bias) and scored via the same generic
`gold_answer` exact-match path as every other closed-set task — no runner
changes needed.

Usage:
    python scripts/prepare_contradiction_consistency_task.py \
        --input data/raw_sources/contradiction_consistency_fa_raw.json \
        --output data/tasks/contradiction_consistency_fa.jsonl \
        --seed 42
"""

import argparse
import json
import random

SYSTEM_PROMPT_FA = (
    "تو یک دستیار متخصص در استنتاج منطقی زبان فارسی هستی. دو جمله به تو داده "
    "می‌شود: «جمله‌ی اول» (فرض) و «جمله‌ی دوم» (فرضیه). باید رابطه‌ی منطقیِ "
    "جمله‌ی دوم را نسبت به جمله‌ی اول از میان سه حالت زیر تعیین کنی:\n"
    "- استلزام: اگر جمله‌ی اول درست باشد، جمله‌ی دوم هم حتماً درست است.\n"
    "- تناقض: جمله‌ی اول و دوم نمی‌توانند هر دو همزمان درست باشند.\n"
    "- خنثی: جمله‌ی دوم نه لزوماً از جمله‌ی اول نتیجه می‌شود و نه با آن در "
    "تضاد است؛ ممکن است درست یا نادرست باشد.\n\n"
    "با دقت هر دو جمله را بررسی کن و فقط بر اساس آنچه واقعاً بیان شده "
    "قضاوت کن، نه بر اساس حدس یا اطلاعات بیرونی. در پایان، دقیقاً یک خط "
    "جداگانه با این قالب بنویس (فقط حرفِ گزینه، بدون هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"

OPTION_LETTERS = ["الف", "ب", "ج"]

# ترتیب ثابت برچسب‌ها (خودِ ترتیب نمایش به ازای هر نمونه با seed جابه‌جا می‌شود)
RELATION_LABELS_FA = {
    "entailment": "استلزام (اگر جمله‌ی اول درست باشد، جمله‌ی دوم هم حتماً درست است)",
    "contradiction": "تناقض (جمله‌ی اول و دوم نمی‌توانند هر دو همزمان درست باشند)",
    "neutral": "خنثی (نه لزوماً نتیجه می‌شود، نه در تضاد است)",
}

VALID_SUBTYPES = {"nli_relation"}
VALID_RELATIONS = set(RELATION_LABELS_FA.keys())


def convert(input_path: str, output_path: str, seed: int = 42):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    skipped = 0
    per_relation_count: dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in raw_samples:
            subtype = item.get("subtype")
            premise = item.get("premise")
            hypothesis = item.get("hypothesis")
            gold_relation = item.get("gold_relation")
            raw_id = item.get("id")

            if (
                subtype not in VALID_SUBTYPES
                or not premise
                or not hypothesis
                or gold_relation not in VALID_RELATIONS
                or raw_id is None
            ):
                skipped += 1
                print(f"skipped (malformed sample): id={raw_id}")
                continue

            # ترتیب نمایش سه گزینه به‌صورت قطعی و قابل‌تکرار جابه‌جا می‌شود
            rng = random.Random(f"{seed}-{raw_id}")
            relations = list(RELATION_LABELS_FA.items())
            rng.shuffle(relations)

            options_lines = []
            correct_letter = None
            for letter, (relation_key, label_text) in zip(OPTION_LETTERS, relations):
                options_lines.append(f"{letter}) {label_text}")
                if relation_key == gold_relation:
                    correct_letter = letter

            question = (
                f"جمله‌ی اول: {premise}\n"
                f"جمله‌ی دوم: {hypothesis}\n\n"
                "رابطه‌ی منطقی جمله‌ی دوم نسبت به جمله‌ی اول کدام است؟\n"
                + "\n".join(options_lines)
            )

            task_sample = {
                "sample_id": f"contradiction_consistency_{raw_id:04d}",
                "task_type": "contradiction_consistency",
                "problem_fa": question,
                "gold_answer": correct_letter,
                "requires_cot_judging": False,
                "apply_verifiable_reasoning_rubric": False,
                "apply_persian_stability_rubric": True,
                "system_prompt_fa": SYSTEM_PROMPT_FA,
                "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
                "extra": {
                    "subtype": subtype,
                    "premise_group": item.get("premise_group"),
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "gold_relation": gold_relation,
                    "correct_option": correct_letter,
                    "raw_id": raw_id,
                },
            }
            out_f.write(json.dumps(task_sample, ensure_ascii=False) + "\n")
            per_relation_count[gold_relation] = per_relation_count.get(gold_relation, 0) + 1

    converted = len(raw_samples) - skipped
    print(f"{converted} contradiction/consistency samples converted and saved to: {output_path} ({skipped} skipped)")
    for relation, count in sorted(per_relation_count.items()):
        print(f"  {relation}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed برای جابه‌جایی قطعی و قابل‌تکرار ترتیب گزینه‌ها")
    args = parser.parse_args()
    convert(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
