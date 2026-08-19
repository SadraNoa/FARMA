# Phase 2 — Proverbs (ضرب‌المثل) Task

## Overview

This task evaluates a model's understanding of Persian proverbs
(ضرب‌المثل). Each sample gives a proverb and asks the model to pick its
correct meaning/usage out of 4 multiple-choice options — the same
exact-match-friendly design as Aroozi and MMLU-lite.

## Data

- **Source:** hand-authored, 49 unique samples (an accidental duplicate
  entry — two phrasings of «خواهی نشوی رسوا، همرنگ جماعت شو» — was
  removed before finalizing).
- **Subtype:** all samples are `meaning_identification`.

### Raw format (`data/raw_sources/proverbs_fa_raw.json`)

```json
{
  "id": 1,
  "task_type": "proverb",
  "subtype": "meaning_identification",
  "proverb": "کاچی به از هیچی",
  "gold_meaning": "داشتن چیزی کم یا ناقص، بهتر از نداشتن آن است؛ در نبود گزینه بهتر، باید ارزش داشته‌های اندک را دانست.",
  "manual_distractors": [
    "برای رسیدن به موفقیت بزرگ باید همیشه از امکانات کوچک صرف‌نظر کرد.",
    "کارهای عجولانه معمولاً نتیجه مطلوبی ندارند."
  ],
  "usage_note": null
}
```

### Task-ready format (`data/tasks/proverbs_fa.jsonl`)

Produced by `scripts/prepare_proverbs_task.py`, 4 options per question
(`--num-options 4`).

## Why distractors use a dataset-wide pool (unlike Script Disambiguation)

Unlike homographs (Script Disambiguation), a proverb's meaning is
self-contained — any other proverb's `gold_meaning` in the dataset is a
plausible-sounding *wrong* answer for a given proverb, since the two
proverbs are about unrelated situations. So `prepare_proverbs_task.py`
mirrors Aroozi's mixed approach:

1. Hand-authored `manual_distractors` (2 per sample in this dataset) are
   used first.
2. The script fills any remaining slots (2 more, to reach 4 options total)
   from a dataset-wide pool of every other sample's `gold_meaning`.

With 49 samples and only 2 manual distractors per item, most questions end
up mixing hand-authored and pool-sourced wrong options — verified to have
**zero duplicate options within a single question** across the whole
dataset.

## Scoring methodology

- **`correctness` (0/1):** computed automatically by the shared runner via
  exact-match between `extracted_final_answer` and `gold_answer` (the
  correct option letter), after `normalize_fa_text` normalization. No
  judge model or runner changes were needed.
- **`persian_stability`** (`rubrics/persian_stability.json`) is applied to
  the raw output.
- **`verifiable_reasoning`** is intentionally **not** applied, since a hard
  gold answer already exists.

## Running this task

```bash
# 1. Convert the raw hand-authored data into task-ready multiple-choice format:
python scripts/prepare_proverbs_task.py \
  --input data/raw_sources/proverbs_fa_raw.json \
  --output data/tasks/proverbs_fa.jsonl \
  --num-options 4 \
  --seed 42

# 2. Run a candidate model against the task:
python -m src.pipeline.runner \
  --task-file data/tasks/proverbs_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/proverbs_fa__gpt-oss-20b-fa-cot-v1.jsonl

# 3. Aggregate results across all tasks/models:
python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

## Output fields relevant to this task

In `data/scored_results/*.jsonl`, each row (`ScoredResult`) includes:

- `correctness`: `0` or `1`, exact-match of the chosen option letter
  against `gold_answer`.
- `persian_stability_llm_total` (0–8), `code_switch_rate`,
  `persian_script_ratio`.
- `verifiable_reasoning_total`: always `null` for this task (not applied).
- `raw_output`: the full model response.

## Known limitations

- 4-option MCQ with a 25% random-guessing floor, same as Aroozi/MMLU-lite.
- Exact-match on the option letter is format-sensitive, the same
  known tradeoff documented for other MCQ tasks in this framework.
- 49 samples is a small slice of the Persian proverb corpus; coverage is
  not exhaustive and skews toward commonly cited proverbs.
- Some `gold_meaning` phrasings are fairly long/explanatory (to fully
  capture nuance), which can make distinguishing the correct option from
  a well-written distractor nontrivial even for a careful human reader —
  this is by design (the task is meant to be genuinely hard), but worth
  keeping in mind when interpreting low scores.
