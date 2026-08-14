# Phase 1 — Jalali Calendar Task

## Overview

This task evaluates a model's ability to perform date-related reasoning
grounded in the **Jalali (Persian solar hijri) calendar**, in Persian:
converting between Jalali and Gregorian dates, judging leap years, adding
or subtracting days/months from a Jalali date, and computing the day-count
distance between two Jalali dates. It is a deterministic, exact-match task,
in the same spirit as BBH-lite's `arithmetic` and `navigate` categories,
but focused specifically on calendar arithmetic that Persian-language
models are expected to handle correctly given how central the Jalali
calendar is to everyday Persian usage.

## Data

- **Source:** entirely **synthetically generated**, no manual authoring or
  translation. Every question and its `gold_answer` are produced
  deterministically using the `jdatetime` library
  (`scripts/generate_jalali_calendar_data.py`, seeded with `random.seed(42)`
  for reproducibility), so correctness of the gold answers is guaranteed by
  construction rather than by manual review.
- **Sample size:** 150 samples across 4 categories:

| Category | Count | Description |
|---|---|---|
| `date_conversion` | 38 | Convert a given date from Jalali to Gregorian or from Gregorian to Jalali. |
| `leap_year` | 38 | Determine whether a given Jalali year is a leap year. |
| `date_arithmetic` | 37 | Add or subtract a number of days or months to/from a given Jalali date. |
| `date_distance` | 37 | Compute the number of days between two given Jalali dates. |

### Answer format conventions

- Jalali dates are written as `"<day in Persian digits> <Persian month name> <year in Persian digits>"`,
  e.g. `"۳۰ فروردین ۱۴۰۲"`.
- Gregorian dates are written as `"<day> <English month name> <year>"`,
  e.g. `"5 June 2021"`.
- Leap-year answers are `"بله"` / `"خیر"`.
- Day-distance answers are a plain number in Persian digits, e.g. `"۹۰۰"`.

These conventions are fixed by the generation script so that every sample
in a given category has a consistently formatted gold answer, which is
important for reliable exact-match scoring.

### Raw format (`data/raw_sources/jalali_calendar_fa_raw.json`)

```json
{
  "category": "date_conversion",
  "question": "تاریخ ۱۳ شهریور ۱۳۷۷ (تقویم جلالی) در تقویم میلادی چه روزی است؟",
  "answer": "4 September 1998"
}
```

### Task-ready format (`data/tasks/jalali_calendar_fa.jsonl`)

Produced by `scripts/prepare_jalali_task.py`, which validates each sample's
`category` against the fixed set of 4 known categories and wraps it into
the framework's shared `TaskSample` schema:

```json
{
  "sample_id": "jalali_date_conversion_0001",
  "task_type": "jalali_calendar",
  "problem_fa": "تاریخ ۱۳ شهریور ۱۳۷۷ (تقویم جلالی) در تقویم میلادی چه روزی است؟",
  "gold_answer": "4 September 1998",
  "requires_cot_judging": false,
  "apply_verifiable_reasoning_rubric": false,
  "apply_persian_stability_rubric": true,
  "system_prompt_fa": "تو یک دستیار دقیق فارسی‌زبان هستی که در محاسبات تقویم جلالی و میلادی مهارت داری. ...",
  "answer_extraction_regex": "پاسخ نهایی:\\s*(.+)",
  "extra": {
    "category": "date_conversion"
  }
}
```

`sample_id` encodes the category so per-category accuracy can be sliced
out at aggregation time, the same convention used by the BBH-lite task.

## Why no manual data collection?

Calendar arithmetic is fully deterministic given a correct conversion
algorithm — there is no ambiguity or judgment call in "what date is 73
days before 8 Farvardin 1387," so hand-authoring or translating examples
would add human-error risk (transcription mistakes, off-by-one errors)
without adding any value. Generating with `jdatetime` guarantees every
gold answer is correct by construction, and `random.seed(42)` makes the
dataset fully reproducible if it needs to be regenerated or extended.

## Scoring methodology

- **`correctness` (0/1):** computed automatically by the shared runner
  (`src/pipeline/runner.py`) via exact-match between
  `extracted_final_answer` and `gold_answer`, after `normalize_fa_text`
  normalization (whitespace, `ي`/`ك` unification, trailing punctuation).
  No judge model is involved in this comparison.
- **`persian_stability`** (`rubrics/persian_stability.json`) is applied to
  the raw output, the same as in the BBH-lite task, to track whether the
  model's short Persian reasoning stays linguistically stable even on
  fully deterministic calculations.
- **`verifiable_reasoning`** is intentionally **not** applied, for the same
  reason as in BBH-lite: a hard gold answer already exists, so judging the
  reasoning chain's internal validity adds cost without much additional
  signal.

## Running this task

```bash
# 1. (Already done) Generate the raw synthetic data:
python scripts/generate_jalali_calendar_data.py

# 2. (Already done) Convert the raw data into task-ready format:
python scripts/prepare_jalali_task.py \
  --input data/raw_sources/jalali_calendar_fa_raw.json \
  --output data/tasks/jalali_calendar_fa.jsonl

# 3. Run a candidate model against the task:
python -m src.pipeline.runner \
  --task-file data/tasks/jalali_calendar_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/jalali_calendar_fa__gpt-oss-20b-fa-cot-v1.jsonl

# 4. Aggregate results across all tasks/models:
python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

## Output fields relevant to this task

In `data/scored_results/*.jsonl`, each row (`ScoredResult`) includes:

- `correctness`: `0` or `1`, exact-match against `gold_answer`.
- `persian_stability_llm_total` (0–8), `code_switch_rate`, `persian_script_ratio`.
- `verifiable_reasoning_total`: always `null` for this task (not applied).
- `raw_output`: the full model response, kept for manual spot-checking.

## Known limitations

- Exact-match is sensitive to formatting: a model that answers
  `"1402/01/30"` or `"30 فروردین سال 1402"` instead of `"۳۰ فروردین ۱۴۰۲"`
  will fail exact-match even though the date itself is correct.
  `normalize_fa_text` does not currently normalize date-format variance or
  digit systems (Persian vs. Latin digits), so this is a known source of
  false negatives worth spot-checking before trusting `date_conversion` and
  `date_arithmetic` accuracy at face value.
- `date_arithmetic` month-addition/subtraction samples clamp the resulting
  day to the target month's max day (e.g. adding a month to the 31st of a
  31-day month when the target month only has 30 days) rather than
  overflowing into the next month — this mirrors common calendar-library
  convention but is a design choice models are not told about explicitly,
  and could be considered a slightly ambiguous edge case.
- The dataset only covers Jalali years roughly in the 1370–1450 range
  (~1990s–2070s Gregorian); it does not test extreme dates far outside
  this window.
