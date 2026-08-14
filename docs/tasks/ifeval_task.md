# Phase 1 — IFEval-lite Task

## Overview

This task evaluates a model's ability to follow **explicit, machine-checkable
instructions** in Persian: producing output with a specific sentence,
word, or paragraph count; including or excluding a specific word;
repeating a word an exact number of times; or starting/ending the response
with a specific word. It is a lightweight, Persian-native adaptation of
Google's IFEval benchmark — hence "IFEval-lite" — restricted to constraint
types that can be verified with a plain rule-based checker rather than an
LLM judge.

Unlike every other task so far in FARMA, IFEval-lite samples have **no
`gold_answer` text to match**. Instead, each sample carries a structured,
machine-checkable **constraint spec**, and correctness is the boolean
outcome of running that constraint's checker against the model's raw
output.

## Data

- **Source:** entirely **synthetically generated**, combining 40
  hand-picked Persian topics (provided by the project owner) with
  rule-verifiable constraints generated deterministically
  (`scripts/generate_ifeval_lite_data.py`, seeded with `random.seed(7)`).
- **Sample size:** 120 samples — 3 per topic, one from each constraint
  category:

| Category | Count | Constraint types |
|---|---|---|
| `length_constraint` | 40 | `sentence_count` (exact), `word_count` (at least), `paragraph_count` (exact, blank-line separated) |
| `keyword_constraint` | 40 | `must_include`, `must_include_exact_count`, `must_exclude` |
| `position_constraint` | 40 | `must_start_with`, `must_end_with` |

### Raw format (`data/raw_sources/ifeval_lite_fa_raw.json`)

```json
{
  "category": "length_constraint",
  "topic": "تغییرات آب‌وهوا",
  "question": "درباره‌ی «تغییرات آب‌وهوا» بنویس. پاسخ باید دقیقاً ۵ جمله داشته باشد؛ نه بیشتر و نه کمتر.",
  "constraint": {"type": "sentence_count", "mode": "exact", "value": 5}
}
```

### Task-ready format (`data/tasks/ifeval_lite_fa.jsonl`)

Produced by `scripts/prepare_ifeval_task.py`, which validates each sample's
`category`, keeps `gold_answer` as `null`, and stores the constraint spec
under `extra.constraint`:

```json
{
  "sample_id": "ifeval_length_constraint_0000",
  "task_type": "ifeval_lite",
  "problem_fa": "درباره‌ی «تغییرات آب‌وهوا» بنویس. پاسخ باید دقیقاً ۵ جمله داشته باشد؛ نه بیشتر و نه کمتر.",
  "gold_answer": null,
  "requires_cot_judging": false,
  "apply_verifiable_reasoning_rubric": false,
  "apply_persian_stability_rubric": true,
  "system_prompt_fa": "تو یک دستیار فارسی‌زبان هستی که دقیقاً طبق دستورالعمل‌های داده‌شده عمل می‌کنی. ...",
  "answer_extraction_regex": "(?s)(.*)",
  "extra": {
    "category": "length_constraint",
    "topic": "تغییرات آب‌وهوا",
    "constraint": {"type": "sentence_count", "mode": "exact", "value": 5}
  }
}
```

Note `answer_extraction_regex` is `"(?s)(.*)"` for this task — the whole
raw response is the object being checked (its sentence count, its first
word, whether a word appears in it, etc.), not a `"پاسخ نهایی:"` line, so
there is nothing to strip out during extraction.

## Why no gold_answer, and why not a judge?

There is no single correct string for "write about climate change in
exactly 5 sentences" — many different 5-sentence responses are equally
valid. What's checkable is only the *constraint*, not the content. An LLM
judge could check this too, but it would be slower, more expensive, and
strictly less reliable than a plain rule-based counter for something as
mechanical as counting sentences or checking a word's presence — so this
task is deliberately scored without any judge model involvement.

## A small addition to the core runner

Every prior task fit one of two existing scoring paths in
`src/pipeline/runner.py`: exact-match against `gold_answer`, or an LLM
judge rubric. IFEval-lite's rule-based constraint checking is a third,
genuinely different kind of check, so `score_sample()` in
`src/pipeline/runner.py` gained one small `elif` branch:

```python
elif sample.extra.get("constraint") is not None:
    passed, reason_fa = check_constraint(generation.raw_output, sample.extra["constraint"])
    result.correctness = int(passed)
    result.notes = reason_fa
```

This reuses the existing `ScoredResult.correctness` field (already
documented as a generic 0/1 field, previously only populated from
`gold_answer` matches) and the existing `notes` field, so no schema change
was needed — only this one additional branch, gated behind
`extra.constraint`, which no other task's samples set. Tasks that don't
use this field are completely unaffected.

The actual per-constraint-type logic lives in
`src/utils/instruction_utils.py` (`check_constraint` and its 8 individual
checkers: `check_sentence_count`, `check_word_count`,
`check_paragraph_count`, `check_must_include`,
`check_must_include_exact_count`, `check_must_exclude`,
`check_must_start_with`, `check_must_end_with`), covered by unit tests in
`tests/test_instruction_utils.py`.

## Scoring methodology

- **`correctness` (0/1):** the boolean result of
  `check_constraint(raw_output, constraint)` for the sample's constraint
  spec. `notes` holds a short Persian explanation of the measured value
  vs. the target (e.g. `"تعداد جمله‌ها: 4 (هدف: exact 5)"`), useful for
  spot-checking failures.
- **`persian_stability`** (`rubrics/persian_stability.json`) is applied to
  the raw output, the same as in BBH-lite and the Jalali calendar task, to
  track whether the model's Persian stays fluent and stable while also
  satisfying a structural constraint.
- **`verifiable_reasoning`** is not applicable here — these are not
  reasoning chains to begin with, just constrained short-form writing.

## Running this task

```bash
# 1. (Already done) Generate the raw synthetic data:
python scripts/generate_ifeval_lite_data.py

# 2. (Already done) Convert the raw data into task-ready format:
python scripts/prepare_ifeval_task.py \
  --input data/raw_sources/ifeval_lite_fa_raw.json \
  --output data/tasks/ifeval_lite_fa.jsonl

# 3. Run a candidate model against the task:
python -m src.pipeline.runner \
  --task-file data/tasks/ifeval_lite_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/ifeval_lite_fa__gpt-oss-20b-fa-cot-v1.jsonl

# 4. Aggregate results across all tasks/models:
python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

## Output fields relevant to this task

In `data/scored_results/*.jsonl`, each row (`ScoredResult`) includes:

- `correctness`: `0` or `1`, result of the rule-based constraint checker.
- `notes`: short Persian explanation of the measured value vs. the target.
- `persian_stability_llm_total` (0–8), `code_switch_rate`, `persian_script_ratio`.
- `verifiable_reasoning_total`: always `null` for this task (not applied).
- `raw_output`: the full model response, kept for manual spot-checking.

## Known limitations

- `sentence_count` relies on `split_sentences` (shared with
  `persian_stability`), which splits on `.`, `!`, `؟`, `?`. Abbreviations,
  decimal numbers, or ellipses inside a sentence can cause miscounts.
- `must_start_with` / `must_end_with` compare the first/last whitespace
  token after stripping common edge punctuation and ZWNJ; a model that
  wraps the target word in quotes, bold markdown (`**کلمه**`), or attaches
  unexpected punctuation may fail the check even though a human reader
  would consider the constraint satisfied.
- `word_count` and `paragraph_count` use simple whitespace/blank-line
  splitting, which does not account for models that use markdown bullet
  lists or numbered lists as "paragraphs."
- The 3 constraint categories included here (length, keyword, position)
  are a deliberate subset of what a full IFEval-style benchmark covers
  (e.g. no formatting constraints like JSON-only output, no combined
  multi-constraint samples). Broader coverage can be added later by
  extending `_CHECKERS` in `src/utils/instruction_utils.py` and the
  generation script.
