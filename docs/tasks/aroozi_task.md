# Phase 1 — Aroozi (Persian Classical Meter) Task

## Overview

This task evaluates a model's ability to identify the classical عروضی
(prosodic) meter of a Persian verse (`بیت`) drawn from canonical poets
(حافظ، سعدی، مولانا و...). Rather than asking the model to produce the
full metrical pattern as free text — which is extremely format-sensitive
and hard to score reliably with exact-match — each sample is turned into a
**multiple-choice question**: the model is shown the verse plus a small
set of candidate meters (one correct, the rest distractors sampled from
the other meters actually present in the dataset) and must pick the
correct option letter. This keeps the task deterministic and cheaply
scorable via exact-match, in the same spirit as BBH-lite and Jalali
calendar, while still requiring genuine metrical scansion (تقطیع) to
answer correctly rather than superficial pattern matching.

## Data

- **Source:** hand-curated real verses with human-verified gold meters
  (`gold_meter_pattern` — the عروضی foot pattern — and `gold_meter_name` —
  the named بحر — for each verse), attributed to their poet and source
  (e.g. غزل شماره). Not synthetically generated, since determining a
  verse's meter is a non-trivial task in itself.
- **Sample size:** 100 samples, drawn from 10 distinct meters, 10 samples
  per meter.
- **Subtype:** all current samples are `meter_identification`; the schema
  leaves room for other `aroozi` subtypes later (e.g. scansion/تقطیع
  production) without breaking this format.

### Raw format (`data/raw_sources/aroozi_meter_fa_raw.json`)

```json
{
  "id": 84,
  "task_type": "aroozi",
  "subtype": "meter_identification",
  "verse": "منظور خردمند من آن ماه که او را / با حسن ادب شیوهٔ صاحب‌نظری بود",
  "gold_meter_pattern": "مفعول مفاعیل مفاعیل فعولن",
  "gold_meter_name": "بحر هزج مثمن اخرب مکفوف محذوف",
  "poet": "حافظ",
  "source": "غزل ۲۱۶"
}
```

> **Note:** the full 100-sample dataset (10 meters × 10 verses) ships in
> this repo at `data/raw_sources/aroozi_meter_fa_raw.json`, and the
> corresponding task-ready file at `data/tasks/aroozi_meter_fa.jsonl` was
> generated from it with the default `--num-options 4 --seed 42`. If the
> raw file is ever extended or edited, re-run
> `scripts/prepare_aroozi_task.py` to regenerate the task-ready file.

### Task-ready format (`data/tasks/aroozi_meter_fa.jsonl`)

Produced by `scripts/prepare_aroozi_task.py`, which:

1. Builds a pool of every unique `(gold_meter_name, gold_meter_pattern)`
   pair seen across the whole raw dataset (10 pairs, for the full
   dataset).
2. For each verse, samples `--num-options - 1` **distractor** meters from
   that pool (excluding the verse's own correct meter), so wrong options
   are always real meters that appear elsewhere in the dataset — not
   arbitrary made-up patterns.
3. Shuffles the correct option in among the distractors and assigns
   option letters (`الف`, `ب`, `ج`, `د`, ...).
4. Wraps the verse + rendered options into `problem_fa`, and sets
   `gold_answer` to the correct option's letter.

Option generation is **deterministic**: each sample uses its own
`random.Random(f"{seed}-{raw_id}")`, seeded from `--seed` (default `42`)
and the sample's own `id`, so re-running the script produces byte-identical
output — important for reproducibility and for keeping option sets stable
across repeated runs/models.

```json
{
  "sample_id": "aroozi_meter_0084",
  "task_type": "aroozi",
  "problem_fa": "بیت زیر را از نظر وزن عروضی بررسی کن:\n\"منظور خردمند من آن ماه که او را / با حسن ادب شیوهٔ صاحب‌نظری بود\"\n\nوزن این بیت کدام است؟\nالف) بحر خفیف مسدس مخبون (فعلاتن مفاعلن فعلن)\nب) بحر هزج مثمن اخرب مکفوف محذوف (مفعول مفاعیل مفاعیل فعولن)\n...",
  "gold_answer": "ب",
  "requires_cot_judging": false,
  "apply_verifiable_reasoning_rubric": false,
  "apply_persian_stability_rubric": true,
  "system_prompt_fa": "تو یک دستیار متخصص در عروض کلاسیک فارسی هستی. ...",
  "answer_extraction_regex": "پاسخ نهایی:\\s*(.+)",
  "extra": {
    "subtype": "meter_identification",
    "verse": "منظور خردمند من آن ماه که او را / با حسن ادب شیوهٔ صاحب‌نظری بود",
    "poet": "حافظ",
    "source": "غزل ۲۱۶",
    "gold_meter_name": "بحر هزج مثمن اخرب مکفوف محذوف",
    "gold_meter_pattern": "مفعول مفاعیل مفاعیل فعولن",
    "options": {
      "الف": {"meter_name": "بحر خفیف مسدس مخبون", "meter_pattern": "فعلاتن مفاعلن فعلن"},
      "ب": {"meter_name": "بحر هزج مثمن اخرب مکفوف محذوف", "meter_pattern": "مفعول مفاعیل مفاعیل فعولن"}
    },
    "correct_option": "ب",
    "raw_id": 84
  }
}
```

`extra.options` and `extra.correct_option` are kept alongside the plain
`gold_answer` letter so that scored results can be joined back to the
full option set (and to `poet`/`source`) at analysis time without
re-parsing `problem_fa`.

## Why multiple-choice instead of free-text meter production?

Asking a model to freely produce `مفعول مفاعیل مفاعیل فعولن` as text is
brittle to score: spacing, alternate foot transliterations, and whether
the model names the بحر, the foot pattern, or both, all vary in ways
`normalize_fa_text`'s light normalization (whitespace, `ي`/`ك`,
punctuation) doesn't resolve. Turning it into a closed-set choice over
**real meters drawn from the dataset itself** keeps the interesting part
of the task (the model actually has to scan the verse to tell metrically
similar meters apart) while making scoring a simple, reliable exact-match
on a single option letter — the same tradeoff BBH-lite and Jalali
calendar make for their own gold answers.

## Scoring methodology

- **`correctness` (0/1):** computed automatically by the shared runner
  (`src/pipeline/runner.py`) via exact-match between
  `extracted_final_answer` and `gold_answer` (the correct option letter),
  after `normalize_fa_text` normalization. No judge model or runner
  changes were needed — this task reuses the same generic `gold_answer`
  path as BBH-lite and Jalali calendar.
- **`persian_stability`** (`rubrics/persian_stability.json`) is applied to
  the raw output, the same as BBH-lite and Jalali calendar.
- **`verifiable_reasoning`** is intentionally **not** applied, for the same
  reason as the other exact-match tasks: a hard gold answer already
  exists.

## Running this task

```bash
# 1. Convert the raw hand-curated data into task-ready multiple-choice format:
python scripts/prepare_aroozi_task.py \
  --input data/raw_sources/aroozi_meter_fa_raw.json \
  --output data/tasks/aroozi_meter_fa.jsonl \
  --num-options 4 \
  --seed 42

# 2. Run a candidate model against the task:
python -m src.pipeline.runner \
  --task-file data/tasks/aroozi_meter_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/aroozi_meter_fa__gpt-oss-20b-fa-cot-v1.jsonl

# 3. Aggregate results across all tasks/models:
python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

`--num-options` defaults to 4 and must be between 2 and 6 (letters
`الف`–`و`), and can't exceed the number of unique meters present in the
raw file (10, for the full dataset) — the script raises a clear error
instead of silently producing malformed questions if it does.

## Output fields relevant to this task

In `data/scored_results/*.jsonl`, each row (`ScoredResult`) includes:

- `correctness`: `0` or `1`, exact-match of the chosen option letter
  against `gold_answer`.
- `persian_stability_llm_total` (0–8), `code_switch_rate`,
  `persian_script_ratio`.
- `verifiable_reasoning_total`: always `null` for this task (not applied).
- `raw_output`: the full model response, kept for manual spot-checking
  (e.g. to see whether the model's stated تقطیع reasoning was actually
  sound even when it picked the wrong letter, or vice versa).

## Known limitations

- Exact-match on the option letter is format-sensitive: the system prompt
  explicitly asks for *only* the bare letter (e.g. `الف`) after
  `پاسخ نهایی:`, but a model that instead writes `الف)` or repeats the
  full option text (`الف) بحر خفیف مسدس مخبون`) will fail exact-match
  even though it clearly picked the right option. This mirrors the same
  known tradeoff documented for BBH-lite and Jalali calendar; worth a
  spot-check pass on `raw_output` before trusting aggregate accuracy at
  face value, or extending `normalize_fa_text`/the extraction regex if
  this turns out to be common in practice.
- With only 10 distinct meters in the pool, at `--num-options 4` a
  purely random guesser scores ~25% — keep this baseline in mind when
  interpreting model accuracy, and consider a larger `--num-options` if a
  higher/lower guessing floor is wanted.
- Distractors are sampled only from meters that already exist elsewhere
  in the dataset, not from the full universe of classical Persian meters
  — so a model could in principle do reasonably well by memorizing "which
  10 meters this benchmark uses" rather than truly scanning each verse.
  This is a deliberate simplicity/scorability tradeoff, not an attempt to
  cover the full space of عروضی meters.
- The dataset only covers 10 distinct meters (each with 10 verses), not
  the full space of classical Persian عروضی meters — this is a curated,
  fixed benchmark set, not a comprehensive coverage of Persian prosody.
