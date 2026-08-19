# Phase 2 — Script Disambiguation (ابهام‌زدایی رسم‌الخط) Task

## Overview

This task evaluates a model's ability to disambiguate Persian **homographs**
— words that are written identically without diacritics (اعراب) but have
more than one valid reading/meaning depending on context — e.g. «کرم» can
be read کِرم (worm) or کَرَم (generosity). Each sample gives a sentence
containing the ambiguous word (usually used twice, once in each sense, to
make the intended sense in the tested half unambiguous from context) and
asks the model to pick the correct (reading, meaning) pair for that
specific occurrence out of a small set of multiple-choice options — the
same exact-match-friendly design as Aroozi and MMLU-lite.

## Data

- **Source:** hand-authored, 50 samples.
- **Sample size:** 50 samples covering 25 distinct homographs (2 samples
  per word: each sample tests the *other* sense of the same word in a
  different sentence — except «ملک» which has 3 samples covering its 3
  senses: مَلِک/پادشاه, مَلَک/فرشته, مُلک/سرزمین).
- **Subtype:** all current samples are `homograph_disambiguation`.

### Raw format (`data/raw_sources/script_disambiguation_fa_raw.json`)

```json
{
  "id": 1,
  "task_type": "script_disambiguation",
  "subtype": "homograph_disambiguation",
  "sentence": "سحر از خانه بیرون آمد و سحر هنوز در آسمان باقی بود.",
  "ambiguous_word": "سحر",
  "context_hint": "در نیمه دوم جمله، منظور زمان پیش از طلوع خورشید است.",
  "gold_reading": "سَحَر",
  "gold_meaning": "هنگام پیش از طلوع خورشید",
  "manual_distractors": [
    {"reading": "سِحر", "meaning": "جادو و افسون"}
  ],
  "poet_or_source": null
}
```

`context_hint` disambiguates *which* occurrence of the word in the
sentence the question is about, since the sentence deliberately contains
both senses.

### Task-ready format (`data/tasks/script_disambiguation_fa.jsonl`)

Produced by `scripts/prepare_script_disambiguation_task.py`.

## Why distractors are same-word-only (not a dataset-wide pool)

Aroozi and Proverbs fall back to a **dataset-wide pool** when a sample
doesn't have enough hand-authored distractors — any other meter, or any
other proverb's meaning, is a plausible-sounding wrong answer regardless
of which verse/proverb the question is about.

That assumption **does not hold** for homographs: a wrong reading is only
meaningful if it's an alternate reading of the *same written word*.
Offering "دَوْر — نوبت، مرحله یا چرخه" as a wrong option for the word
"سحر" is nonsensical — a model doesn't need to understand the ambiguity of
"سحر" at all to reject it, since it's visibly a different word. Mixing in
cross-word options would make the questions trivially easy and stop
testing disambiguation.

So `prepare_script_disambiguation_task.py` builds a **per-word pool**
(`build_word_pool`): for each `ambiguous_word`, it collects every known
`(reading → meaning)` mapping from across the whole raw file — both each
sample's own `gold_reading`/`gold_meaning` and every `manual_distractors`
entry for that word — and only ever draws distractors for a sample from
that word's own pool.

### Deduplication is by reading, not by (reading, meaning)

Because the same reading's meaning is sometimes phrased slightly
differently across different raw samples (e.g. "داور یا فردی که درباره
اختلاف یا مسابقه داوری می‌کند" in one sample vs. just "داور" in another,
both for the reading حَکَم), the pool is keyed on **reading alone**, with
one canonical meaning chosen per reading (preferring the phrasing used
wherever that reading appears as a sample's own `gold_reading`, since
that's usually the more complete/deliberate phrasing). Without this, two
reworded copies of the *same* correct reading could appear as two
different options in one question — one correct, one an accidental
"distractor" that a careful model would have to guess was actually wrong
despite being semantically right.

## Variable option count instead of a fixed `--num-options`

Because most homographs in this dataset have exactly 2 known senses, most
generated questions are naturally **2-option** MCQs; the 3 samples for
«ملک» (which has 3 senses) become **3-option** MCQs. `prepare_*.py` takes
`--max-options` (default 4, an upper cap in case a future word has more
senses) and `--min-options` (default 2 — a sample is skipped if its word
has no known alternate reading at all). With the current 50-sample file,
this produces **47 two-option and 3 three-option** questions, with 0
skipped.

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
python scripts/prepare_script_disambiguation_task.py \
  --input data/raw_sources/script_disambiguation_fa_raw.json \
  --output data/tasks/script_disambiguation_fa.jsonl \
  --max-options 4 \
  --seed 42

# 2. Run a candidate model against the task:
python -m src.pipeline.runner \
  --task-file data/tasks/script_disambiguation_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/script_disambiguation_fa__gpt-oss-20b-fa-cot-v1.jsonl

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

`extra.num_options` in the task-ready jsonl records how many options that
particular sample had (2 or 3 in the current dataset), useful for
adjusting the guessing-floor baseline per sample at analysis time.

## Known limitations

- With only 2–3 options per question (vs. 4 for Aroozi/MMLU-lite), the
  random-guessing floor is higher (50%–33%) — keep this in mind when
  interpreting accuracy, especially for the 2-option majority of samples.
- Exact-match on the option letter is format-sensitive, the same
  known tradeoff documented for Aroozi/BBH-lite/Jalali calendar.
- The dataset covers 25 distinct homographs, each with only their 2 (or
  for «ملک», 3) most common senses — not a comprehensive inventory of
  Persian homographs or of every sense a given homograph can carry.
