# Phase 2 — Ieham / Wordplay (ایهام) Task

## Overview

This is the third and final Phase 2 task (`script_disambiguation` ✅,
`proverbs` ✅ previously completed). It evaluates a model's ability to
recognize **ایهام** — the classical Persian rhetorical device where a word
carries two simultaneously valid meanings within a verse (بیت): a
"near"/obvious sense and a "far"/subtle sense that carries the poet's
actual point. Each sample gives a verse and asks the model to pick the
correct (near, far) meaning **pair** for a specified ambiguous word out of
a small set of multiple-choice options — the same exact-match-friendly
design as Script Disambiguation and Proverbs.

## Data

- **Source:** hand-authored, 34 unique samples (an accidental near-duplicate
  — the exact same verse, word, and far-meaning, with only the near-meaning
  phrasing slightly different, id 3 vs. id 24 of the original 35-sample
  submission — was identified and removed).
- **Subtype:** all samples are `double_meaning_identification`.
- Several verses contribute more than one sample, each testing a
  *different* ambiguous word within the same بیت (e.g. the Hafez couplet
  "ز گریه مردم چشمم..." yields three separate samples for «مردمان»,
  «مردم», and «چشم»). This is intentional, not a duplicate.

### Raw format (`data/raw_sources/ieham_fa_raw.json`)

```json
{
  "id": 1,
  "task_type": "ieham",
  "subtype": "double_meaning_identification",
  "verse": "ز گریه مردم چشمم نشسته در خون است / ببین که در طلبت حال مردمان چون است",
  "ambiguous_word": "مردمان",
  "gold_meaning_near": "مردم و انسان‌ها",
  "gold_meaning_far": "مردمک‌های چشم",
  "manual_distractors": [
    {"near": "یاران و دوستان", "far": "اشک‌های چشم"},
    {"near": "اهالی شهر", "far": "سیاهی چشم"}
  ],
  "poet": "حافظ",
  "source": "دیوان حافظ، غزل ۵۴"
}
```

### Task-ready format (`data/tasks/ieham_fa.jsonl`)

Produced by `scripts/prepare_ieham_task.py`. Each option renders as
"نزدیک: ... / دور: ...".

## Why distractors are same-word-only (same fix as Script Disambiguation)

Like homographs, a wrong (near, far) pair is only meaningful if it's an
alternate meaning-pair of the *same* ambiguous word — a dataset-wide pool
(safe for Proverbs, where any other proverb's meaning is plausible) would
be unsafe here: offering "نزدیک: گورخر / دور: قبر" (the pair for «گور»)
as a wrong option for the word «مشتری» is trivially rejectable without
any real understanding of *this* verse's wordplay. So
`prepare_ieham_task.py` builds a **per-word pool** (`build_word_pool`),
mirroring Script Disambiguation exactly: distractors for a sample are
drawn only from other (near, far) pairs recorded for that sample's own
`ambiguous_word` across the whole raw file.

Most words in this dataset (28 of 31) appear in exactly one sample, so
their 2 `manual_distractors` alone yield a **3-option MCQ**. Three words
repeat across multiple samples («شیرین» ×2, «مشتری» ×2, «مهر» ×2), pooling
their distractors into up to **4-option MCQs**. `--max-options`/
`--min-options` (default 4/2) replace a fixed `--num-options`, same as
Script Disambiguation.

### Deduplication is keyed on the far-meaning, not the (near, far) pair

The far-meaning is the specific, named referent of the wordplay (e.g.
"شیرین، معشوقهٔ فرهاد", "خورشید", "سیارهٔ مشتری") and stays consistent
across repeated samples for a word, while the near-meaning is a
descriptive gloss that gets paraphrased slightly differently sample to
sample (e.g. "خوشایند و دلپذیر" vs. "دلپذیر و خوشایند" for شیرین). Pooling
naively on the full pair produced exactly the bug seen in Script
Disambiguation: two reworded copies of the same referent appearing as two
different options in one question (caught during QA — see below). The
pool is keyed on `far` alone, with one canonical `near` phrasing per `far`
(preferring wherever that far is a sample's own `gold_meaning_far`).

## QA finding fixed before finalizing

During generation, the same paraphrase-variance issue surfaced on a
**distractor**, not just a gold pair: id 3 and id 23 (both testing «شیرین»
→ «شیرین، معشوقهٔ فرهاد») each had their own distractor referring to
Khosrow, phrased as "معشوقهٔ خسرو" in one and bare "خسرو" in the other.
Naive far-keying treated these as two distinct entries, so both ended up
as separate options in the same 4-option question — two visually
different but semantically identical wrong answers. Fixed by normalizing
the phrasing to "معشوقهٔ خسرو" in both raw entries, collapsing them to one
pool entry; both «شیرین» questions are now clean 3-option MCQs.

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
python scripts/prepare_ieham_task.py \
  --input data/raw_sources/ieham_fa_raw.json \
  --output data/tasks/ieham_fa.jsonl \
  --max-options 4 \
  --seed 42

# 2. Run a candidate model against the task:
python -m src.pipeline.runner \
  --task-file data/tasks/ieham_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/ieham_fa__gpt-oss-20b-fa-cot-v1.jsonl

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
particular sample had (2, 3, or 4 in the current dataset), useful for
adjusting the guessing-floor baseline per sample at analysis time.
`extra.poet`/`extra.source` preserve attribution for reference.

## Known limitations

- With 2–4 options per question (vs. a fixed 4 for Aroozi/MMLU-lite/
  Proverbs), the random-guessing floor varies by sample (50%–25%) — keep
  this in mind when interpreting accuracy, especially for the many
  3-option (33% floor) samples.
- Exact-match on the option letter is format-sensitive, the same known
  tradeoff documented for other MCQ tasks in this framework.
- Same-word-only pooling is a strict quality guarantee, but it also means
  most questions only have 3 options rather than 4 — adding more samples
  per repeated word (or more `manual_distractors` per sample) would let
  more questions reach 4 options.
- The dataset covers 31 distinct ambiguous words across classical poets
  (mostly Hafez and Saadi) — not a comprehensive inventory of Persian
  ایهام instances or poets.
