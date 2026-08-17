# Phase 3 — Minimal Pairs (Grammaticality Judgment) Task

## Overview

Implements the **minimal pairs** sub-task of Phase 3: a benchmark that
evaluates whether a model has genuine, generalizable knowledge of Persian
grammar rather than pattern-matching on surface fluency. Each item presents
two sentences that differ by a single, targeted grammatical edit — one
sentence is grammatical, the other is a minimally-changed ungrammatical
variant — and the model must pick which one is correct.

This is the classic "minimal pairs" methodology used by linguistic
acceptability benchmarks (e.g. BLiMP for English): by holding everything
constant except the one grammatical feature under test, a wrong answer is
much harder to attribute to unrelated factors (vocabulary, topic, sentence
length) than a free-form grammar question would allow.

Like Aroozi and script-disambiguation, this is a closed-set task built
entirely on the Phase 0 infrastructure — no changes to `runner.py` were
needed.

## Data

- `data/raw_sources/minimal_pairs_fa_raw.json` — 64 hand-authored pairs
  (all original content, not sourced from any external corpus) across 8
  grammatical phenomena, 8 pairs each:

| Category | Rule tested |
|---|---|
| `subject_verb_agreement` | Human plural subject requires a plural verb |
| `verb_final_order` | Standard-register Persian is verb-final; fronting the verb before the object is ungrammatical |
| `ra_marker_definite_object` | A direct object made definite by a demonstrative (این/آن) requires the «را» marker |
| `negation_prefix_position` | The negation prefix «نـ» must attach directly before «می‌», not be reordered |
| `tense_adverb_agreement` | A past-time adverb (دیروز, سال گذشته, ...) requires a past-tense verb |
| `no_double_plural` | The plural marker (ها/ان) attaches once; doubling it is ungrammatical |
| `comparative_suffix_attachment` | The comparative suffix «تر» attaches directly to the adjective, not detached |
| `locative_preposition_compatibility` | Static location with «بودن» takes «در», not «به» |

Each raw record: `id`, `category`, `category_label`, `rule` (human-readable
explanation, not shown to the model), `correct_sentence`,
`incorrect_sentence`.

### Task Preparation Script

`scripts/prepare_minimal_pairs_task.py`:
- For each pair, deterministically shuffles which sentence is labeled الف
  vs. ب (`random.Random(f"{seed}-{raw_id}")`), so the model can't game the
  task by always picking one letter.
- Validates both sentences are present and non-identical; skips/logs
  anything malformed.
- Produces `data/tasks/minimal_pairs_fa.jsonl` (64/64 converted, 0 skipped).

Unlike Aroozi and script-disambiguation, there is no distractor pool to
build — a minimal pair is a fixed 2-option (الف/ب) choice by construction,
since the "distractor" is always the paired ungrammatical sentence.

## Pipeline Integration

Reuses the framework's generic `gold_answer` exact-match path in
`runner.py` — `gold_answer` is the letter of the grammatical sentence. No
runner changes required.
`persian_stability` rubric applied to raw output; `verifiable_reasoning` is
not applied (a hard gold answer already exists).

## Why Closed-Set Instead of Free-Text Grammaticality Rating?

Asking a model to freely explain "what's wrong" with a sentence, or to
output a 1–5 naturalness score, is brittle to grade automatically and
introduces judge-model subjectivity for what is fundamentally a binary,
rule-governed fact. Presenting the grammatical/ungrammatical pair together
and asking for a single option letter keeps the interesting part of the
task (the model has to actually notice the minimal edit) while making
scoring a reliable, cheap exact-match — the same tradeoff Aroozi and
script-disambiguation make.

## Configuration

`scripts/prepare_minimal_pairs_task.py` exposes:
- `--seed` (default 42) — controls deterministic, reproducible الف/ب
  position assignment.

There is no `--num-options` (always exactly 2, by construction).

## Known Limitations

- 8 categories × 8 pairs is intentionally "lite" — enough to catch a model
  that's fluent-but-inconsistent on each rule, not an exhaustive grammar
  suite. Extending coverage (e.g. ezafe constructions, more verb-tense
  phenomena, agreement in relative clauses) is straightforward: add more
  `(correct_sentence, incorrect_sentence)` pairs under new or existing
  `category` keys in the raw JSON and re-run the prep script.
- Some phenomena (e.g. ezafe marking) were deliberately excluded because
  standard Persian orthography doesn't write the ezafe vowel, so a written
  minimal pair couldn't reliably isolate the feature without relying on
  diacritics the model wouldn't normally see.
- As with the other closed-set tasks, a model that picks the right sentence
  but wraps its answer in a format the extraction regex doesn't expect will
  be scored as incorrect despite understanding the grammar correctly.

## Running this task

```bash
python scripts/prepare_minimal_pairs_task.py \
  --input data/raw_sources/minimal_pairs_fa_raw.json \
  --output data/tasks/minimal_pairs_fa.jsonl \
  --seed 42

python -m src.pipeline.runner \
  --task-file data/tasks/minimal_pairs_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/minimal_pairs_fa__gpt-oss-20b-fa-cot-v1.jsonl

python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```
