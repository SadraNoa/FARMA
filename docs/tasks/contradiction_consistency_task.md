# Phase 3 — Contradiction & Consistency (NLI) Task

## Overview

Implements the **contradiction & consistency** sub-task of Phase 3: a
standard 3-way Natural Language Inference (NLI) benchmark, in the tradition
of SNLI/MNLI/FarsTail. Given a premise and a hypothesis, the model must
classify their logical relationship as one of:

- **استلزام (entailment)** — if the premise is true, the hypothesis must be true.
- **تناقض (contradiction)** — the premise and hypothesis cannot both be true.
- **خنثی (neutral)** — neither necessarily follows nor conflicts; the hypothesis could be true or false independent of the premise.

This directly targets what the sub-task name describes: can the model tell
when two statements are mutually incompatible (contradiction) versus
compatible-and-connected (entailment) versus merely compatible-but-unrelated
(neutral)? Like Minimal Pairs, this is a closed-set task built entirely on
Phase 0/1 infrastructure — no `runner.py` changes were needed.

## Data

`data/raw_sources/contradiction_consistency_fa_raw.json` — 60 hand-authored
samples (all original content), built from **20 premises**, each paired
with exactly one entailed, one contradicting, and one neutral hypothesis —
perfectly balanced 20/20/20 across labels. Grouping three controlled
hypotheses under a shared premise (tracked via `premise_group`) mirrors the
"one family, N variants" pattern used by script-disambiguation and minimal
pairs: it prevents the label from being guessable off surface features of
the premise alone (e.g. topic or sentence length), since the same premise
appears with all three labels.

Premises span everyday topics — daily habits, weather, exam results,
prices, sports outcomes, company finances, schedules — deliberately kept
simple and unambiguous so the correct relation is a fact about logic, not a
judgment call about real-world plausibility.

Each raw record: `id`, `premise_group`, `premise`, `hypothesis`,
`gold_relation` (one of `entailment` / `contradiction` / `neutral`).

### Task Preparation Script

`scripts/prepare_contradiction_consistency_task.py`:
- Presents the premise and hypothesis, then all **three** relation labels
  (with a one-line definition each) as options, in a per-sample
  deterministically shuffled order (`random.Random(f"{seed}-{raw_id}")`) so
  the model can't exploit a fixed position bias (e.g. "answer is always ب").
- Unlike Aroozi/MMLU-lite, there's no distractor *pool* to sample from — the
  three options are always exactly the three NLI labels; only their display
  order changes.
- Validates `gold_relation` is one of the three valid labels and both
  sentences are present; skips/logs anything malformed.
- Produces `data/tasks/contradiction_consistency_fa.jsonl` (60/60
  converted, 0 skipped).

## Pipeline Integration

Reuses the framework's generic `gold_answer` exact-match path in
`runner.py` — `gold_answer` is the letter of the correct relation label. No
runner changes required.
`persian_stability` rubric applied to raw output; `verifiable_reasoning` is
not applied (a hard gold answer already exists).

## Why 3-Way Labels Instead of Binary Contradiction Detection?

A binary "متناقض / سازگار" framing collapses entailment and neutral into
one bucket, which hides a common failure mode: a model that thinks every
compatible-but-unrelated hypothesis is "obviously implied" (mistaking
neutral for entailment) looks identical to a well-calibrated model under a
binary score, but is not actually reasoning correctly. The standard 3-way
NLI split (used by SNLI/MNLI and the Persian FarsTail benchmark) keeps that
distinction visible while remaining a clean, cheap-to-score closed-set
choice — the same tradeoff Aroozi and Minimal Pairs make for their own
domains.

## Configuration

`scripts/prepare_contradiction_consistency_task.py` exposes:
- `--seed` (default 42) — controls deterministic, reproducible option-order
  shuffling.

There is no `--num-options` (always exactly 3 — the fixed NLI label set).

## Known Limitations

- 20 premise groups is intentionally "lite" — enough to catch a model that
  systematically confuses entailment with neutral, or misses direct
  negation-based contradictions, not an exhaustive NLI suite. Extending
  coverage (harder entailments requiring multi-step inference, numerical
  reasoning, temporal reasoning) is straightforward: add more
  `(premise, entailment_hyp, contradiction_hyp, neutral_hyp)` groups to the
  raw JSON and re-run the prep script.
- The neutral/entailment boundary can be genuinely fuzzy in NLI more
  broadly; each item here was written so the entailment holds under a
  strict "must be true," and the neutral hypothesis is plausible but never
  forced — but as with any NLI dataset, a small fraction of judgment calls
  is inherent to the task format itself, not a bug in this implementation.
- As with the other closed-set tasks, a model that identifies the right
  relation but wraps its answer in a format the extraction regex doesn't
  expect will be scored as incorrect despite reasoning correctly.

## Running this task

```bash
python scripts/prepare_contradiction_consistency_task.py \
  --input data/raw_sources/contradiction_consistency_fa_raw.json \
  --output data/tasks/contradiction_consistency_fa.jsonl \
  --seed 42

python -m src.pipeline.runner \
  --task-file data/tasks/contradiction_consistency_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/contradiction_consistency_fa__gpt-oss-20b-fa-cot-v1.jsonl

python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```
