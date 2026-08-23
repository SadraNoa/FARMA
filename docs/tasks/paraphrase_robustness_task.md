# Phase 3 — Paraphrase Robustness Task

## Overview

Implements the **paraphrase robustness** sub-task of Phase 3: tests
whether a model's answer to a factual question stays correct — and stays
*the same* — when the underlying question is reworded, with the fact being
asked about held fixed. A model that answers a question correctly once but
flips its answer when the exact same question is phrased differently isn't
demonstrating real understanding; it's pattern-matching on surface wording.

Like Abstention and Contradiction/Consistency, this is a closed-set task
built entirely on Phase 0/1 infrastructure — no `runner.py` changes needed.

## Data

`data/raw_sources/paraphrase_robustness_fa_raw.json` — 60 hand-authored
samples (all original content), built from **15 families**. Each family is
one `(passage, correct_answer, 3 foils)` tuple expressed as **4
differently-worded questions** that all ask exactly the same thing —
varying word order, synonyms, formal/informal register, and direct vs.
indirect question form.

Passages are short, neutral, fictional-style factual statements (business
hours, budgets, production numbers, event attendance) — the same kind of
content used in the Abstention task — deliberately simple so there's never
ambiguity about what the "same fact" is across paraphrases.

Each raw record: `id`, `family_id`, `paraphrase_index` (1–4), `passage`,
`question` (this sample's specific wording), `correct_answer`, `foils`
(exactly 3 plausible-wrong answers, shared across all 4 paraphrases of the
family).

### Task Preparation Script

`scripts/prepare_paraphrase_robustness_task.py`:
- Every paraphrase in a family shares the same passage, the same correct
  answer, and the same 3 foils — only the question's wording differs. This
  isolates the one variable the task is designed to probe.
- Option **display order** is still shuffled per-sample
  (`random.Random(f"{seed}-{raw_id}")`, keyed by the individual sample's
  `id`, not the shared `family_id`) — so the correct letter can land in a
  different position across the 4 paraphrases of one family. This is
  deliberate: without it, a model that always answers "the same letter as
  last time" within a family could look artificially robust for the wrong
  reason (position memorization instead of actually re-reading each
  reworded question).
- Validates exactly 3 foils are present, distinct from the correct answer,
  and both `family_id`/`paraphrase_index` are set; skips/logs anything
  malformed.
- Produces `data/tasks/paraphrase_robustness_fa.jsonl` (60/60 converted, 0
  skipped, 15 families of exactly 4 paraphrases each).

## Pipeline Integration

Reuses the framework's generic `gold_answer` exact-match path in
`runner.py` — `gold_answer` is the letter of the correct option for that
specific paraphrase. No runner changes required.
`persian_stability` rubric applied to raw output; `verifiable_reasoning` is
not applied (a hard gold answer already exists).

Robustness itself is a **family-level aggregate**, not a per-sample field:
whether a model is "robust" on family N is whether all 4 of its
`family_id == N` results are individually correct (and, more loosely,
whether they're *mutually consistent* even when wrong — e.g. always picking
the same wrong foil across paraphrases is a different failure mode than
picking a different wrong foil each time). `extra.family_id` and
`extra.paraphrase_index` are carried through into the scored output
specifically so this can be computed at aggregation time without needing
any change to the runner's per-sample scoring path.

## Why Vary Option Order Across Paraphrases Instead of Keeping It Fixed?

An earlier design considered keeping option order identical across all 4
paraphrases in a family, so any letter-flip could be attributed purely to
the reworded question. That was rejected: a fixed option order per family
means a model could get all 4 "right" by memorizing a single letter for the
family early on and repeating it, regardless of whether it re-read each
paraphrase — which would make the metric measure the wrong thing.
Reshuffling per-sample forces the model to actually resolve the question
text to a specific option every time, which is what the task is meant to
measure in the first place.

## Configuration

`scripts/prepare_paraphrase_robustness_task.py` exposes:
- `--seed` (default 42) — controls deterministic, reproducible option-order
  shuffling.

There is no `--num-options` (always exactly 4: the correct answer + the 3
authored foils).

## Known Limitations

- 15 families × 4 paraphrases is intentionally "lite" — enough to catch a
  model whose answers are sensitive to superficial rewording, not an
  exhaustive robustness suite. Extending coverage (more paraphrase types
  per family — e.g. adding irrelevant clauses, negation-of-negation
  rephrasings) is straightforward: add more `(passage, correct_answer,
  foils, questions)` groups to the raw JSON and re-run the prep script.
- All paraphrases in this version are questions about a single
  numeric/short factual value; paraphrase robustness for open-ended or
  multi-part answers is out of scope for this "lite" version.
- As with the other closed-set tasks, a model that identifies the right
  option but wraps its answer in a format the extraction regex doesn't
  expect will be scored as incorrect despite reasoning correctly.
- This implementation scores each paraphrase independently; computing the
  actual "robustness rate" (fraction of families where all 4 paraphrases
  agree) requires a small aggregation step over `extra.family_id` that is
  not itself part of this PR — it groups cleanly with existing scored
  output, so no schema or runner change is needed to add it later.

## Running this task

```bash
python scripts/prepare_paraphrase_robustness_task.py \
  --input data/raw_sources/paraphrase_robustness_fa_raw.json \
  --output data/tasks/paraphrase_robustness_fa.jsonl \
  --seed 42

python -m src.pipeline.runner \
  --task-file data/tasks/paraphrase_robustness_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/paraphrase_robustness_fa__gpt-oss-20b-fa-cot-v1.jsonl

python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```
