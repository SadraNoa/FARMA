# Phase 3 — Abstention Task

## Overview

Implements the **abstention** sub-task of Phase 3: tests whether a model
correctly recognizes the *limits* of what it can answer, instead of always
confidently producing some answer. This is a well-known LLM calibration
failure mode — a model that always answers looks fine on answerable
questions but silently hallucinates whenever the needed information is
missing or the question's premise is false.

Like Minimal Pairs and Contradiction/Consistency, this is a closed-set task
built entirely on Phase 0/1 infrastructure — no `runner.py` changes needed.

## Data

`data/raw_sources/abstention_fa_raw.json` — 54 hand-authored samples (all
original content), built from **18 short passages**, each paired with
exactly three questions:

| Subtype | What it tests |
|---|---|
| `answerable` | The passage directly states the fact the question asks for — the model must give the real answer. |
| `insufficient_info` | The question is on-topic but the passage never states that fact — the model must recognize it cannot answer, not guess. |
| `false_premise` | The question presupposes something the passage explicitly contradicts — the model must flag the false premise instead of answering the presupposed question. |

Grouping three question types under a shared passage (tracked via
`family_id`) follows the same "one family, N controlled variants" pattern
used by script-disambiguation, minimal pairs, and contradiction/
consistency: because every family contains all three question types, the
model can't succeed by adopting a fixed strategy like "always answer" or
"always hedge."

Each raw record: `id`, `family_id`, `subtype`, `passage`, `question`,
`real_answer` (the passage's actual fact), `foil_answer` (a
plausible-but-wrong fact used as a distractor).

### Task Preparation Script

`scripts/prepare_abstention_task.py`:
- For **every** question — regardless of subtype — presents the exact same
  four candidate options: the real answer, the foil answer, a fixed
  "insufficient information" statement (`اطلاعات کافی در متن داده نشده
  است.`), and a fixed "false premise" statement (`پیش‌فرض این سؤال نادرست
  است.`). Only the display order changes per sample
  (`random.Random(f"{seed}-{raw_id}")`); which one is *correct* depends
  entirely on the question's subtype.
- This is what makes the task a genuine test of abstention rather than
  plain reading comprehension: a model that just learns "pick the
  answer-shaped option" fails the `insufficient_info` and `false_premise`
  items, and a model that learns to always hedge fails the `answerable`
  items.
- Validates all fields are present and `real_answer != foil_answer`;
  skips/logs anything malformed.
- Produces `data/tasks/abstention_fa.jsonl` (54/54 converted, 0 skipped,
  exactly 18/18/18 across subtypes).

## Pipeline Integration

Reuses the framework's generic `gold_answer` exact-match path in
`runner.py` — `gold_answer` is the letter of whichever option is correct
for that sample's subtype. No runner changes required.
`persian_stability` rubric applied to raw output; `verifiable_reasoning` is
not applied (a hard gold answer already exists).

## Why Fixed Shared Options Instead of Free-Text "I don't know"?

Grading free-text refusals reliably is hard: models phrase abstention in
many different ways ("I'm not sure", "the text doesn't say", "insufficient
data"), and a keyword-matching grader either over- or under-credits
legitimate hedging language, while an LLM-judge grader reintroduces
subjectivity for what should be an objective fact (was the information
actually in the passage or not?). Fixing the exact abstention phrasing as
one of a small closed set of options — the same phrasing on every single
item — removes that ambiguity entirely while still requiring the model to
make the real underlying judgment call.

## Configuration

`scripts/prepare_abstention_task.py` exposes:
- `--seed` (default 42) — controls deterministic, reproducible option-order
  shuffling.

There is no `--num-options` (always exactly 4: real answer, foil, and the
two fixed abstention statements).

## Known Limitations

- 18 families × 3 subtypes is intentionally "lite" — enough to catch a
  model with a systematic over-answering or over-hedging bias, not an
  exhaustive abstention suite. Extending coverage (multi-hop false
  premises, questions with partially-available information, ambiguous
  rather than simply false premises) is straightforward: add more
  `(passage, real_answer, foil_answer, three questions)` groups to the raw
  JSON and re-run the prep script.
- The `false_premise` questions here are all direct, single-fact
  contradictions of the passage (e.g. passage says Saturday, question
  presupposes Sunday) — subtler false premises requiring multi-step
  inference are out of scope for this "lite" version.
- As with the other closed-set tasks, a model that identifies the right
  option but wraps its answer in a format the extraction regex doesn't
  expect will be scored as incorrect despite reasoning correctly.

## Running this task

```bash
python scripts/prepare_abstention_task.py \
  --input data/raw_sources/abstention_fa_raw.json \
  --output data/tasks/abstention_fa.jsonl \
  --seed 42

python -m src.pipeline.runner \
  --task-file data/tasks/abstention_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/abstention_fa__gpt-oss-20b-fa-cot-v1.jsonl

python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```
