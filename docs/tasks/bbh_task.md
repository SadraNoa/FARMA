# Phase 1 — Persian Math Reasoning Benchmark: BBH-lite (Logic) Task

## Overview

This task evaluates a model's ability to perform short, deterministic
logical reasoning **in Persian**: object ordering / spatial reasoning,
navigation (return-to-origin) puzzles, boolean logic (propositional
connectives: AND / OR / NOT), and simple multi-step arithmetic word
problems. It is a lightweight, Persian-native adaptation of the spirit of
BIG-Bench Hard (BBH) — hence "BBH-lite" — focused on tasks with a single,
unambiguous correct answer rather than open-ended reasoning.

Unlike the Math task, this task is **not** primarily about judging the
quality of the reasoning chain. Every sample has a short, well-defined
`gold_answer`, so the main signal is straightforward exact-match
correctness. This makes BBH-lite a useful counterpart to the Math task: it
gives a fair, cheap read on models that were **not** specifically
fine-tuned to produce long Persian CoT, since a short correct final answer
is enough to score well here.

## Data

- **Source:** hand-authored, natively in Persian (not translated), to avoid
  translation artifacts affecting a deterministic-answer task.
- **Sample size:** 120 samples, split evenly across 4 categories (30 each):

| Category | Description |
|---|---|
| `object_ordering` | Ordering people/objects in a queue or row from a set of relative-position clues (e.g. "in front of", "behind", "left of", "right of"). |
| `navigate` | Sequences of cardinal-direction steps (north/south/east/west); the model must determine whether the final position coincides with the starting point. |
| `boolean_logic` | Propositional logic questions using AND / OR / NOT (و / یا / نه) over 2–3 boolean variables, including negated compounds. |
| `arithmetic` | Short multi-step word problems involving addition and subtraction of everyday objects (apples, pencils, boxes, etc.). |

### Raw format (`data/raw_sources/bbh_lite_fa_raw.json`)

```json
{
  "category": "object_ordering",
  "question": "چهار نفر علی، رضا، سارا و مریم در صف هستند. علی جلوتر از رضا و سارا جلوتر از علی است. مریم پشت رضا است. ترتیب صف از جلو به عقب چیست؟",
  "answer": "سارا، علی، رضا، مریم"
}
```

### Task-ready format (`data/tasks/bbh_lite_fa.jsonl`)

Produced by `scripts/prepare_bbh_task.py`, which validates each sample's
`category` against the fixed set of 4 known categories and wraps it into
the framework's shared `TaskSample` schema:

```json
{
  "sample_id": "bbh_object_ordering_0000",
  "task_type": "bbh_lite",
  "problem_fa": "چهار نفر علی، رضا، سارا و مریم در صف هستند. ...",
  "gold_answer": "سارا، علی، رضا، مریم",
  "requires_cot_judging": false,
  "apply_verifiable_reasoning_rubric": false,
  "apply_persian_stability_rubric": true,
  "system_prompt_fa": "تو یک دستیار منطقی و دقیق فارسی‌زبان هستی. ...",
  "answer_extraction_regex": "پاسخ نهایی:\\s*(.+)",
  "extra": {
    "category": "object_ordering"
  }
}
```

`sample_id` encodes the category so that per-category accuracy can be
sliced out of `data/scored_results/*.jsonl` at aggregation time by joining
on the (implicit) prefix or via `extra.category` once merged back in.

## Why exact-match here but not in the Math task?

The Math task deliberately has no `gold_answer` because the point is to
test *reasoning quality*, and a single correct final number says little
about that. BBH-lite is the opposite case by design: every sample was
authored to have exactly one short, checkable correct answer (an ordering,
a "بله"/"خیر", or a number), so exact-match is both sufficient and far
cheaper than judge-model scoring. This also makes BBH-lite the right
control group to compare against Math and other CoT-heavy tasks when
separating "the model can reason at all" from "the model produces
high-quality long-form Persian CoT."

## Scoring methodology

- **`correctness` (0/1):** computed automatically by the shared runner
  (`src/pipeline/runner.py`) by comparing `extracted_final_answer` against
  `gold_answer` after light normalization (`normalize_fa_text`): trimming
  whitespace, unifying Arabic/Persian `ي`/`ك` variants, and stripping
  trailing punctuation. No judge model is involved in this comparison.
- **`persian_stability`** (`rubrics/persian_stability.json`) is still
  applied to the raw output, combining rule-based metrics (code-switch
  rate, Persian script ratio) with LLM-judged metrics (grammatical
  fluency, terminology consistency, degradation over length). This lets us
  check whether a model stays in fluent, stable Persian even on short,
  easy reasoning — a useful contrast with the longer Math CoTs.
- **`verifiable_reasoning`** is intentionally **not** applied to this task;
  with a hard gold answer already available, judging the reasoning chain's
  internal logical validity adds cost without much additional signal.

## Running this task

```bash
# 1. (Already done) Convert the raw hand-authored data into task-ready format:
python scripts/prepare_bbh_task.py \
  --input data/raw_sources/bbh_lite_fa_raw.json \
  --output data/tasks/bbh_lite_fa.jsonl

# 2. Run a candidate model against the task:
python -m src.pipeline.runner \
  --task-file data/tasks/bbh_lite_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/bbh_lite_fa__gpt-oss-20b-fa-cot-v1.jsonl

# 3. Aggregate results across all tasks/models:
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

- Exact-match on `object_ordering` answers is somewhat format-sensitive
  (e.g. a model listing the same correct order but joined with "و" instead
  of "،", or using different spacing) can be marked incorrect even though
  the ordering itself is right. `normalize_fa_text` only handles whitespace,
  a couple of character variants, and trailing punctuation, not this kind
  of formatting variance — worth a spot-check pass before trusting
  aggregate `object_ordering` accuracy at face value.
- `arithmetic` gold answers use Persian digits (۰–۹); a model answering in
  Latin digits will fail exact-match unless `answer_extraction_regex`
  captures it and normalization is extended to map digit systems. This is
  a known gap, not yet handled by `normalize_fa_text`.
- The 4 categories are fixed and hand-authored rather than sampled from a
  larger pool, so this task measures competence on these specific pattern
  families, not general logical reasoning coverage. Treat it as a
  lightweight signal, not a substitute for a full BBH-style suite.
