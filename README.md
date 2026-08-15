# FARMA — Farsi Reasoning & Multi-dimensional Assessment

A benchmark framework for evaluating language models fine-tuned for Persian
(built on top of `gpt-oss-20b`), with special focus on the quality and
stability of Persian chain-of-thought (CoT) reasoning.

## Current status: infrastructure + Math + BBH-lite + Jalali calendar + IFEval-lite + Aroozi tasks

This version includes the **shared infrastructure** of the project plus
five **Phase 1 tasks**: Math, BBH-lite (logic), Jalali calendar,
IFEval-lite, and Aroozi (Persian classical meter).

### Math task
- 199 problems from OpenMathReasoning, translated to Persian (1 sample was
  dropped due to a translation error).
- This task has **no gold_answer**. The goal is to evaluate the quality of
  the reasoning chain itself, not just exact-match on a final answer.
  Scoring is done entirely through the `verifiable_reasoning` rubric via the
  judge model.
- `scripts/prepare_math_task.py` converts the raw translated data (with
  `[LATEX_N]` placeholders) into the executable `data/tasks/math_fa.jsonl`
  format.

See `docs/tasks/math_task.md` for full details on data format and scoring
methodology.

### BBH-lite (logic) task
- 120 hand-authored Persian samples, split evenly across 4 categories (30
  each): `object_ordering`, `navigate`, `boolean_logic`, `arithmetic`.
- This task **has a gold_answer** for every sample, so `correctness` is
  computed automatically by the runner via exact-match (after light
  normalization). The `persian_stability` rubric is still applied to check
  language stability even on short answers; `verifiable_reasoning` is not
  applied since a hard gold answer already exists.
- `scripts/prepare_bbh_task.py` converts the raw hand-authored data into
  the executable `data/tasks/bbh_lite_fa.jsonl` format.

See `docs/tasks/bbh_task.md` for full details on data format and scoring
methodology.

### Jalali calendar task
- 150 fully synthetic samples, generated deterministically with the
  `jdatetime` library (no manual authoring or translation), split across 4
  categories: `date_conversion` (38), `leap_year` (38), `date_arithmetic`
  (37), `date_distance` (37).
- Every sample has a `gold_answer` guaranteed correct by construction, so
  `correctness` is computed automatically by the runner via exact-match.
  `persian_stability` is applied; `verifiable_reasoning` is not.
- `scripts/generate_jalali_calendar_data.py` produces the raw source file;
  `scripts/prepare_jalali_task.py` converts it into the executable
  `data/tasks/jalali_calendar_fa.jsonl` format.

See `docs/tasks/jalali_calendar_task.md` for full details on data format
and scoring methodology.

### IFEval-lite task
- 120 fully synthetic samples: 40 hand-picked Persian topics × 3
  rule-verifiable instruction constraints each (`length_constraint`,
  `keyword_constraint`, `position_constraint`).
- Samples have **no `gold_answer`**. Instead, each carries a structured
  constraint spec under `extra.constraint`, checked by a rule-based
  verifier (`src/utils/instruction_utils.py`) — no judge model involved.
  `persian_stability` is still applied; `verifiable_reasoning` is not.
- `scripts/generate_ifeval_lite_data.py` produces the raw source file;
  `scripts/prepare_ifeval_task.py` converts it into the executable
  `data/tasks/ifeval_lite_fa.jsonl` format.
- This is the only task so far that required a small addition to
  `src/pipeline/runner.py` (one `elif` branch in `score_sample()`), since
  rule-based constraint checking is a genuinely different scoring path
  from exact-match or judge rubrics. See `docs/tasks/ifeval_task.md` for
  details.

See `docs/tasks/ifeval_task.md` for full details on data format and
scoring methodology.

### Aroozi (Persian classical meter) task
- 100 hand-curated real verses from canonical poets (حافظ، سعدی، مولانا و...),
  drawn from 10 distinct classical عروضی meters (~10 verses per meter).
- Each verse is converted into a **multiple-choice** question: the model
  must pick the correct meter out of `--num-options` candidates (1 correct
  + distractors sampled from the other meters present in the dataset), so
  `gold_answer` is always a single option letter (`الف`/`ب`/`ج`/`د`/...).
  `correctness` is computed automatically by the runner via exact-match,
  the same as BBH-lite and Jalali calendar — no runner changes were
  needed. `persian_stability` is applied; `verifiable_reasoning` is not.
- `scripts/prepare_aroozi_task.py` converts the raw hand-curated data
  (`data/raw_sources/aroozi_meter_fa_raw.json`) into the executable
  `data/tasks/aroozi_meter_fa.jsonl` format, generating distractor options
  deterministically (seeded per-sample) so option sets are reproducible.

See `docs/tasks/aroozi_task.md` for full details on data format and
scoring methodology.

## Project structure

```
FARMA/
├── configs/
│   └── models.yaml          # Candidate models (vLLM) and judge models (OpenRouter/GapGPT)
├── rubrics/
│   ├── verifiable_reasoning.json   # Logical quality of reasoning chain (0-12)
│   └── persian_stability.json      # Persian language stability (rule-based + LLM, 0-8)
├── src/
│   ├── providers/
│   │   ├── base.py                 # Abstract Provider class
│   │   ├── openai_compatible.py    # Shared implementation for vLLM/OpenRouter/GapGPT
│   │   └── factory.py              # Builds a provider from configs/models.yaml
│   ├── judges/
│   │   ├── rubric_judge.py             # Judge for the verifiable_reasoning rubric
│   │   └── persian_stability_judge.py  # Judge for the persian_stability rubric
│   ├── utils/
│   │   ├── text_utils.py           # Rule-based helpers: code-switching, answer extraction, normalization
│   │   └── instruction_utils.py    # Rule-based IFEval-lite constraint checkers
│   └── pipeline/
│       ├── schemas.py              # Data models (TaskSample, ModelGeneration, ScoredResult)
│       ├── runner.py               # Generic runner: generates model output + applies rubrics
│       └── aggregate.py            # Aggregates results into a summary report
├── scripts/
│   ├── prepare_math_task.py                # Converts raw math data into task-ready jsonl
│   ├── prepare_bbh_task.py                 # Converts raw BBH-lite data into task-ready jsonl
│   ├── generate_jalali_calendar_data.py    # Generates the raw Jalali calendar data
│   ├── prepare_jalali_task.py              # Converts raw Jalali calendar data into task-ready jsonl
│   ├── generate_ifeval_lite_data.py        # Generates the raw IFEval-lite data
│   ├── prepare_ifeval_task.py              # Converts raw IFEval-lite data into task-ready jsonl
│   └── prepare_aroozi_task.py              # Converts raw Aroozi data into task-ready multiple-choice jsonl
├── data/
│   ├── raw_sources/      # Raw, unprocessed source data per task
│   ├── tasks/            # Task-ready jsonl files (input to the runner)
│   ├── raw_outputs/      # Raw model outputs (gitignored)
│   └── scored_results/   # Scored results (gitignored)
├── docs/
│   └── tasks/             # One methodology doc per task
├── tests/
│   ├── test_text_utils.py
│   ├── test_instruction_utils.py
│   └── test_prepare_aroozi_task.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt --break-system-packages
cp .env.example .env
# fill in .env with your real API keys / endpoints
```

## Models

- **Candidate models (under evaluation):** served via vLLM
  (`configs/models.yaml` → `candidate_models`).
- **Judge models (LLM-judge):** called via OpenRouter or GapGPT
  (`configs/models.yaml` → `judge_models`). Both are OpenAI-compatible, so a
  single shared class (`OpenAICompatibleProvider`) covers all three
  (vLLM/OpenRouter/GapGPT).

Before running anything, edit `configs/models.yaml` and replace the
placeholder `model_id` values with your actual model names.

## Two shared rubrics

### 1. `verifiable_reasoning` (0–12)
Measures the logical quality of the reasoning chain: whether each step is
supported by the previous one, whether there are logical jumps, whether the
conclusion follows from the premises. Used across Math, BBH, contradiction &
consistency, minimal pairs, abstention, and deep brainstorm tasks.

### 2. `persian_stability` (0–8 for the LLM part, plus two rule-based metrics)
Measures the stability and quality of Persian throughout the reasoning
chain: unwanted code-switching rate to English (rule-based, free), ratio of
Persian script characters (rule-based), grammatical fluency, terminology
consistency, and quality degradation between the first and last third of the
text (all three via LLM-judge). Applicable to any task with CoT.

## How to run a task

```bash
python -m src.pipeline.runner \
  --task-file data/tasks/math_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/math_fa__gpt-oss-20b-fa-cot-v1.jsonl

python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

## Tests

```bash
pytest tests/ -v
```

## Roadmap

- [x] Phase 0: shared infrastructure
- [x] Phase 1 — Math (done)
- [x] Phase 1 — BBH-lite (logic) (done)
- [x] Phase 1 — Jalali calendar (done)
- [x] Phase 1 — IFEval-lite (done)
- [x] Phase 1 — Aroozi (meter) (done)
- [ ] Phase 2 — MMLU-lite, script disambiguation, proverbs, wordplay (ieham)
- [ ] Phase 3 — Minimal pairs, contradiction & consistency, abstention, paraphrase robustness, multi-constraint
- [ ] Phase 4 — Persian controllability, deep brainstorm
