# FARMA — Farsi Reasoning & Multi-dimensional Assessment

A benchmark framework for evaluating language models fine-tuned for Persian
(built on top of `gpt-oss-20b`), with special focus on the quality and
stability of Persian chain-of-thought (CoT) reasoning.

## Current status: infrastructure + Math task

This version includes the **shared infrastructure** of the project plus the
**first Phase 1 task (Math)**.

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
│   │   └── text_utils.py           # Rule-based helpers: code-switching, answer extraction, normalization
│   └── pipeline/
│       ├── schemas.py              # Data models (TaskSample, ModelGeneration, ScoredResult)
│       ├── runner.py               # Generic runner: generates model output + applies rubrics
│       └── aggregate.py            # Aggregates results into a summary report
├── scripts/
│   └── prepare_math_task.py        # Converts raw math data into task-ready jsonl
├── data/
│   ├── raw_sources/      # Raw, unprocessed source data per task
│   ├── tasks/            # Task-ready jsonl files (input to the runner)
│   ├── raw_outputs/      # Raw model outputs (gitignored)
│   └── scored_results/   # Scored results (gitignored)
├── docs/
│   └── tasks/             # One methodology doc per task
├── tests/
│   └── test_text_utils.py
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
- [ ] Phase 1 — BBH-lite (logic), Jalali calendar, IFEval-lite, Aroozi (meter)
- [ ] Phase 2 — MMLU-lite, script disambiguation, proverbs, wordplay (ieham)
- [ ] Phase 3 — Minimal pairs, contradiction & consistency, abstention, paraphrase robustness, multi-constraint
- [ ] Phase 4 — Persian controllability, deep brainstorm
