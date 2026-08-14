# Math Task (Persian Reasoning)

## Overview

This task evaluates a model's ability to solve mathematical problems while
reasoning **in Persian**, using a chain-of-thought (CoT) generated entirely
in Persian. Unlike a typical math benchmark, the primary signal here is not
"did you get the right number" — it's **whether the reasoning chain itself
is logically sound, complete, and free of unjustified jumps**, evaluated in
the language the model was fine-tuned to reason in.

This design choice follows directly from how the underlying model family was
trained: the CoT-capable variants were fine-tuned specifically to reason in
Persian, so the benchmark needs to test that capability directly rather than
only checking a final numeric answer.

## Data

- **Source:** [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning),
  a large public dataset of math problems originally in English.
- **Sample size:** 200 problems were translated to Persian; 1 was dropped
  due to a translation/formatting error, leaving **199 usable problems**.
- **Translation method:** Problems were translated with LaTeX expressions
  extracted and replaced by placeholders (`[LATEX_1]`, `[LATEX_2]`, ...) to
  avoid corrupting mathematical notation during translation, then
  re-inserted into the Persian text.

### Raw format (`data/raw_sources/openmathreasoning_200_translated.json`)

```json
{
  "problem_source": "aops_c4_high_school_math",
  "problem_en": "If r, s, t are prime numbers and lcm(p,q) = r^2 t^4 s^2 ...",
  "problem_fa": "اگر [LATEX_1] اعداد اول باشند و [LATEX_2] ...",
  "latex": [
    "r,s,t",
    "\\operatorname{lcm}(p,q)=r^2t^4s^2",
    "p,q\\in\\mathbb{Z}^{+}",
    "(p,q)"
  ]
}
```

### Task-ready format (`data/tasks/math_fa.jsonl`)

Produced by `scripts/prepare_math_task.py`, which fills the `[LATEX_N]`
placeholders back into the Persian text (wrapped in `$...$`) and wraps each
problem into the framework's shared `TaskSample` schema:

```json
{
  "sample_id": "math_0000",
  "task_type": "math",
  "problem_fa": "مجموعه‌ای شامل $N$ توپ با $C$ رنگ در نظر بگیرید ...",
  "gold_answer": null,
  "requires_cot_judging": true,
  "apply_verifiable_reasoning_rubric": true,
  "apply_persian_stability_rubric": true,
  "system_prompt_fa": "تو یک ریاضی‌دان دقیق و فارسی‌زبان هستی. ...",
  "answer_extraction_regex": "پاسخ نهایی:\\s*(.+)",
  "extra": {
    "problem_source": "aops_c6_high_school_olympiads",
    "problem_en": "Given a group of N balls ..."
  }
}
```

Note `gold_answer` is intentionally `null` for every sample in this task.

## Why no gold answer?

Exact-match scoring against a reference answer would only tell us whether
the final number is correct — it says nothing about whether the reasoning
that produced it was valid, in Persian, and free of logical gaps. Since the
main hypothesis being tested by this framework is that fine-tuning improved
**Persian reasoning quality** (not just final-answer accuracy), this task
is scored purely on the reasoning chain via the judge model.

If exact-answer accuracy is needed later, `gold_answer` can be added to
`data/tasks/math_fa.jsonl` per sample and `correctness` will be computed
automatically by the runner (it already supports this — it's just unused
for this task by design).

## Scoring methodology

Each generated response is scored using the shared **`verifiable_reasoning`**
rubric (`rubrics/verifiable_reasoning.json`), applied by the judge model
(configured under `judge_models` in `configs/models.yaml`):

| Criterion | Scale | What it checks |
|---|---|---|
| `step_support` | 0–3 | Is each step supported by the problem statement or the previous step? |
| `conclusion_validity` | 0–3 | Does the final conclusion logically follow from the premises? |
| `no_logical_jump` | 0–3 (reverse-scored) | Are there unjustified jumps or missing steps? |
| `answer_consistency` | 0–3 | Is the final answer consistent with the reasoning shown (not just accidentally correct)? |

**Total: 0–12.**

In addition, the **`persian_stability`** rubric (`rubrics/persian_stability.json`)
is applied to the same CoT output, combining rule-based metrics (code-switch
rate to English, Persian script ratio) with LLM-judged metrics (grammatical
fluency, terminology consistency, quality degradation across the length of
the response). This is what lets the framework separately report "the model
reasons correctly" vs. "the model reasons correctly *in stable Persian*".

## Running this task

```bash
# 1. (Already done) Convert raw translated data into task-ready format:
python scripts/prepare_math_task.py \
  --input data/raw_sources/openmathreasoning_200_translated.json \
  --output data/tasks/math_fa.jsonl

# 2. Run a candidate model against the task:
python -m src.pipeline.runner \
  --task-file data/tasks/math_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/math_fa__gpt-oss-20b-fa-cot-v1.jsonl

# 3. Aggregate results across all tasks/models:
python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

## Output fields relevant to this task

In `data/scored_results/*.jsonl`, each row (`ScoredResult`) includes:

- `correctness`: always `null` for this task (no gold answer).
- `verifiable_reasoning_total` (0–12) and its per-criterion breakdown.
- `persian_stability_llm_total` (0–8), `code_switch_rate`, `persian_script_ratio`.
- `raw_output`: the full model response, kept for manual spot-checking.

## Known limitations

- This task currently only covers CoT-capable model variants meaningfully —
  non-reasoning variants can still be run through it, but their score will
  mostly reflect the absence of a step-by-step chain rather than reasoning
  quality per se. Compare against the BBH-lite and IFEval-lite tasks for a
  fairer read on non-reasoning variants.
- The judge model itself is fallible; spot-check a sample of `raw_output`
  and `verifiable_reasoning_breakdown.brief_reason_fa` fields before trusting
  aggregate scores at face value, especially early on.
