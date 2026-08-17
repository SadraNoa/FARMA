# Phase 4 — Deep Brainstorm Task

## Overview
Evaluates depth and diversity of the reasoning chain on open-ended
brainstorming problems, not just the correctness of a final answer.
Targets the model's Persian chain-of-thought quality specifically:
whether it explores genuinely distinct approaches, self-corrects, and
localizes ideas to Iranian/Persian-speaking context rather than reading
as a direct translation of an English-default answer.

Unlike Math/BBH/Jalali/Aroozi/MMLU-lite, this task has **no gold_answer**
and is scored entirely through LLM-judge rubrics — three of them, stacked:

1. **verifiable_reasoning** (existing, shared rubric) — logical quality of
   the reasoning chain: step_support, conclusion_validity, no_logical_jump,
   answer_consistency. This rubric already listed `"deep_brainstorm"` in
   its `applies_to` before this task was implemented, so no changes to
   `rubrics/verifiable_reasoning.json` were needed.
2. **persian_stability** (existing, shared rubric) — same as every other
   CoT task: code-switching rate, script ratio, grammatical fluency,
   terminology consistency, degradation over length.
3. **deep_brainstorm** (new, `rubrics/deep_brainstorm.json`) — the
   dimensions specific to this task that the two rubrics above don't
   cover: `idea_diversity` (0-3), `self_correction` (0-2),
   `cultural_grounding` (0-3).

The new rubric is deliberately a companion, not a replacement: reasoning
depth and trace-answer coherence are already reasonably captured by
verifiable_reasoning's `no_logical_jump` and `answer_consistency`, so
`deep_brainstorm.json` only adds what's genuinely missing.

## Why the "trace vs. final answer" split isn't needed here
An earlier draft of this task considered splitting model output into a
separate reasoning trace and final answer (motivated by the Harmony
format's `analysis`/`final` channels used by gpt-oss models). This turned
out to be unnecessary given how the rest of FARMA already works: every
CoT task in this framework asks the model to end its response with a
`پاسخ نهایی: <answer>` line (see `DEFAULT_SYSTEM_PROMPT_FA` in
`runner.py`), and judges are simply given the *entire* raw output as
`cot` (see `rubric_judge.py`'s `VerifiableReasoningJudge.score()`). This
task follows that same convention — no special provider-level handling of
reasoning-channel fields was needed.

## Implemented Components

### 1. Schema extension
`src/pipeline/schemas.py`: added `apply_deep_brainstorm_rubric: bool` to
`TaskSample`, and `deep_brainstorm_total` / `deep_brainstorm_breakdown` to
`ScoredResult`.

### 2. Rubric
`rubrics/deep_brainstorm.json` — same JSON shape as
`verifiable_reasoning.json` (criteria list with anchors, system/user
prompt templates), so it plugs into the existing judge pattern directly.

### 3. Judge
`src/judges/deep_brainstorm_judge.py` — `DeepBrainstormJudge`, structured
identically to `VerifiableReasoningJudge` (single JSON judge call, same
error handling for unparseable output).

### 4. Runner integration
`src/pipeline/runner.py`: added `needs_db` detection, `db_judge`
instantiation (only when at least one sample requests it), and a
corresponding scoring branch in `score_sample()` — same pattern as the
existing `vr_judge`/`ps_judge` branches, no restructuring of the function.

### 5. Data
- `data/raw_sources/deep_brainstorm_fa_raw.json` — 4 hand-authored
  open-ended Persian prompts across 4 categories (`open_problem_solving`,
  `creative_ideation`, `policy_or_social`, `product_or_design`).
- `scripts/prepare_deep_brainstorm_task.py` — converts raw prompts into
  `data/tasks/deep_brainstorm_fa.jsonl` (TaskSample format), validating
  category membership and rejecting malformed samples.

### 6. Tests
`tests/test_prepare_deep_brainstorm_task.py` — malformed-sample rejection,
`TaskSample` schema conformance, all-three-rubrics-enabled check, category
preservation, unique sample IDs.

## Known limitations
- Only 4 seed prompts — enough to validate the pipeline end-to-end, not
  enough for a statistically meaningful score. Expand before treating
  results as representative.
- `deep_brainstorm.json`'s criteria have not yet been validated against
  real judge output (only smoke-tested against a fake server returning
  canned JSON). Anchor wording may need adjustment once real judge
  responses are reviewed.

## Status
Implemented and smoke-tested end-to-end through the real `runner.py`
(fake evaluee + fake judge server). Not yet run against a real vLLM
instance or real judge API.
