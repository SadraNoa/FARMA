# Phase 4 — Persian Controllability Task

## Overview
Measures whether the model can simultaneously satisfy multiple
Persian-specific stylistic/structural constraints in a single response:
register (formality level), pure-Persian lexicon, sentence structure,
paragraph count, dialect, and combinations of several at once.

This is the closest sibling to IFEval-lite in the existing task set — both
check rule-verifiable constraints against raw model output rather than
using gold-answer exact-match — but differs in two ways that required new
infrastructure rather than reusing IFEval-lite's path directly:

1. **Multiple constraints per sample.** IFEval-lite's `extra.constraint`
   holds a single constraint dict. This task needs a *list*
   (`extra.constraints`), since the whole point is testing whether several
   constraints can be satisfied together.
2. **Not all constraints are rule-checkable.** IFEval-lite's constraints
   (sentence/word/paragraph count, must-include/exclude, start/end word)
   are all deterministic. This task adds constraint types that require
   linguistic judgment — pure-Persian lexicon detection, register
   consistency, dialect matching — which a static wordlist can't reliably
   check given Persian's derivational morphology and compounding. These
   are delegated to the judge model instead.

## Constraint types and how each is checked

| Type | Values (examples) | Checked by |
|---|---|---|
| `paragraph_count` | exact integer | rule (`controllability_utils.py`) |
| `sentence_structure` | `verb_final`, `no_ke_clauses` | rule, heuristic only (see Known Limitations) |
| `lexical` | `pure_persian_only` | judge (`PersianControllabilityJudge`) |
| `register` | `formal_official`, `literary`, ... | judge |
| `dialect` | `dari`, `tajik`, `tehrani` | judge |

Each constraint also carries a `hard: bool` flag. Violating any hard
constraint forces the sample's overall result toward failure
(`controllability_hard_fail = true`), regardless of how many other
constraints were satisfied — a single unmet hard requirement (e.g. "must
be pure Persian") should not be averaged away by several easy soft
constraints being satisfied.

## Why one dynamic judge call instead of one call per constraint
An earlier draft of this task (built before this integration) issued a
separate judge API call per judge-checkable constraint plus one more for
semantic fidelity. Integrating into FARMA's existing judge pattern (one
JSON call per rubric, see `rubric_judge.py`) prompted a redesign:
`PersianControllabilityJudge` now builds ONE prompt per sample, listing
only the judge-checkable constraints actually present on that sample
(via `rubrics/persian_controllability.json`'s `constraint_prompts_fa`
templates), and asks for a single JSON response covering all of them plus
semantic fidelity together. This matches the rest of the framework's
judge-call economics and keeps behavior consistent with how
`VerifiableReasoningJudge`/`PersianStabilityJudge`/`DeepBrainstormJudge`
work.

## Semantic Fidelity
Independent of which constraints are present, every sample also gets a
1-5 semantic fidelity score: did satisfying the constraints come at the
cost of losing the original request's intended meaning? This is always
scored, even for samples with zero judge-checkable constraints (rule-only
samples still get a semantic fidelity check).

## This task does NOT use verifiable_reasoning or persian_stability
Responses here are typically short, single-purpose pieces of writing (an
announcement, a paragraph, a short passage), not long CoT chains — the
constraint/register/lexical checks already cover the language-quality
concerns that matter for this task. `apply_verifiable_reasoning_rubric`
and `apply_persian_stability_rubric` are both `false` for every sample.

## Implemented Components

### 1. Schema extension
`src/pipeline/schemas.py`: added `apply_persian_controllability_rubric`
to `TaskSample`, and `controllability_hard_fail` /
`controllability_satisfaction_rate` / `controllability_semantic_fidelity`
/ `controllability_breakdown` to `ScoredResult`.

### 2. Rule-based checkers
`src/utils/controllability_utils.py` — `check_paragraph_count`,
`check_sentence_structure` (heuristic, see Known Limitations),
dispatched via `check_rule_constraint()`. Same
`(passed: bool, reason_fa: str)` convention as
`src/utils/instruction_utils.py`.

### 3. Rubric + judge
`rubrics/persian_controllability.json` +
`src/judges/persian_controllability_judge.py` — dynamic per-sample prompt
construction (see above), single JSON call per sample.

### 4. Runner integration
`src/pipeline/runner.py`: added an `elif sample.extra.get("constraints")`
branch in `score_sample()` (parallel to IFEval-lite's single-constraint
`elif`), which checks rule constraints directly, calls the judge for
judge-checkable constraints + semantic fidelity, and combines both into
the `controllability_*` result fields. `needs_pc` detection and
`pc_judge` instantiation added to `run()`, same pattern as the other
judges.

### 5. Data
- `data/raw_sources/persian_controllability_fa_raw.json` — 7
  hand-authored samples across easy/medium/hard difficulty, including two
  dialect items, two triple-combined-hard-constraint items.
- `scripts/prepare_persian_controllability_task.py` — converts raw
  samples into `data/tasks/persian_controllability_fa.jsonl`, validating
  constraint types and difficulty, rejecting malformed/empty-constraint
  samples.

### 6. Tests
- `tests/test_prepare_persian_controllability_task.py` — malformed-sample
  rejection (empty constraints, unknown type, bad difficulty), schema
  conformance, constraint/difficulty preservation, confirms the two
  generic CoT rubrics stay off for this task.
- `tests/test_controllability_utils.py` — direct unit tests for both rule
  checkers plus the dispatch function, including the fail-closed behavior
  for unknown constraint/rule-name values.

## Known limitations
- `sentence_structure` checks (`verb_final`, `no_ke_clauses`) are simple
  heuristics, not a real parser. `verb_final` only flags sentences ending
  in a digit or trailing comma — it does not verify the last word is
  actually a verb. This under-flags violations; treat `sentence_structure`
  results as a weak signal until a better checker is written.
- Only 7 seed samples — sufficient to validate the pipeline, not for a
  statistically meaningful per-difficulty report.
- The judge prompt for dialect matching has not been validated against
  real judge output on real dialect text; only smoke-tested against a
  fake server.

## Status
Implemented and smoke-tested end-to-end through the real `runner.py`
(fake evaluee + fake judge server), including verified hard-fail behavior
when a hard constraint is violated. Not yet run against a real vLLM
instance or real judge API.
