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

### MMLU-lite task (Phase 2, in progress)
- Sourced from the gated HF dataset `raia-center/khayyam-challenge`
  (PersianMMLU). **This dataset's CC-ND license explicitly forbids building
  a derivative benchmarking dataset from it**, so unlike every task above,
  no raw or task-ready data for this task is committed to the repo. Data
  stays entirely local under `data/local_only/` (gitignored); running it
  requires your own `HF_TOKEN` after accepting the dataset's license form.
- `scripts/prepare_mmlu_task.py` lets you choose, per topic, how many
  samples to draw (`--samples-per-topic`, `--topic-samples-json`, or
  interactively) before conversion. Each question becomes a multiple-choice
  `TaskSample` with `gold_answer` = the correct option letter, reusing the
  same exact-match scoring path as BBH-lite/Jalali/Aroozi — no runner
  changes needed.

See `docs/tasks/mmlu_task.md` for full details, including the licensing
constraints and required setup.

### Script Disambiguation task (Phase 2)
- 50 hand-authored samples covering 25 distinct Persian homographs (words
  written identically without diacritics but with more than one valid
  reading/meaning, e.g. «کرم» = کِرم/کَرَم), 2 samples per word (except
  «ملک» with 3 samples for its 3 senses).
- Each sample is a sentence containing the ambiguous word; the model must
  pick the correct (reading, meaning) pair for that occurrence out of a
  small set of multiple-choice options. `gold_answer` is the correct
  option's letter, so `correctness` is computed automatically by the
  runner via exact-match — no runner changes needed.
- Unlike Aroozi/MMLU-lite, distractors are **same-word-only**: a
  dataset-wide pool would offer nonsensical cross-word options (e.g. a
  «دور» reading as a wrong answer for «سحر»). `scripts/prepare_script_disambiguation_task.py`
  builds a per-word pool instead, so option count is **variable per
  sample** (2 options for most words, 3 for «ملک»).
- `persian_stability` is applied; `verifiable_reasoning` is not.

See `docs/tasks/script_disambiguation_task.md` for full details.

### Proverbs task (Phase 2)
- 49 hand-authored samples (`meaning_identification`), each a Persian
  proverb with a 4-option MCQ over its correct meaning/usage.
- Distractors mix hand-authored `manual_distractors` (2/sample) with a
  dataset-wide pool of other samples' `gold_meaning` (unlike Script
  Disambiguation, cross-proverb pool fallback is safe here since a
  proverb's meaning is self-contained). `correctness` is computed
  automatically by the runner via exact-match on the option letter — no
  runner changes needed.
- `persian_stability` is applied; `verifiable_reasoning` is not.

See `docs/tasks/proverbs_task.md` for full details.

### Ieham / Wordplay task (Phase 2)
- 34 hand-authored samples (`double_meaning_identification`), each a
  classical Persian verse with an MCQ over the correct (near-meaning,
  far-meaning) pair for a specified ambiguous word.
- Like Script Disambiguation, distractors are **same-word-only** (a
  per-word pool keyed on the far-meaning, not the full pair, to avoid
  paraphrase-variance duplicates) — a dataset-wide pool would offer
  trivially-unrelated meaning pairs. Option count is variable (mostly
  3-option; up to 4-option for the 3 repeated words). `correctness` is
  computed automatically by the runner via exact-match on the option
  letter — no runner changes needed.
- `persian_stability` is applied; `verifiable_reasoning` is not.

See `docs/tasks/ieham_task.md` for full details.

### Minimal pairs task (Phase 3)
- 64 hand-authored Persian sentence pairs across 8 grammatical phenomena
  (subject-verb agreement, verb-final word order, the «را» definite-object
  marker, negation-prefix placement, tense/adverb agreement, plural
  double-marking, the comparative suffix «تر», and locative preposition
  compatibility) — one sentence per pair is grammatical, the other a
  minimally-edited ungrammatical variant.
- `scripts/prepare_minimal_pairs_task.py` deterministically randomizes
  which sentence is labeled الف vs. ب per sample, then produces a 2-option
  `TaskSample` with `gold_answer` = the grammatical sentence's letter —
  reusing the same exact-match scoring path as every prior closed-set task.
  No distractor pool needed (unlike Aroozi/MMLU-lite), since the pair
  itself is already a fixed 2-way choice.

See `docs/tasks/minimal_pairs_task.md` for full details.

### Contradiction & consistency task (Phase 3)
- Standard 3-way NLI (entailment / contradiction / neutral), built from 20
  hand-authored premises, each paired with one entailed, one contradicting,
  and one neutral hypothesis (60 samples, perfectly balanced 20/20/20).
- `scripts/prepare_contradiction_consistency_task.py` presents the premise,
  hypothesis, and all three relation labels (order deterministically
  shuffled per sample) as a `TaskSample` with `gold_answer` = the correct
  label's letter — same exact-match scoring path as every prior closed-set
  task; no distractor pool needed since the label set is fixed at 3.

See `docs/tasks/contradiction_consistency_task.md` for full details.

### Abstention task (Phase 3)
- Tests whether a model recognizes the limits of what it can answer,
  built from 18 hand-authored passages, each paired with an `answerable`,
  an `insufficient_info`, and a `false_premise` question (54 samples,
  perfectly balanced 18/18/18).
- `scripts/prepare_abstention_task.py` shows every question — regardless
  of subtype — the same four options (real answer, plausible-wrong foil,
  a fixed "insufficient information" statement, a fixed "false premise"
  statement), order deterministically shuffled per sample. Which option is
  correct depends only on the subtype, so a model can't succeed by always
  answering or always hedging. Produces a `TaskSample` with `gold_answer` =
  the correct option's letter — same exact-match scoring path as every
  prior closed-set task.

See `docs/tasks/abstention_task.md` for full details.

### Paraphrase robustness task (Phase 3)
- Tests whether a model's answer stays correct and consistent when the same
  underlying question is reworded, built from 15 hand-authored
  `(passage, correct_answer, 3 foils)` families, each expressed as 4
  differently-worded questions asking exactly the same thing (60 samples).
- `scripts/prepare_paraphrase_robustness_task.py` keeps the passage, correct
  answer, and foils identical across all 4 paraphrases of a family — only
  the question wording changes — while still reshuffling option display
  order per sample (keyed off the individual sample, not the shared family)
  so a model can't look robust by memorizing a letter position instead of
  re-reading each paraphrase. Produces a `TaskSample` with `gold_answer` =
  the correct option's letter — same exact-match scoring path as every
  prior closed-set task. Family-level robustness (do all 4 paraphrases
  agree?) is left as an aggregation-time computation over `extra.family_id`.

See `docs/tasks/paraphrase_robustness_task.md` for full details.

### Multi-Constraint task (Phase 3)
- 80 fully-synthetic samples (`scripts/generate_multi_constraint_data.py`,
  same generation style as Phase 1's IFEval-lite), combining 2-3
  simultaneous rule-checkable constraints per sample across 4 categories
  (`length_keyword`, `length_position`, `keyword_position`, `triple_combo`)
  — strictly harder than IFEval-lite's single-constraint version.
- Reuses IFEval-lite's exact constraint vocabulary and rule-based checkers
  (`src/utils/instruction_utils.py`) — no new checker logic. Constraints
  are combined only in verified-compatible groups (at most one length
  constraint per sample; keyword constraints never target overlapping
  include/exclude words; start/end words always differ), confirmed by an
  automated all-80-samples check plus a hand-verified feasibility test.
- New `extra.rule_constraints` key (list of constraint dicts) and one
  additive `elif` branch in `runner.py`: `correctness = 1` only if **all**
  constraints pass. Deliberately independent from Persian Controllability's
  similarly-named `extra.constraints`, which is judge-based and would be
  the wrong path for this fully rule-based task. No schema changes (reuses
  the existing generic `notes` field for per-constraint pass/fail detail).

See `docs/tasks/multi_constraint_task.md` for full details.

This completes all 5 planned Phase 3 tasks.

### Deep Brainstorm task (Phase 4)
- 4 hand-authored open-ended Persian prompts across 4 categories
  (`open_problem_solving`, `creative_ideation`, `policy_or_social`,
  `product_or_design`).
- No `gold_answer` — scored entirely through three stacked LLM-judge
  rubrics: the existing `verifiable_reasoning` (already listed
  `deep_brainstorm` in its `applies_to`), the existing `persian_stability`,
  and a new `deep_brainstorm` rubric (`rubrics/deep_brainstorm.json`)
  covering idea diversity, self-correction, and cultural grounding —
  dimensions the two generic rubrics don't reach.
- `scripts/prepare_deep_brainstorm_task.py` converts the raw prompts into
  the executable `data/tasks/deep_brainstorm_fa.jsonl` format.

See `docs/tasks/deep_brainstorm_task.md` for full details.

### Persian Controllability task (Phase 4)
- 7 hand-authored samples testing whether the model can satisfy multiple
  Persian-specific stylistic constraints simultaneously (register,
  pure-Persian lexicon, sentence structure, paragraph count, dialect),
  including hard-constraint and multi-constraint-combined cases.
- Each sample carries a *list* of constraints (`extra.constraints`), split
  between rule-checked types (`paragraph_count`, `sentence_structure` —
  `src/utils/controllability_utils.py`) and judge-checked types (`lexical`,
  `register`, `dialect` — `rubrics/persian_controllability.json` +
  `src/judges/persian_controllability_judge.py`), plus an independent
  semantic-fidelity score. Violating any `hard: true` constraint forces
  `controllability_hard_fail = true` for that sample.
- `scripts/prepare_persian_controllability_task.py` converts the raw
  samples into `data/tasks/persian_controllability_fa.jsonl`.

See `docs/tasks/persian_controllability_task.md` for full details.

## Project structure

```
FARMA/
├── configs/
│   └── models.yaml          # Candidate models (vLLM) and judge models (OpenRouter/GapGPT)
├── rubrics/
│   ├── verifiable_reasoning.json   # Logical quality of reasoning chain (0-12)
│   ├── persian_stability.json      # Persian language stability (rule-based + LLM, 0-8)
│   ├── deep_brainstorm.json        # Idea diversity, self-correction, cultural grounding (0-8)
│   └── persian_controllability.json # Judge-checkable constraints (lexical/register/dialect) + semantic fidelity
├── src/
│   ├── providers/
│   │   ├── base.py                 # Abstract Provider class
│   │   ├── openai_compatible.py    # Shared implementation for vLLM/OpenRouter/GapGPT
│   │   └── factory.py              # Builds a provider from configs/models.yaml
│   ├── judges/
│   │   ├── rubric_judge.py             # Judge for the verifiable_reasoning rubric
│   │   ├── persian_stability_judge.py  # Judge for the persian_stability rubric
│   │   ├── deep_brainstorm_judge.py    # Judge for the deep_brainstorm rubric
│   │   └── persian_controllability_judge.py  # Dynamic per-sample judge for controllability constraints
│   ├── utils/
│   │   ├── text_utils.py           # Rule-based helpers: code-switching, answer extraction, normalization
│   │   ├── instruction_utils.py    # Rule-based IFEval-lite constraint checkers
│   │   └── controllability_utils.py # Rule-based checkers for Persian Controllability (paragraph_count, sentence_structure)
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
│   ├── prepare_aroozi_task.py              # Converts raw Aroozi data into task-ready multiple-choice jsonl
│   ├── prepare_mmlu_task.py                # Fetches/samples Khayyam Challenge (PersianMMLU), local-only (see docs/tasks/mmlu_task.md)
│   ├── prepare_deep_brainstorm_task.py     # Converts raw Deep Brainstorm prompts into task-ready jsonl
│   └── prepare_persian_controllability_task.py  # Converts raw Persian Controllability samples into task-ready jsonl
├── data/
│   ├── raw_sources/      # Raw, unprocessed source data per task (committed, except MMLU-lite)
│   ├── tasks/            # Task-ready jsonl files (input to the runner; committed, except MMLU-lite)
│   ├── raw_outputs/      # Raw model outputs (gitignored)
│   ├── scored_results/   # Scored results (gitignored)
│   └── local_only/       # MMLU-lite raw cache + task jsonl only — gitignored, never committed (CC-ND license)
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
- [x] Phase 2 — MMLU-lite (done; data stays local-only, see `docs/tasks/mmlu_task.md`)
- [x] Phase 2 — Script disambiguation (done; see `docs/tasks/script_disambiguation_task.md`)
- [x] Phase 2 — Proverbs (done; see `docs/tasks/proverbs_task.md`)
- [x] Phase 2 — Wordplay / ieham (done; see `docs/tasks/ieham_task.md`)
- [x] Phase 3 — Minimal pairs (done; see `docs/tasks/minimal_pairs_task.md`)
- [x] Phase 3 — Contradiction & consistency (done; see `docs/tasks/contradiction_consistency_task.md`)
- [x] Phase 3 — Abstention (done; see `docs/tasks/abstention_task.md`)
- [x] Phase 3 — Paraphrase robustness (done; see `docs/tasks/paraphrase_robustness_task.md`)
- [x] Phase 3 — Multi-constraint (done; see `docs/tasks/multi_constraint_task.md`) — **Phase 3 complete**
- [x] Phase 4 — Persian controllability (done; see `docs/tasks/persian_controllability_task.md`)
- [x] Phase 4 — Deep brainstorm (done; see `docs/tasks/deep_brainstorm_task.md`)
