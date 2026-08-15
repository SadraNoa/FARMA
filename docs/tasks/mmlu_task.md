# Phase 2 — MMLU-lite (Persian General Knowledge) Task

## ⚠️ Licensing — read before running

This task sources its data from the gated Hugging Face dataset
[`raia-center/khayyam-challenge`](https://huggingface.co/datasets/raia-center/khayyam-challenge)
(a.k.a. **PersianMMLU** / **Khayyam Challenge**), built from Iran's national
university-entrance exams and Kanoon Farhangi Amoozesh materials.

The dataset is distributed under a **CC-ND (No Derivatives)** license. Its
own README states this explicitly:

> "This license prohibits the creation of derivative works, **including the
> development of benchmarking datasets derived from PersianMMLU**." Usage is
> restricted to non-commercial academic research.

Because converting a sample of this data into a FARMA task file is exactly
the kind of derivative benchmarking artifact the license is talking about,
**this task deliberately does not follow the same pattern as the other
Phase 1 tasks**:

- There is **no `data/raw_sources/khayyam_*.json`** and **no
  `data/tasks/mmlu_lite_fa.jsonl`** committed to this repo (unlike Math,
  BBH, Jalali, IFEval, Aroozi).
- `scripts/prepare_mmlu_task.py` writes **only** under `data/local_only/`,
  which is listed in `.gitignore`. Both the raw HF cache and the converted
  task-ready jsonl stay purely local to whoever runs the script.
- Access requires accepting the dataset's license form on Hugging Face and
  providing your own `HF_TOKEN` in `.env` — it is not shipped or
  redistributed by this project in any form.
- If you plan to publish results, a paper, or a fork of this repo, you are
  responsible for making sure your use stays within "non-commercial academic
  research" and does **not** involve redistributing the dataset or a derived
  benchmark file. This doc does not constitute legal advice.

## Overview

Each sample is an existing multiple-choice question from the source dataset
(question + a fixed set of options + one correct option), spanning a wide
range of subjects/topics from the Iranian educational curriculum
(comparable in spirit to English MMLU). The model must pick the correct
option letter. Because the correct answer is already a single closed-set
option (not something we generate distractors for, unlike Aroozi), this
reuses the framework's generic exact-match `gold_answer` path — no runner
or judge-model changes needed.

## Data

- **Source:** `raia-center/khayyam-challenge` on Hugging Face (gated,
  CC-ND). See the licensing section above.
- **Sample size:** chosen by the user at conversion time, per topic — see
  "Running this task" below. This is intentionally **not** a fixed number
  baked into the repo, since no sampled data is committed.
- The dataset schema (exact column names for topic/question/choices/answer/
  difficulty/educational stage) could not be verified while writing this
  doc, since the dataset is access-gated. Run `--inspect` first (see below)
  to confirm the real column names before converting, and pass overrides
  via `--topic-col` / `--question-col` / `--choices-col` / `--answer-col` /
  etc. if they differ from the script's defaults.

### Task-ready format (`data/local_only/mmlu_lite_fa.jsonl`, gitignored)

Produced by `scripts/prepare_mmlu_task.py`. Illustrative shape (not real
dataset content):

```json
{
  "sample_id": "mmlu_lite_زیست‌شناسی_0003",
  "task_type": "mmlu_lite",
  "problem_fa": "<question text>\n\nالف) ...\nب) ...\nج) ...\nد) ...",
  "gold_answer": "ج",
  "requires_cot_judging": false,
  "apply_verifiable_reasoning_rubric": false,
  "apply_persian_stability_rubric": true,
  "system_prompt_fa": "...",
  "answer_extraction_regex": "پاسخ نهایی:\\s*(.+)",
  "extra": {
    "topic": "زیست‌شناسی",
    "options": {"الف": "...", "ب": "...", "ج": "...", "د": "..."},
    "correct_option": "ج",
    "subject": "...",
    "difficulty": "...",
    "educational_stage": "...",
    "source_dataset": "raia-center/khayyam-challenge",
    "raw_id": 1234
  }
}
```

`answer` in the raw dataset may be encoded as a 0/1-based index, a Latin
letter, or the correct option's own text — `resolve_gold_letter()` in the
script handles all three; run with `--inspect` if conversion errors suggest
a format it doesn't recognize.

## Sampling: choosing how many questions per topic

Per the task requirement, the number of samples taken from each topic is
chosen **before evaluation**, not hard-coded:

- `--samples-per-topic N` — same N for every topic, non-interactive.
- `--topic-samples-json '{"ریاضی": 15, "زیست‌شناسی": 5}'` — per-topic
  overrides (topics not listed fall back to `--samples-per-topic` if given).
- If neither is passed, the script prompts interactively for a single N
  after showing how many topics were found.

Sampling is deterministic per topic (`random.Random(f"{seed}-{topic}")`,
default `seed=42`), so re-running with the same seed and counts reproduces
the same sample set. If a topic has fewer available questions than
requested, all of them are taken (no error).

## Scoring methodology

- **`correctness` (0/1):** computed automatically by the shared runner via
  exact-match between `extracted_final_answer` and `gold_answer` (the
  correct option letter), same mechanism as BBH-lite / Jalali / Aroozi.
- **`persian_stability`** is applied to the raw output.
- **`verifiable_reasoning`** is **not** applied — a hard gold answer already
  exists, same reasoning as the other exact-match tasks.

## Running this task

```bash
# 0. Fill in HF_TOKEN in .env (must have accepted the dataset's license
#    form on its Hugging Face page first).

# 1. Confirm the real column names/splits (dataset is gated, schema unverified here):
python scripts/prepare_mmlu_task.py --inspect

# 2. See topics and how many questions each has:
python scripts/prepare_mmlu_task.py --list-topics

# 3. Convert — interactively, or with an explicit count:
python scripts/prepare_mmlu_task.py --samples-per-topic 10 --seed 42
# (writes to data/local_only/mmlu_lite_fa.jsonl by default — gitignored)

# 4. Run the candidate model (already served via vLLM, per configs/models.yaml):
python -m src.pipeline.runner \
  --task-file data/local_only/mmlu_lite_fa.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/mmlu_lite_fa__gpt-oss-20b-fa-cot-v1.jsonl

# 5. Aggregate as usual:
python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

`--output` and `--raw-cache` are hard-restricted to paths under
`data/local_only/` — the script refuses to write anywhere else, as a guard
against accidentally producing a file that could later get committed.

## Known limitations

- Because the dataset's exact schema wasn't accessible while building this
  task (gated access), `scripts/prepare_mmlu_task.py` ships with best-guess
  default column names (`topic`, `question`, `choices`, `answer`, ...) and
  relies on `--inspect` plus CLI overrides to adapt to the real schema —
  expect to need at least one `--inspect` run before the first real
  conversion.
- Same option-letter format-sensitivity caveat as Aroozi/BBH/Jalali: a model
  that writes `الف)` or repeats full option text instead of the bare letter
  will fail exact-match despite picking the right option.
- No data or sample counts from this task are reproducible from the repo
  alone (by design, per the license) — anyone re-running this task needs
  their own HF access and to choose their own per-topic sample counts.
