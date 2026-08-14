<<<<<<< HEAD
# Persian Reasoning Benchmark

فریمورک ارزیابی مدل‌های زبانی فاین‌تیون‌شده روی زبان فارسی (بر پایه‌ی
`gpt-oss-20b`)، با تمرکز ویژه روی کیفیت و پایداری زنجیره‌ی استدلال فارسی
(Chain-of-Thought).

## وضعیت فعلی: فاز ۰ (زیرساخت مشترک)

این commit فقط شامل **زیرساخت مشترک** پروژه است که پیش‌نیاز تمام تسک‌های
بعدی (ریاضی، BBH، تقویم جلالی، IFEval، عروض، و ادامه‌ی تسک‌های فاز ۲ تا ۴)
است. هیچ داده‌ی تسک خاصی هنوز اضافه نشده — این کار در commitهای بعدی و
جداگانه انجام می‌شود.

## ساختار پروژه

```
persian-reasoning-benchmark/
├── configs/
│   └── models.yaml          # تعریف مدل‌های کاندید (vLLM) و مدل‌های داور (OpenRouter/GapGPT)
├── rubrics/
│   ├── verifiable_reasoning.json   # rubric کیفیت منطقی زنجیره استدلال (۰-۱۲)
│   └── persian_stability.json      # rubric پایداری زبان فارسی (rule-based + LLM، ۰-۸)
├── src/
│   ├── providers/
│   │   ├── base.py                 # کلاس انتزاعی Provider
│   │   ├── openai_compatible.py    # پیاده‌سازی مشترک برای vLLM/OpenRouter/GapGPT
│   │   └── factory.py              # ساخت provider از روی configs/models.yaml
│   ├── judges/
│   │   ├── rubric_judge.py             # داور rubric استدلال قابل‌راستی‌آزمایی
│   │   └── persian_stability_judge.py  # داور rubric پایداری زبان فارسی
│   ├── utils/
│   │   └── text_utils.py           # توابع rule-based: code-switching، استخراج جواب، نرمال‌سازی
│   └── pipeline/
│       ├── schemas.py              # مدل‌های داده (TaskSample, ModelGeneration, ScoredResult)
│       ├── runner.py               # اجراکننده‌ی عمومی: تولید خروجی مدل + اعمال rubricها
│       └── aggregate.py            # تجمیع نتایج و تولید گزارش خلاصه
├── data/
│   ├── tasks/            # فایل‌های jsonl نمونه‌های هر تسک (در commitهای بعدی اضافه می‌شوند)
│   ├── raw_outputs/      # خروجی خام مدل‌ها (gitignore شده)
│   └── scored_results/   # نتایج امتیازدهی‌شده (gitignore شده)
=======
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
>>>>>>> 4d0469a (feat(phase1): implement Persian math benchmark)
├── tests/
│   └── test_text_utils.py
├── requirements.txt
├── .env.example
└── .gitignore
```

<<<<<<< HEAD
## نصب
=======
## Installation
>>>>>>> 4d0469a (feat(phase1): implement Persian math benchmark)

```bash
pip install -r requirements.txt --break-system-packages
cp .env.example .env
<<<<<<< HEAD
# مقادیر .env را با کلیدهای واقعی خودتان پر کنید
```

## مدل‌ها

- **مدل‌های کاندید (تحت آزمون):** از طریق vLLM سرو می‌شوند (`configs/models.yaml` → `candidate_models`).
- **مدل‌های داور (LLM-judge):** از طریق OpenRouter یا GapGPT فراخوانی می‌شوند
  (`configs/models.yaml` → `judge_models`). هر دو provider سازگار با OpenAI
  Chat Completions API هستند، پس یک کلاس مشترک (`OpenAICompatibleProvider`)
  برای هر سه (vLLM/OpenRouter/GapGPT) کافی است.

## دو Rubric مشترک

### ۱. `verifiable_reasoning` (۰ تا ۱۲)
کیفیت منطقی زنجیره‌ی استدلال را می‌سنجد: آیا هر گام از گام قبل پشتیبانی
می‌شود، آیا پرش منطقی وجود دارد، آیا نتیجه از مقدمات می‌آید. این rubric در
تسک‌های ریاضی، BBH، تناقض/سازگاری، دوگانه‌ی حداقلی، abstention و
برین‌استورم عمیق استفاده می‌شود.

### ۲. `persian_stability` (۰ تا ۸ برای بخش LLM + دو متریک rule-based)
پایداری و کیفیت زبان فارسی در طول زنجیره را می‌سنجد: نرخ code-switching
ناخواسته به انگلیسی (rule-based، رایگان)، نسبت کاراکترهای فارسی (rule-based)،
روانی دستوری، ثبات اصطلاحات، و افت کیفیت بین یک‌سوم ابتدایی و پایانی متن
(هر سه با LLM-judge). این rubric روی همه‌ی تسک‌های دارای CoT قابل اعمال است.

## نحوه‌ی اجرا (بعد از اضافه شدن داده‌ی یک تسک)

```bash
python -m src.pipeline.runner \
  --task-file data/tasks/<task_name>.jsonl \
  --model-name gpt-oss-20b-fa-cot-v1 \
  --output-file data/scored_results/<task_name>__gpt-oss-20b-fa-cot-v1.jsonl
=======
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
>>>>>>> 4d0469a (feat(phase1): implement Persian math benchmark)

python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

<<<<<<< HEAD
## تست
=======
## Tests
>>>>>>> 4d0469a (feat(phase1): implement Persian math benchmark)

```bash
pytest tests/ -v
```

<<<<<<< HEAD
## نقشه‌ی راه (Roadmap)

- [x] فاز ۰: زیرساخت مشترک (این commit)
- [ ] فاز ۱: ریاضی، BBH-lite، تقویم جلالی، IFEval-lite، عروض
- [ ] فاز ۲: MMLU-lite، ابهام‌زدایی خط فارسی، ضرب‌المثل، ایهام و جناس
- [ ] فاز ۳: دوگانه‌ی حداقلی، تناقض و سازگاری، Abstention، پایداری پارافریز، چندقیدی
- [ ] فاز ۴: کنترل‌پذیری فارسی، برین‌استورم عمیق
=======
## Roadmap

- [x] Phase 0: shared infrastructure
- [x] Phase 1 — Math (done)
- [ ] Phase 1 — BBH-lite (logic), Jalali calendar, IFEval-lite, Aroozi (meter)
- [ ] Phase 2 — MMLU-lite, script disambiguation, proverbs, wordplay (ieham)
- [ ] Phase 3 — Minimal pairs, contradiction & consistency, abstention, paraphrase robustness, multi-constraint
- [ ] Phase 4 — Persian controllability, deep brainstorm
>>>>>>> 4d0469a (feat(phase1): implement Persian math benchmark)
