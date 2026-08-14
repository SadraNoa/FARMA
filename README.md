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
├── tests/
│   └── test_text_utils.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## نصب

```bash
pip install -r requirements.txt --break-system-packages
cp .env.example .env
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

python -m src.pipeline.aggregate \
  --results-dir data/scored_results \
  --out report.json
```

## تست

```bash
pytest tests/ -v
```

## نقشه‌ی راه (Roadmap)

- [x] فاز ۰: زیرساخت مشترک (این commit)
- [ ] فاز ۱: ریاضی، BBH-lite، تقویم جلالی، IFEval-lite، عروض
- [ ] فاز ۲: MMLU-lite، ابهام‌زدایی خط فارسی، ضرب‌المثل، ایهام و جناس
- [ ] فاز ۳: دوگانه‌ی حداقلی، تناقض و سازگاری، Abstention، پایداری پارافریز، چندقیدی
- [ ] فاز ۴: کنترل‌پذیری فارسی، برین‌استورم عمیق
