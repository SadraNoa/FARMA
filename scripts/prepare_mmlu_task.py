"""
Phase 2 — MMLU-lite (Persian) task, sourced from the gated HF dataset
`raia-center/khayyam-challenge` (a.k.a. PersianMMLU / Khayyam Challenge).

*** هشدار لایسنس — قبل از استفاده حتماً بخوانید ***
این دیتاست تحت لایسنس CC-ND منتشر شده و به‌صراحت "ساخت هر بنچمارک
مشتق‌شده از PersianMMLU" را ممنوع کرده است (فقط برای پژوهش آکادمیک
غیرتجاری، بدون امکان توزیع/کامیت مجدد داده). به همین دلیل، برخلاف بقیه‌ی
تسک‌های این پروژه (Math, BBH, Jalali, IFEval, Aroozi)، این اسکریپت:
  - هیچ فایل raw_source یا task-ready‌ای در data/raw_sources/ یا
    data/tasks/ (که در ریپو کامیت می‌شوند) نمی‌نویسد.
  - همه‌چیز — کش خام دیتاست و jsonl نهایی — را فقط زیر data/local_only/
    می‌نویسد که در .gitignore قرار دارد و هرگز نباید کامیت شود.
  - نیازمند HF_TOKEN در .env است (باید فرم رضایت لایسنس را در صفحه‌ی
    دیتاست هاگینگ‌فیس امضا کرده باشید).
اگر قصد انتشار/عمومی‌کردن این ریپو یا نتایج را دارید، مسئولیت پایبندی به
شرایط لایسنس (پژوهش غیرتجاری، عدم توزیع مجدد داده) با شماست.

این اسکریپت هر رکورد چندگزینه‌ای خام را به فرمت مشترک TaskSample تبدیل
می‌کند (مثل عروض: gold_answer یک حرف گزینه است، پس correctness توسط
runner عمومی و از طریق exact-match محاسبه می‌شود — نیازی به تغییر
runner.py یا داور LLM نیست).

نکته‌ی مهم درباره‌ی نمونه‌گیری بر اساس "مبحث" (topic): کاربر پیش از
ارزیابی تعیین می‌کند از هر مبحث چند نمونه برداشته شود؛ این کار یا با
--samples-per-topic (اعمال یکسان روی همه‌ی مباحث) یا با
--topic-samples-json (override دلخواه به ازای هر مبحث) یا در حالت
تعاملی (بدون پاس‌دادن هیچ‌کدام) انجام می‌شود.

Usage (اول همیشه اسکیمای واقعی دیتاست را با --inspect ببینید، چون
دیتاست گیت‌شده است و ستون‌ها را از قبل نمی‌دانیم):

    # ۱. دیدن ستون‌ها/configها/splitهای واقعی دیتاست
    python scripts/prepare_mmlu_task.py --inspect

    # ۲. دیدن لیست مباحث و تعداد نمونه‌ی هرکدام
    python scripts/prepare_mmlu_task.py --list-topics

    # ۳. تبدیل تعاملی (از کاربر تعداد نمونه به ازای هر مبحث پرسیده می‌شود)
    python scripts/prepare_mmlu_task.py

    # ۴. تبدیل غیرتعاملی با تعداد یکسان برای همه‌ی مباحث
    python scripts/prepare_mmlu_task.py --samples-per-topic 10 --seed 42

    # ۵. تبدیل با override دلخواه به ازای هر مبحث
    python scripts/prepare_mmlu_task.py --topic-samples-json '{"ریاضی": 15, "زیست‌شناسی": 5}'
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Optional

DEFAULT_DATASET = "raia-center/khayyam-challenge"
DEFAULT_OUTPUT = "data/local_only/mmlu_lite_fa.jsonl"
DEFAULT_RAW_CACHE = "data/local_only/khayyam_challenge_raw_cache.json"

LICENSE_NOTICE_FA = (
    "این دیتاست (raia-center/khayyam-challenge) تحت لایسنس CC-ND است: فقط "
    "برای پژوهش آکادمیک غیرتجاری و بدون ساخت/توزیع بنچمارک مشتق‌شده. این "
    "فایل و کش خام مرتبط با آن باید همیشه در data/local_only/ (gitignored) "
    "بمانند و هرگز نباید کامیت یا منتشر شوند."
)

# ترتیب حروف گزینه‌ها؛ حداکثر ۶ گزینه پشتیبانی می‌شود (هم‌راستا با aroozi)
OPTION_LETTERS = ["الف", "ب", "ج", "د", "ه", "و"]

SYSTEM_PROMPT_FA = (
    "تو یک دستیار هوشمند فارسی‌زبان هستی که در آزمون‌های چندگزینه‌ای عمومی "
    "(علوم، ادبیات، ریاضی، دین و...) شرکت می‌کنی. سوال زیر را با دقت بخوان و "
    "در صورت نیاز مختصر استدلال کن. در پایان پاسخ خود، دقیقاً یک خط جداگانه "
    "با این قالب بنویس (فقط حرفِ گزینه، بدون هیچ توضیح اضافه):\n"
    "پاسخ نهایی: <حرف گزینه، مثلاً الف>"
)

ANSWER_EXTRACTION_REGEX = r"پاسخ نهایی:\s*(.+)"


# --------------------------------------------------------------------------
# توابع خالص (بدون وابستگی به شبکه/HF) — قابل تست مستقیم با رکوردهای ساختگی
# --------------------------------------------------------------------------

def slugify_topic(topic: str) -> str:
    """مبحث را به یک شناسه‌ی ساده و امن برای sample_id تبدیل می‌کند."""
    slug = re.sub(r"\s+", "_", topic.strip())
    slug = re.sub(r"[^\w\u0600-\u06FF_]", "", slug)
    return slug or "topic"


def group_by_topic(records: list[dict], topic_col: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for rec in records:
        topic = rec.get(topic_col)
        if topic is None:
            continue
        groups.setdefault(str(topic), []).append(rec)
    return groups


def sample_group(records: list[dict], n_requested: int, seed_key: str) -> list[dict]:
    """نمونه‌گیری قطعی (deterministic) از یک مبحث: اگر n_requested بیشتر از
    تعداد موجود باشد، همه‌ی رکوردهای موجود برگردانده می‌شوند."""
    if n_requested <= 0:
        return []
    rng = random.Random(seed_key)
    pool = list(records)
    rng.shuffle(pool)
    return pool[:n_requested]


def extract_choices(record: dict, choices_col: Optional[str], choices_cols: Optional[list[str]]) -> list[str]:
    """گزینه‌ها را یا از یک ستون لیستی (choices_col) یا از چند ستون مجزا
    (choices_cols، مثلاً option_1..option_4) استخراج می‌کند."""
    if choices_col is not None:
        raw = record.get(choices_col)
        if isinstance(raw, (list, tuple)):
            return [str(c) for c in raw]
        if isinstance(raw, str):
            # برخی دیتاست‌ها choices را به‌صورت رشته‌ی JSON ذخیره می‌کنند
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, (list, tuple)):
                    return [str(c) for c in parsed]
            except json.JSONDecodeError:
                pass
        raise ValueError(f"ستون گزینه‌ها ('{choices_col}') لیست نیست: {raw!r}")
    if choices_cols:
        return [str(record[c]) for c in choices_cols if record.get(c) not in (None, "")]
    raise ValueError("باید یکی از --choices-col یا --choices-cols را مشخص کنید.")


def resolve_gold_letter(choices: list[str], answer: Any, option_letters: list[str]) -> str:
    """مقدار ستون answer را (که ممکن است اندیس صفر/یک‌مبنا، حرف، یا خودِ متن
    گزینه‌ی درست باشد) به حرف گزینه (الف/ب/ج/د...) نگاشت می‌کند."""
    n = len(choices)
    if n > len(option_letters):
        raise ValueError(f"تعداد گزینه‌ها ({n}) از حروف پشتیبانی‌شده ({len(option_letters)}) بیشتر است.")

    # حالت ۱: اندیس عددی (0-based یا 1-based)
    if isinstance(answer, int):
        if 0 <= answer < n:
            return option_letters[answer]
        if 1 <= answer <= n:
            return option_letters[answer - 1]
        raise ValueError(f"اندیس answer={answer} خارج از بازه‌ی گزینه‌ها ({n} گزینه) است.")

    if isinstance(answer, str):
        s = answer.strip()

        # حالت ۲: رشته‌ی عددی
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < n:
                return option_letters[idx]
            if 1 <= idx <= n:
                return option_letters[idx - 1]

        # حالت ۳: حرف لاتین A/B/C/D یا a/b/c/d
        if len(s) == 1 and s.upper() in "ABCDEF":
            idx = ord(s.upper()) - ord("A")
            if 0 <= idx < n:
                return option_letters[idx]

        # حالت ۴: از قبل حرف فارسی معتبر است (الف/ب/ج/د/ه/و)
        if s in option_letters:
            return s

        # حالت ۵: متنِ خودِ گزینه‌ی درست را می‌دهد (تطبیق دقیق روی متن گزینه‌ها)
        for letter, choice_text in zip(option_letters, choices):
            if choice_text.strip() == s:
                return letter

    raise ValueError(f"نتوانستم مقدار answer={answer!r} را به یکی از گزینه‌ها نگاشت کنم.")


def build_task_sample(
    record: dict,
    idx: int,
    topic: str,
    question_col: str,
    choices_col: Optional[str],
    choices_cols: Optional[list[str]],
    answer_col: str,
    subject_col: Optional[str],
    difficulty_col: Optional[str],
    stage_col: Optional[str],
    id_col: Optional[str],
) -> dict:
    question = record.get(question_col)
    if not question:
        raise ValueError(f"ستون سوال ('{question_col}') خالی است.")

    choices = extract_choices(record, choices_col, choices_cols)
    if len(choices) < 2:
        raise ValueError(f"تعداد گزینه‌های معتبر کمتر از ۲ است: {choices!r}")

    letters = OPTION_LETTERS[: len(choices)]
    correct_letter = resolve_gold_letter(choices, record.get(answer_col), letters)

    options_map = {letter: text for letter, text in zip(letters, choices)}
    options_lines = [f"{letter}) {text}" for letter, text in zip(letters, choices)]
    problem_fa = f"{question}\n\n" + "\n".join(options_lines)

    raw_id = record.get(id_col) if id_col else None
    topic_slug = slugify_topic(topic)
    sample_id = f"mmlu_lite_{topic_slug}_{idx:04d}"

    extra: dict[str, Any] = {
        "topic": topic,
        "options": options_map,
        "correct_option": correct_letter,
        "source_dataset": DEFAULT_DATASET,
        "license_notice": LICENSE_NOTICE_FA,
    }
    if subject_col and record.get(subject_col) is not None:
        extra["subject"] = record.get(subject_col)
    if difficulty_col and record.get(difficulty_col) is not None:
        extra["difficulty"] = record.get(difficulty_col)
    if stage_col and record.get(stage_col) is not None:
        extra["educational_stage"] = record.get(stage_col)
    if raw_id is not None:
        extra["raw_id"] = raw_id

    return {
        "sample_id": sample_id,
        "task_type": "mmlu_lite",
        "problem_fa": problem_fa,
        "gold_answer": correct_letter,
        "requires_cot_judging": False,
        "apply_verifiable_reasoning_rubric": False,
        "apply_persian_stability_rubric": True,
        "system_prompt_fa": SYSTEM_PROMPT_FA,
        "answer_extraction_regex": ANSWER_EXTRACTION_REGEX,
        "extra": extra,
    }


def convert_records(
    records: list[dict],
    topic_col: str,
    question_col: str,
    answer_col: str,
    choices_col: Optional[str] = None,
    choices_cols: Optional[list[str]] = None,
    subject_col: Optional[str] = None,
    difficulty_col: Optional[str] = None,
    stage_col: Optional[str] = None,
    id_col: Optional[str] = None,
    samples_per_topic: Optional[int] = None,
    topic_samples_override: Optional[dict[str, int]] = None,
    seed: int = 42,
) -> tuple[list[dict], dict[str, int], int]:
    """هسته‌ی تبدیل: گروه‌بندی بر اساس مبحث + نمونه‌گیری + ساخت TaskSample.
    برمی‌گرداند: (لیست TaskSampleها, شمار انتخاب‌شده به ازای هر مبحث, تعداد رد‌شده)."""
    groups = group_by_topic(records, topic_col)
    topic_samples_override = topic_samples_override or {}

    task_samples: list[dict] = []
    per_topic_selected: dict[str, int] = {}
    skipped = 0

    for topic, topic_records in groups.items():
        n_requested = topic_samples_override.get(topic, samples_per_topic)
        if n_requested is None:
            raise ValueError(
                f"تعداد نمونه برای مبحث '{topic}' مشخص نشده (نه samples_per_topic و نه override)."
            )
        selected = sample_group(topic_records, n_requested, seed_key=f"{seed}-{topic}")
        per_topic_selected[topic] = 0
        for i, rec in enumerate(selected):
            try:
                task_sample = build_task_sample(
                    rec, i, topic,
                    question_col=question_col, choices_col=choices_col, choices_cols=choices_cols,
                    answer_col=answer_col, subject_col=subject_col, difficulty_col=difficulty_col,
                    stage_col=stage_col, id_col=id_col,
                )
            except ValueError as e:
                skipped += 1
                print(f"skipped (malformed sample) topic={topic!r} idx={i}: {e}")
                continue
            task_samples.append(task_sample)
            per_topic_selected[topic] += 1

    return task_samples, per_topic_selected, skipped


def write_jsonl(task_samples: list[dict], output_path: str) -> None:
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in task_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# توابع وابسته به شبکه/HF — بارگذاری دیتاست گیت‌شده
# --------------------------------------------------------------------------

def _load_hf_dataset(dataset: str, config: Optional[str], split: Optional[str]):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "پکیج 'datasets' نصب نیست. اجرا کنید: pip install datasets --break-system-packages"
        ) from e

    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None
    if not hf_token:
        print(
            "هشدار: HF_TOKEN در .env تنظیم نشده. چون این دیتاست گیت‌شده است، "
            "بارگذاری احتمالاً با خطای دسترسی شکست می‌خورد مگر قبلاً با "
            "`huggingface-cli login` وارد شده باشید."
        )

    kwargs: dict[str, Any] = {"token": hf_token}
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset, split=split, **kwargs) if split else load_dataset(dataset, **kwargs)
    return ds


def _dataset_to_records(ds) -> list[dict]:
    """هم DatasetDict (چند split) و هم Dataset تکی را به لیست dict تخت تبدیل می‌کند."""
    try:
        from datasets import DatasetDict
    except ImportError:
        DatasetDict = ()  # type: ignore

    records: list[dict] = []
    if isinstance(ds, DatasetDict):
        for split_name, split_ds in ds.items():
            for row in split_ds:
                row = dict(row)
                row.setdefault("_split", split_name)
                records.append(row)
    else:
        for row in ds:
            records.append(dict(row))
    return records


def load_or_fetch_records(
    dataset: str, config: Optional[str], split: Optional[str],
    raw_cache_path: str, refresh: bool = False,
) -> list[dict]:
    """کش محلی (data/local_only/...) را می‌خواند، وگرنه از HF می‌گیرد و کش می‌کند.
    این کش هم فقط لوکال است (gitignored) — نه raw_sources رسمی پروژه."""
    if not refresh and os.path.exists(raw_cache_path):
        with open(raw_cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ds = _load_hf_dataset(dataset, config, split)
    records = _dataset_to_records(ds)

    Path(os.path.dirname(raw_cache_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(raw_cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"({len(records)} رکورد خام کش شد در: {raw_cache_path} — این فایل gitignored است)")
    return records


def cmd_inspect(args):
    from datasets import load_dataset
    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or None

    try:
        builder_configs = load_dataset(args.dataset, token=hf_token)
        print("Splits/columns یافت‌شده:")
        for split_name, split_ds in builder_configs.items():
            print(f"  split='{split_name}'  rows={len(split_ds)}  columns={split_ds.column_names}")
            if len(split_ds) > 0:
                print(f"    نمونه‌ی اول (فقط نام فیلدها): {list(split_ds[0].keys())}")
    except Exception as e:
        print(f"بارگذاری با تنظیمات پیش‌فرض شکست خورد ({e}). اگر دیتاست چند config دارد، "
              f"--config را مشخص کنید یا با huggingface-cli login وارد شوید.")


def cmd_list_topics(args):
    records = load_or_fetch_records(args.dataset, args.config, args.split, args.raw_cache, args.refresh_cache)
    groups = group_by_topic(records, args.topic_col)
    print(f"{len(groups)} مبحث یافت شد (ستون='{args.topic_col}'):")
    for topic, recs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {topic}: {len(recs)}")


def _prompt_samples_per_topic(topics: list[str]) -> int:
    print(f"{len(topics)} مبحث یافت شد. چند نمونه از هر مبحث برداشته شود؟")
    while True:
        raw = input("تعداد نمونه به ازای هر مبحث: ").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("لطفاً یک عدد صحیح مثبت وارد کنید.")


def cmd_convert(args):
    print(LICENSE_NOTICE_FA)
    records = load_or_fetch_records(args.dataset, args.config, args.split, args.raw_cache, args.refresh_cache)

    groups = group_by_topic(records, args.topic_col)
    if not groups:
        raise SystemExit(f"هیچ رکوردی با ستون مبحث '{args.topic_col}' پیدا نشد؛ با --inspect ستون‌های واقعی را ببینید.")

    topic_samples_override = None
    if args.topic_samples_json:
        topic_samples_override = json.loads(args.topic_samples_json)

    samples_per_topic = args.samples_per_topic
    if samples_per_topic is None and not topic_samples_override:
        samples_per_topic = _prompt_samples_per_topic(list(groups.keys()))

    choices_cols = args.choices_cols.split(",") if args.choices_cols else None

    task_samples, per_topic_selected, skipped = convert_records(
        records,
        topic_col=args.topic_col,
        question_col=args.question_col,
        answer_col=args.answer_col,
        choices_col=args.choices_col,
        choices_cols=choices_cols,
        subject_col=args.subject_col,
        difficulty_col=args.difficulty_col,
        stage_col=args.stage_col,
        id_col=args.id_col,
        samples_per_topic=samples_per_topic,
        topic_samples_override=topic_samples_override,
        seed=args.seed,
    )

    write_jsonl(task_samples, args.output)

    print(f"\n{len(task_samples)} نمونه‌ی MMLU-lite ساخته و ذخیره شد در: {args.output} ({skipped} رد شد)")
    print("توزیع نهایی به ازای هر مبحث:")
    for topic, count in sorted(per_topic_selected.items()):
        print(f"  {topic}: {count}")
    print(f"\nیادآوری: {args.output} و {args.raw_cache} هر دو در data/local_only/ و gitignored هستند — "
          "هرگز آن‌ها را کامیت یا منتشر نکنید.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="آماده‌سازی تسک MMLU-lite از دیتاست گیت‌شده‌ی Khayyam Challenge")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=None, help="نام config دیتاست HF (در صورت وجود چند subset)")
    parser.add_argument("--split", default=None, help="نام split (خالی = همه‌ی splitها ترکیب می‌شوند)")
    parser.add_argument("--raw-cache", default=DEFAULT_RAW_CACHE, help="مسیر کش لوکال دیتای خام (gitignored)")
    parser.add_argument("--refresh-cache", action="store_true", help="نادیده‌گرفتن کش و دانلود مجدد از HF")

    parser.add_argument("--inspect", action="store_true", help="فقط splitها/ستون‌های واقعی دیتاست را نشان بده و خارج شو")
    parser.add_argument("--list-topics", action="store_true", help="فقط لیست مباحث و تعداد نمونه‌ی هرکدام را نشان بده")

    parser.add_argument("--topic-col", default="topic", help="نام ستون مبحث")
    parser.add_argument("--question-col", default="question", help="نام ستون متن سوال")
    parser.add_argument("--choices-col", default="choices", help="نام ستونی که گزینه‌ها را به‌صورت لیست نگه می‌دارد")
    parser.add_argument("--choices-cols", default=None, help="در صورت نبودن ستون لیستی: نام ستون‌های مجزا با کاما، مثلاً option_1,option_2,option_3,option_4")
    parser.add_argument("--answer-col", default="answer", help="نام ستون پاسخ درست (اندیس، حرف، یا متن گزینه)")
    parser.add_argument("--subject-col", default="subject", help="نام ستون درس/موضوع کلی (اختیاری، فقط متادیتا)")
    parser.add_argument("--difficulty-col", default="difficulty", help="نام ستون سطح دشواری (اختیاری، فقط متادیتا)")
    parser.add_argument("--stage-col", default="educational_stage", help="نام ستون مقطع تحصیلی (اختیاری، فقط متادیتا)")
    parser.add_argument("--id-col", default="id", help="نام ستون شناسه‌ی خام (اختیاری، فقط متادیتا)")

    parser.add_argument("--samples-per-topic", type=int, default=None, help="تعداد نمونه به ازای هر مبحث؛ اگر ندهید و override هم نباشد، به‌صورت تعاملی پرسیده می‌شود")
    parser.add_argument("--topic-samples-json", default=None, help='override دلخواه به ازای هر مبحث، مثلاً \'{"ریاضی": 15}\'')
    parser.add_argument("--seed", type=int, default=42, help="seed برای نمونه‌گیری قطعی و قابل‌تکرار")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="مسیر خروجی jsonl (باید زیر data/local_only/ بماند)")
    return parser


def main():
    args = build_arg_parser().parse_args()

    for path_arg, flag in [(args.output, "--output"), (args.raw_cache, "--raw-cache")]:
        if not str(path_arg).replace("\\", "/").startswith("data/local_only/"):
            raise SystemExit(
                f"برای رعایت لایسنس CC-ND این دیتاست، {flag} باید زیر data/local_only/ باشد "
                "(همان مسیری که در .gitignore قرار دارد)."
            )

    if args.inspect:
        cmd_inspect(args)
        return
    if args.list_topics:
        cmd_list_topics(args)
        return
    cmd_convert(args)


if __name__ == "__main__":
    main()
