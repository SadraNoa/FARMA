"""
Generates the raw Multi-Constraint source file
(data/raw_sources/multi_constraint_fa_raw.json), fully synthetic — same
approach as scripts/generate_ifeval_lite_data.py (Phase 1), but combining
2-3 rule-checkable constraints per sample instead of 1. This is the fifth
and final planned Phase 3 task.

Reuses the exact same constraint vocabulary as IFEval-lite
(src/utils/instruction_utils.py): sentence_count, word_count,
paragraph_count, must_include, must_include_exact_count, must_exclude,
must_start_with, must_end_with. What's new here is *combining* several of
these simultaneously on one response, which is strictly harder than
IFEval-lite's single-constraint version — a model has to track and
satisfy every constraint at once, not just one.

IMPORTANT — combinability, not just juxtaposition:
Naively picking N random constraints could produce an unsatisfiable
combination (e.g. "must include X exactly twice" + "must exclude X" is a
straight contradiction; two conflicting length constraints like an exact
sentence count together with a high minimum word count can be very hard to
hit at the same time by chance). So constraints are combined in
deliberately-compatible groups:

  - Group L (length): sentence_count / word_count / paragraph_count —
    at most ONE per sample, since stacking two length constraints risks
    accidental self-contradiction.
  - Group K (keyword): must_include / must_include_exact_count /
    must_exclude — up to two per sample, always on two *different* words
    drawn from disjoint word lists (required vs. forbidden), so a
    must_include and a must_exclude can never target the same word.
  - Group P (position): must_start_with / must_end_with — both can be
    combined together as long as the start-word and end-word differ (and
    since every sample also carries a length constraint requiring more
    than one word, a single response word can't simultaneously satisfy
    both anyway).

Four combo categories are generated, rotating over a topic list:
  - length_keyword       (1 from L, 1 from K)
  - length_position      (1 from L, 1 from P — start, end, or both)
  - keyword_position     (1-2 from K, 1 from P)
  - triple_combo         (1 from L, 1 from K, 1 from P) — the hardest tier

Usage:
    python scripts/generate_multi_constraint_data.py
"""

import json
import random

random.seed(11)

TOPICS = [
    "تغییرات آب‌وهوا", "اهمیت خواب کافی", "مزایای مطالعه‌ی کتاب", "تاریخچه‌ی جاده ابریشم",
    "زندگی در شهرهای بزرگ در مقابل روستا", "اهمیت یادگیری زبان دوم", "تأثیر شبکه‌های اجتماعی بر جوانان",
    "فواید ورزش منظم", "آینده‌ی خودروهای برقی", "اهمیت بازیافت زباله", "غذاهای سنتی ایرانی",
    "سفر به اصفهان", "تأثیر هوش مصنوعی بر بازار کار", "اهمیت صرفه‌جویی در مصرف آب",
    "زندگی در دوران باستان ایران", "فواید یوگا و مدیتیشن", "چالش‌های کار از راه دور",
    "تأثیر موسیقی بر روحیه‌ی انسان", "اهمیت آموزش آنلاین", "حفاظت از گونه‌های در حال انقراض",
]
assert len(TOPICS) == 20

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


def to_fa_digits(n) -> str:
    return "".join(FA_DIGITS[int(ch)] if ch.isdigit() else ch for ch in str(n))


# دو فهرست کاملاً مجزا برای must_include/must_include_exact_count در برابر
# must_exclude، تا هیچ نمونه‌ای هرگز یک کلمه‌ی مشترک بین «باید باشد» و
# «نباید باشد» نداشته باشد.
REQUIRED_WORDS = [
    "پرنده", "دریا", "ستاره", "کتاب", "چای", "کوه", "باران", "آینه",
    "چراغ", "باغ", "پل", "ساعت", "کلید", "پنجره", "درخت", "جاده",
    "ابر", "شمع", "قلم", "صندلی",
]

FORBIDDEN_WORDS = [
    "متأسفانه", "احتمالاً", "درواقع", "بنابراین", "همچنین",
    "قطعاً", "خوشبختانه", "اساساً", "طبیعتاً", "عملاً",
]

START_WORDS = [
    "امروز", "همه", "واقعیت", "به‌طورکلی", "نخست", "درباره",
    "زندگی", "همیشه", "معمولاً", "اکنون",
]

END_WORDS = [
    "پایان", "است", "دارد", "می‌شود", "بود", "خواهدشد",
    "می‌رسد", "می‌کند", "ادامه", "تمام",
]


def make_length_constraint(idx: int):
    """یک قید طول (Group L) برمی‌گرداند: (توضیح فارسی، دیکشنری constraint)."""
    mode = idx % 3
    if mode == 0:
        n = random.randint(4, 7)
        return (
            f"پاسخ باید دقیقاً {to_fa_digits(n)} جمله داشته باشد؛ نه بیشتر و نه کمتر.",
            {"type": "sentence_count", "mode": "exact", "value": n},
        )
    elif mode == 1:
        n = random.randint(50, 90)
        return (
            f"پاسخ باید حداقل {to_fa_digits(n)} کلمه داشته باشد.",
            {"type": "word_count", "mode": "at_least", "value": n},
        )
    else:
        n = random.randint(2, 3)
        return (
            f"پاسخ باید دقیقاً در {to_fa_digits(n)} پاراگراف نوشته شود؛ هر پاراگراف را با "
            f"یک خط خالی از پاراگراف بعدی جدا کن.",
            {"type": "paragraph_count", "mode": "exact", "value": n},
        )


def make_keyword_constraint(idx: int, offset: int = 0):
    """یک قید کلیدواژه (Group K) برمی‌گرداند."""
    mode = (idx + offset) % 3
    if mode == 0:
        word = REQUIRED_WORDS[(idx + offset) % len(REQUIRED_WORDS)]
        return (
            f"پاسخ باید حتماً حاوی کلمه‌ی «{word}» باشد.",
            {"type": "must_include", "word": word, "count": 1},
        )
    elif mode == 1:
        word = REQUIRED_WORDS[(idx + offset + 7) % len(REQUIRED_WORDS)]
        n = random.randint(2, 3)
        return (
            f"کلمه‌ی «{word}» باید دقیقاً {to_fa_digits(n)} بار در پاسخ تکرار شود.",
            {"type": "must_include_exact_count", "word": word, "count": n},
        )
    else:
        word = FORBIDDEN_WORDS[(idx + offset) % len(FORBIDDEN_WORDS)]
        return (
            f"در پاسخ خود هرگز از کلمه‌ی «{word}» استفاده نکن.",
            {"type": "must_exclude", "word": word},
        )


def make_position_constraint(idx: int, both: bool = False):
    """یک یا دو قید موقعیت (Group P) برمی‌گرداند: لیستی از (توضیح، constraint)."""
    if both:
        start_word = START_WORDS[idx % len(START_WORDS)]
        end_word = END_WORDS[idx % len(END_WORDS)]
        return [
            (f"پاسخ باید دقیقاً با کلمه‌ی «{start_word}» شروع شود.",
             {"type": "must_start_with", "word": start_word}),
            (f"پاسخ باید دقیقاً با کلمه‌ی «{end_word}» به پایان برسد.",
             {"type": "must_end_with", "word": end_word}),
        ]
    if idx % 2 == 0:
        word = START_WORDS[idx % len(START_WORDS)]
        return [(f"پاسخ باید دقیقاً با کلمه‌ی «{word}» شروع شود.",
                  {"type": "must_start_with", "word": word})]
    else:
        word = END_WORDS[idx % len(END_WORDS)]
        return [(f"پاسخ باید دقیقاً با کلمه‌ی «{word}» به پایان برسد.",
                  {"type": "must_end_with", "word": word})]


def build_sample(sample_id: int, category: str, topic: str, parts: list[tuple[str, dict]]):
    descriptions = [p[0] for p in parts]
    constraints = [p[1] for p in parts]
    numbered = "\n".join(f"{to_fa_digits(i + 1)}. {d}" for i, d in enumerate(descriptions))
    question = (
        f"درباره‌ی «{topic}» بنویس. پاسخ تو باید هم‌زمان همه‌ی محدودیت‌های زیر را رعایت کند:\n"
        f"{numbered}"
    )
    return {
        "id": sample_id,
        "category": category,
        "topic": topic,
        "question": question,
        "constraints": constraints,
    }


samples = []
sid = 1

for idx, topic in enumerate(TOPICS):
    # --- length_keyword: 1 length + 1 keyword ---
    parts = [make_length_constraint(idx), make_keyword_constraint(idx)]
    samples.append(build_sample(sid, "length_keyword", topic, parts))
    sid += 1

    # --- length_position: 1 length + 1-2 position ---
    parts = [make_length_constraint(idx + 1)] + make_position_constraint(idx, both=(idx % 2 == 0))
    samples.append(build_sample(sid, "length_position", topic, parts))
    sid += 1

    # --- keyword_position: 1-2 keyword + 1 position ---
    kw_parts = [make_keyword_constraint(idx, offset=1)]
    if idx % 2 == 0:
        kw_parts.append(make_keyword_constraint(idx, offset=13))
    parts = kw_parts + make_position_constraint(idx + 1, both=False)
    samples.append(build_sample(sid, "keyword_position", topic, parts))
    sid += 1

    # --- triple_combo: 1 length + 1 keyword + 1 position (hardest tier) ---
    parts = [
        make_length_constraint(idx + 2),
        make_keyword_constraint(idx, offset=3),
    ] + make_position_constraint(idx + 2, both=False)
    samples.append(build_sample(sid, "triple_combo", topic, parts))
    sid += 1

print(f"{len(samples)} samples generated")
from collections import Counter
print(Counter(s["category"] for s in samples))

with open("data/raw_sources/multi_constraint_fa_raw.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
