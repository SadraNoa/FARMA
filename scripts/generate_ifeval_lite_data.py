import json
import random

random.seed(7)

TOPICS = [
    "تغییرات آب‌وهوا", "اهمیت خواب کافی", "مزایای مطالعه‌ی کتاب", "تاریخچه‌ی جاده ابریشم",
    "زندگی در شهرهای بزرگ در مقابل روستا", "اهمیت یادگیری زبان دوم", "تأثیر شبکه‌های اجتماعی بر جوانان",
    "فواید ورزش منظم", "آینده‌ی خودروهای برقی", "اهمیت بازیافت زباله", "غذاهای سنتی ایرانی",
    "سفر به اصفهان", "تأثیر هوش مصنوعی بر بازار کار", "اهمیت صرفه‌جویی در مصرف آب",
    "زندگی در دوران باستان ایران", "فواید یوگا و مدیتیشن", "چالش‌های کار از راه دور",
    "تأثیر موسیقی بر روحیه‌ی انسان", "اهمیت آموزش آنلاین", "حفاظت از گونه‌های در حال انقراض",
    "فرهنگ چای‌خوری در ایران", "تأثیر بازی‌های ویدیویی بر کودکان", "اهمیت کشاورزی پایدار",
    "تاریخچه‌ی نوروز", "مزایا و معایب شبکه‌های اجتماعی", "اهمیت بهداشت روان",
    "آینده‌ی انرژی‌های تجدیدپذیر", "سفر به شمال ایران", "تأثیر گردشگری بر اقتصاد محلی",
    "اهمیت خواندن روزنامه و اخبار", "زندگی در فضا", "فواید باغبانی خانگی",
    "تأثیر اینترنت اشیا بر زندگی روزمره", "اهمیت حفظ زبان و ادبیات فارسی",
    "چالش‌های کاهش آلودگی هوا", "تأثیر فناوری بر آموزش", "اهمیت کار تیمی",
    "تاریخچه‌ی فرش ایرانی", "مزایای دورکاری برای خانواده‌ها", "آینده‌ی شهرهای هوشمند",
]
assert len(TOPICS) == 40

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


def to_fa_digits(n) -> str:
    return "".join(FA_DIGITS[int(ch)] if ch.isdigit() else ch for ch in str(n))


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

samples = []
sample_idx = 0

for t_idx, topic in enumerate(TOPICS):
    # --- length constraint ---
    length_mode = t_idx % 3
    if length_mode == 0:
        n = random.randint(3, 6)
        q = (f"درباره‌ی «{topic}» بنویس. پاسخ باید دقیقاً {to_fa_digits(n)} جمله داشته باشد؛ "
             f"نه بیشتر و نه کمتر.")
        constraint = {"type": "sentence_count", "mode": "exact", "value": n}
    elif length_mode == 1:
        n = random.randint(40, 90)
        q = (f"درباره‌ی «{topic}» بنویس. پاسخ باید حداقل {to_fa_digits(n)} کلمه داشته باشد.")
        constraint = {"type": "word_count", "mode": "at_least", "value": n}
    else:
        n = random.randint(2, 4)
        q = (f"درباره‌ی «{topic}» بنویس. پاسخ باید دقیقاً در {to_fa_digits(n)} پاراگراف نوشته شود؛ "
             f"هر پاراگراف را با یک خط خالی از پاراگراف بعدی جدا کن.")
        constraint = {"type": "paragraph_count", "mode": "exact", "value": n}
    samples.append({
        "category": "length_constraint",
        "topic": topic,
        "question": q,
        "constraint": constraint,
    })
    sample_idx += 1

    # --- keyword constraint ---
    kw_mode = t_idx % 3
    if kw_mode == 0:
        word = REQUIRED_WORDS[t_idx % len(REQUIRED_WORDS)]
        q = f"درباره‌ی «{topic}» بنویس. پاسخ باید حتماً حاوی کلمه‌ی «{word}» باشد."
        constraint = {"type": "must_include", "word": word, "count": 1}
    elif kw_mode == 1:
        word = REQUIRED_WORDS[(t_idx + 5) % len(REQUIRED_WORDS)]
        n = random.randint(2, 3)
        q = (f"درباره‌ی «{topic}» بنویس. کلمه‌ی «{word}» باید دقیقاً {to_fa_digits(n)} بار "
             f"در پاسخ تکرار شود.")
        constraint = {"type": "must_include_exact_count", "word": word, "count": n}
    else:
        word = FORBIDDEN_WORDS[t_idx % len(FORBIDDEN_WORDS)]
        q = f"درباره‌ی «{topic}» بنویس. در پاسخ خود هرگز از کلمه‌ی «{word}» استفاده نکن."
        constraint = {"type": "must_exclude", "word": word}
    samples.append({
        "category": "keyword_constraint",
        "topic": topic,
        "question": q,
        "constraint": constraint,
    })
    sample_idx += 1

    # --- position constraint ---
    pos_mode = t_idx % 2
    if pos_mode == 0:
        word = START_WORDS[t_idx % len(START_WORDS)]
        q = f"درباره‌ی «{topic}» بنویس. پاسخ باید دقیقاً با کلمه‌ی «{word}» شروع شود."
        constraint = {"type": "must_start_with", "word": word}
    else:
        word = END_WORDS[t_idx % len(END_WORDS)]
        q = f"درباره‌ی «{topic}» بنویس. پاسخ باید دقیقاً با کلمه‌ی «{word}» به پایان برسد."
        constraint = {"type": "must_end_with", "word": word}
    samples.append({
        "category": "position_constraint",
        "topic": topic,
        "question": q,
        "constraint": constraint,
    })
    sample_idx += 1

print(len(samples))
from collections import Counter
print(Counter(s["category"] for s in samples))

with open("/home/claude/ifeval_gen/ifeval_lite_fa_raw.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
