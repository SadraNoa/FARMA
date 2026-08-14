import json
import random
import datetime
import jdatetime

random.seed(42)

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

JALALI_MONTHS = [
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
]

GREGORIAN_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

WEEKDAY_FA = ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه"]


def to_fa_digits(n) -> str:
    s = str(n)
    return "".join(FA_DIGITS[int(ch)] if ch.isdigit() else ch for ch in s)


def jalali_str(d: jdatetime.date) -> str:
    return f"{to_fa_digits(d.day)} {JALALI_MONTHS[d.month - 1]} {to_fa_digits(d.year)}"


def gregorian_str(d: datetime.date) -> str:
    return f"{d.day} {GREGORIAN_MONTHS[d.month - 1]} {d.year}"


def random_jalali_date(year_lo=1375, year_hi=1409):
    year = random.randint(year_lo, year_hi)
    month = random.randint(1, 12)
    if month <= 6:
        day = random.randint(1, 31)
    elif month <= 11:
        day = random.randint(1, 30)
    else:
        day = random.randint(1, 29)  # stay safe re: leap day of esfand
    return jdatetime.date(year, month, day)


samples = []

# 1. conversion: 38 samples, mix jalali->gregorian and gregorian->jalali
conv_count = 38
for i in range(conv_count):
    jd = random_jalali_date()
    if i % 2 == 0:
        gd = jd.togregorian()
        q = f"تاریخ {jalali_str(jd)} (تقویم جلالی) در تقویم میلادی چه روزی است؟"
        a = gregorian_str(gd)
    else:
        gd = jd.togregorian()
        q = f"تاریخ {gregorian_str(gd)} (تقویم میلادی) در تقویم جلالی چه روزی است؟"
        a = jalali_str(jd)
    samples.append({"category": "date_conversion", "question": q, "answer": a})

# 2. leap_year: 38 samples
leap_count = 38
used_years = set()
for i in range(leap_count):
    while True:
        year = random.randint(1370, 1450)
        if year not in used_years:
            used_years.add(year)
            break
    is_leap = jdatetime.date(year, 1, 1).isleap()
    q = f"آیا سال {to_fa_digits(year)} در تقویم جلالی کبیسه است؟"
    a = "بله" if is_leap else "خیر"
    samples.append({"category": "leap_year", "question": q, "answer": a})

# 3. arithmetic: add/subtract days or months, 37 samples
arith_count = 37
for i in range(arith_count):
    jd = random_jalali_date(1380, 1405)
    mode = i % 4
    if mode == 0:
        n = random.randint(1, 90)
        new_d = jd + datetime.timedelta(days=n)
        q = f"اگر {to_fa_digits(n)} روز به تاریخ {jalali_str(jd)} اضافه کنیم، تاریخ جدید (جلالی) چیست؟"
        a = jalali_str(new_d)
    elif mode == 1:
        n = random.randint(1, 90)
        new_d = jd - datetime.timedelta(days=n)
        q = f"اگر {to_fa_digits(n)} روز از تاریخ {jalali_str(jd)} کم کنیم، تاریخ جدید (جلالی) چیست؟"
        a = jalali_str(new_d)
    elif mode == 2:
        n = random.randint(1, 11)
        total_months = (jd.month - 1) + n
        new_year = jd.year + total_months // 12
        new_month = total_months % 12 + 1
        max_day = 31 if new_month <= 6 else (30 if new_month <= 11 else 29)
        new_day = min(jd.day, max_day)
        new_d = jdatetime.date(new_year, new_month, new_day)
        q = f"اگر {to_fa_digits(n)} ماه به تاریخ {jalali_str(jd)} اضافه کنیم، تاریخ جدید (جلالی) چیست؟"
        a = jalali_str(new_d)
    else:
        n = random.randint(1, 11)
        total_months = (jd.month - 1) - n
        new_year = jd.year + total_months // 12
        new_month = total_months % 12 + 1
        max_day = 31 if new_month <= 6 else (30 if new_month <= 11 else 29)
        new_day = min(jd.day, max_day)
        new_d = jdatetime.date(new_year, new_month, new_day)
        q = f"اگر {to_fa_digits(n)} ماه از تاریخ {jalali_str(jd)} کم کنیم، تاریخ جدید (جلالی) چیست؟"
        a = jalali_str(new_d)
    samples.append({"category": "date_arithmetic", "question": q, "answer": a})

# 4. distance: days between two jalali dates, 37 samples
dist_count = 37
for i in range(dist_count):
    jd1 = random_jalali_date(1375, 1405)
    n = random.randint(1, 1000)
    jd2 = jd1 + datetime.timedelta(days=n)
    d1, d2 = (jd1, jd2) if random.random() < 0.5 else (jd2, jd1)
    diff = abs((d2.togregorian() - d1.togregorian()).days)
    q = f"چند روز بین تاریخ {jalali_str(d1)} و تاریخ {jalali_str(d2)} فاصله است؟"
    a = to_fa_digits(diff)
    samples.append({"category": "date_distance", "question": q, "answer": a})

random.shuffle(samples)

print(len(samples))
from collections import Counter
print(Counter(s["category"] for s in samples))

with open("/home/claude/calendar_gen/jalali_calendar_fa_raw.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
