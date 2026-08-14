"""
توابع کمکی rule-based که در چند تسک مختلف (پایداری زبان فارسی، استخراج
جواب نهایی، تسک‌های منطقی و IFEval) استفاده می‌شوند.
"""

import re
import json

_DEFAULT_WHITELIST = {"LATEX", "AI", "GPT", "URL", "JSON", "API", "ID"}

_LATIN_WORD_RE = re.compile(r"[a-zA-Z]{3,}")
_PERSIAN_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[a-zA-Z]")

# جداکننده‌های پایان جمله در فارسی (نقطه، علامت سوال/تعجب فارسی و لاتین)
_SENTENCE_SPLIT_RE = re.compile(r"[.!؟?]+")


def code_switch_rate(text: str, whitelist: set[str] | None = None) -> float:
    """تعداد کلمات لاتین غیرمجاز به ازای هر ۱۰۰ کلمه‌ی متن."""
    whitelist = whitelist or _DEFAULT_WHITELIST
    words = _LATIN_WORD_RE.findall(text)
    flagged = [w for w in words if w.upper() not in whitelist]
    total_words = max(len(text.split()), 1)
    return round(len(flagged) / total_words * 100, 3)


def persian_script_ratio(text: str) -> float:
    """نسبت کاراکترهای فارسی به مجموع کاراکترهای فارسی و لاتین."""
    persian_count = len(_PERSIAN_CHAR_RE.findall(text))
    latin_count = len(_LATIN_CHAR_RE.findall(text))
    total = persian_count + latin_count
    if total == 0:
        return 1.0
    return round(persian_count / total, 4)


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts


def split_first_last_third(text: str) -> tuple[str, str]:
    """متن را برای مقایسه‌ی افت کیفیت، به یک‌سوم ابتدایی و یک‌سوم پایانی تقسیم می‌کند."""
    words = text.split()
    n = len(words)
    if n < 6:
        return text, text
    third = n // 3
    first_third = " ".join(words[:third])
    last_third = " ".join(words[-third:])
    return first_third, last_third


def extract_final_answer(text: str, pattern: str = r"پاسخ نهایی:\s*(.+)") -> str | None:
    """جواب نهایی را طبق یک الگوی regex ثابت از انتهای زنجیره استدلال استخراج می‌کند.
    توصیه می‌شود در پرامپت مدل، خروجی را با همین قالب بخواهیم تا استخراج rule-based قابل‌اعتماد باشد."""
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def normalize_fa_text(text: str) -> str:
    """نرمال‌سازی سبک برای مقایسه‌ی جواب‌ها: حذف فاصله‌های اضافه، یکسان‌سازی نویسه‌های عربی/فارسی رایج."""
    text = text.strip()
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .،؛:\u200c")
    return text


def try_parse_json(text: str) -> dict | None:
    """تلاش برای parse کردن خروجی JSON مدل داور، حتی اگر با ```json محصور شده باشد."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # تلاش دوم: پیدا کردن اولین بلوک {...} در متن
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None
