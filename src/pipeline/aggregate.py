"""
تجمیع فایل‌های scored_results (خروجی runner.py) و تولید یک گزارش خلاصه
شامل میانگین correctness، میانگین امتیاز rubric راستی‌آزمایی و میانگین
پایداری زبان فارسی، برای هر ترکیب (task_type, model_name).

استفاده:
    python -m src.pipeline.aggregate --results-dir data/scored_results --out report.json
"""

import argparse
import json
import glob
from collections import defaultdict


def load_all_results(results_dir: str) -> list[dict]:
    rows = []
    for path in glob.glob(f"{results_dir}/*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        key = (row["task_type"], row["model_name"])
        groups[key].append(row)

    report = {}
    for (task_type, model_name), items in groups.items():
        correctness_vals = [r["correctness"] for r in items if r.get("correctness") is not None]
        vr_vals = [r["verifiable_reasoning_total"] for r in items if r.get("verifiable_reasoning_total") is not None]
        ps_vals = [r["persian_stability_llm_total"] for r in items if r.get("persian_stability_llm_total") is not None]
        cs_vals = [r["code_switch_rate"] for r in items if r.get("code_switch_rate") is not None]

        key_str = f"{task_type} :: {model_name}"
        report[key_str] = {
            "n_samples": len(items),
            "avg_correctness": round(sum(correctness_vals) / len(correctness_vals), 4) if correctness_vals else None,
            "avg_verifiable_reasoning_total_out_of_12": round(sum(vr_vals) / len(vr_vals), 3) if vr_vals else None,
            "avg_persian_stability_llm_total_out_of_8": round(sum(ps_vals) / len(ps_vals), 3) if ps_vals else None,
            "avg_code_switch_rate_per_100_words": round(sum(cs_vals) / len(cs_vals), 3) if cs_vals else None,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="تجمیع نتایج ارزیابی و تولید گزارش خلاصه")
    parser.add_argument("--results-dir", default="data/scored_results")
    parser.add_argument("--out", default="report.json")
    args = parser.parse_args()

    rows = load_all_results(args.results_dir)
    report = aggregate(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nگزارش کامل ذخیره شد در: {args.out}")


if __name__ == "__main__":
    main()
