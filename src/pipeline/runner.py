"""
اجراکننده‌ی عمومی پایپ‌لاین ارزیابی. این ماژول مستقل از تسک خاص است:
هر تسک فقط باید نمونه‌هایش را به فرمت TaskSample (jsonl) آماده کند و بقیه
(فراخوانی مدل، استخراج جواب، اعمال rubricهای مشترک، ذخیره نتیجه) توسط این
runner انجام می‌شود.

استفاده (نمونه):
    python -m src.pipeline.runner \
        --task-file data/tasks/bbh_logic.jsonl \
        --model-name gpt-oss-20b-fa-cot-v1 \
        --output-file data/scored_results/bbh_logic__gpt-oss-20b-fa-cot-v1.jsonl
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from src.providers.factory import get_candidate_provider, get_judge_provider, load_models_config
from src.pipeline.schemas import TaskSample, ModelGeneration, ScoredResult
from src.judges.rubric_judge import VerifiableReasoningJudge
from src.judges.persian_stability_judge import PersianStabilityJudge
from src.utils.text_utils import extract_final_answer, normalize_fa_text
from src.utils.instruction_utils import check_constraint

DEFAULT_SYSTEM_PROMPT_FA = (
    "تو یک دستیار هوشمند فارسی‌زبان هستی. به سوال زیر با دقت و به زبان فارسی پاسخ بده. "
    "در پایان پاسخ خود، دقیقاً یک خط جداگانه با این قالب بنویس:\n"
    "پاسخ نهایی: <جواب کوتاه و دقیق>"
)


def load_task_samples(path: str) -> list[TaskSample]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(TaskSample(**json.loads(line)))
    return samples


def generate_for_sample(provider, sample: TaskSample) -> ModelGeneration:
    system_prompt = sample.system_prompt_fa or DEFAULT_SYSTEM_PROMPT_FA
    result = provider.generate(system_prompt=system_prompt, user_prompt=sample.problem_fa)
    extracted = extract_final_answer(result.text, sample.answer_extraction_regex)
    return ModelGeneration(
        sample_id=sample.sample_id,
        model_name=provider.model_id,
        raw_output=result.text,
        extracted_final_answer=extracted,
        finish_reason=result.finish_reason,
    )


def score_sample(
    sample: TaskSample,
    generation: ModelGeneration,
    vr_judge: VerifiableReasoningJudge | None,
    ps_judge: PersianStabilityJudge | None,
) -> ScoredResult:
    result = ScoredResult(
        sample_id=sample.sample_id,
        model_name=generation.model_name,
        task_type=sample.task_type,
        raw_output=generation.raw_output,
    )

    # درستی ساده (rule-based) در صورت وجود gold_answer
    if sample.gold_answer is not None:
        extracted = generation.extracted_final_answer or ""
        result.correctness = int(
            normalize_fa_text(extracted) == normalize_fa_text(sample.gold_answer)
        )

    # درستی مبتنی بر محدودیت قابل‌راستی‌آزمایی با قانون (IFEval-lite)
    elif sample.extra.get("constraint") is not None:
        passed, reason_fa = check_constraint(generation.raw_output, sample.extra["constraint"])
        result.correctness = int(passed)
        result.notes = reason_fa

    # اعمال rubric استدلال قابل‌راستی‌آزمایی
    if sample.apply_verifiable_reasoning_rubric and vr_judge is not None:
        vr_score = vr_judge.score(
            problem=sample.problem_fa,
            cot=generation.raw_output,
            final_answer=generation.extracted_final_answer or "",
        )
        result.verifiable_reasoning_total = vr_score.total
        result.verifiable_reasoning_breakdown = {
            "step_support": vr_score.step_support,
            "conclusion_validity": vr_score.conclusion_validity,
            "no_logical_jump": vr_score.no_logical_jump,
            "answer_consistency": vr_score.answer_consistency,
            "brief_reason_fa": vr_score.brief_reason_fa,
            "parse_success": vr_score.parse_success,
        }

    # اعمال rubric پایداری زبان فارسی
    if sample.apply_persian_stability_rubric and ps_judge is not None:
        ps_score = ps_judge.score(cot=generation.raw_output)
        result.persian_stability_llm_total = ps_score.llm_part_total
        result.code_switch_rate = ps_score.code_switch_rate_per_100_words
        result.persian_script_ratio = ps_score.persian_script_ratio
        result.persian_stability_breakdown = {
            "grammatical_fluency": ps_score.grammatical_fluency,
            "terminology_consistency": ps_score.terminology_consistency,
            "degradation_over_length": ps_score.degradation_over_length,
            "brief_reason_fa": ps_score.brief_reason_fa,
            "parse_success": ps_score.parse_success,
        }

    return result


def run(task_file: str, model_name: str, output_file: str,
        config_path: str = "configs/models.yaml", judge_name: str = "judge-primary"):
    samples = load_task_samples(task_file)
    candidate_provider = get_candidate_provider(model_name, config_path)

    needs_vr = any(s.apply_verifiable_reasoning_rubric for s in samples)
    needs_ps = any(s.apply_persian_stability_rubric for s in samples)

    vr_judge = None
    ps_judge = None
    if needs_vr or needs_ps:
        judge_provider = get_judge_provider(judge_name, config_path)
        if needs_vr:
            vr_judge = VerifiableReasoningJudge(judge_provider)
        if needs_ps:
            ps_judge = PersianStabilityJudge(judge_provider)

    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc=f"در حال اجرای {model_name} روی {task_file}"):
            generation = generate_for_sample(candidate_provider, sample)
            scored = score_sample(sample, generation, vr_judge, ps_judge)
            out_f.write(scored.model_dump_json() + "\n")

    print(f"نتایج ذخیره شد در: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="اجراکننده عمومی پایپ‌لاین ارزیابی")
    parser.add_argument("--task-file", required=True, help="مسیر فایل jsonl نمونه‌های تسک")
    parser.add_argument("--model-name", required=True, help="نام مدل کاندید مطابق configs/models.yaml")
    parser.add_argument("--output-file", required=True, help="مسیر خروجی jsonl نتایج امتیازدهی‌شده")
    parser.add_argument("--config-path", default="configs/models.yaml")
    parser.add_argument("--judge-name", default="judge-primary")
    args = parser.parse_args()

    run(
        task_file=args.task_file,
        model_name=args.model_name,
        output_file=args.output_file,
        config_path=args.config_path,
        judge_name=args.judge_name,
    )


if __name__ == "__main__":
    main()
