# ruff: noqa: F405, F403, F401, E501, N802
"""Custom evaluation tasks for lighteval — Incepedia port.

This file is a **verbatim port** of huggingface/cosmopedia `evaluation/lighteval_tasks.py`
(commit fetched 2026-04-20). Keeping it byte-for-byte identical is critical: any prompt
tweak, metric swap, or few-shot change would break score comparability with published
Cosmopedia / SmolLM / FineWeb ablation numbers.

Upstream:
    https://github.com/huggingface/cosmopedia/blob/main/evaluation/lighteval_tasks.py
License (upstream): Apache-2.0 — same as this repo.

Snapshot of source kept at: third_party_sources/cosmopedia_lighteval_tasks.py

Tasks exposed:
    TASKS_TABLE        — list of LightevalTaskConfig for --custom_tasks
    TASKS_GROUPS       — pre-built task strings:
        'early-signal' — CSR + MMLU(mc/cloze) + MMLU-Pro + BoolQ + TriviaQA, 0-shot
        'math'         — GSM8k(5-shot) + MATH 7 subsets(4-shot)

Typical usage (one model, early-signal group):
    accelerate launch --num_processes=1 lighteval/run_evals_accelerate.py \
        --model_args="pretrained=<MODEL>" \
        --custom_tasks src/incepedia/eval/lighteval_tasks.py \
        --output_dir experiments/exp_xxx/eval \
        --tasks "$(python -c 'from incepedia.eval.lighteval_tasks import TASKS_GROUPS; print(TASKS_GROUPS[\"early-signal\"])')"

Differences from upstream: NONE (intentional). If upstream changes, decide explicitly
whether to port, and record the diff in an ADR.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

_TASKS_STRINGS: List[Tuple[LightevalTaskConfig, str]] = []
_TASKS: List[LightevalTaskConfig] = []

# ── COMMON_SENSE_REASONING_TASKS ──────────────────────────────────────
COMMON_SENSE_REASONING_TASKS = [
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function="hellaswag_prompt",
        hf_repo="hellaswag",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function="winogrande",
        hf_repo="winogrande",
        hf_subset="winogrande_xl",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function="piqa_harness",
        hf_repo="piqa",
        hf_subset="plain_text",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="siqa",
        prompt_function="siqa_prompt",
        hf_repo="lighteval/siqa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="openbookqa",
        prompt_function="openbookqa",
        hf_repo="openbookqa",
        hf_subset="main",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        generation_size=1,
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        generation_size=1,
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function="commonsense_qa_prompt",
        hf_repo="commonsense_qa",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="mmlu_pro_cloze",
        prompt_function="mmlu_pro_cloze_prompt",
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ),
    LightevalTaskConfig(
        name="mmlu_pro_mc",
        prompt_function="mmlu_pro_mc_prompt",
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select=None,
        generation_size=1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ),
    LightevalTaskConfig(
        name="boolq",
        prompt_function="boolq_prompt",
        hf_repo="super_glue",
        hf_subset="boolq",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        trust_dataset=True,
        stop_sequence=["\n"],
    ),
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function="triviaqa",
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metric=[Metrics.quasi_exact_match_triviaqa],
        generation_size=20,
        trust_dataset=True,
        stop_sequence=["\n", ".", ","],
        few_shots_select="random_sampling_from_train",
    ),
]


def boolq_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['passage']}\nQuestion: {line['question'].capitalize()}?\nAnswer:",
        choices=[" No", " Yes"],  # Only gold
        gold_index=int(line["label"]),
    )


def mmlu_pro_cloze_prompt(line, task_name: str = None):
    """MMLU-Pro prompt without letters."""
    topic = line["category"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["options"]],
        gold_index=line["answer_index"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


def mmlu_pro_mc_prompt(line, task_name: str = None):
    topic = line["category"]
    query = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["options"])])
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(line["options"])],
        gold_index=line["answer_index"],
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
        target_for_fewshot_sorting=LETTER_INDICES[line["answer_index"]],
    )


def commonsense_qa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
        instruction="",
    )


def siqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["context"] + " " + line["question"],
        choices=[f" {c}" for c in [line["answerA"], line["answerB"], line["answerC"]]],
        gold_index=int(line["label"]) - 1,
        instruction="",
    )


def hellaswag_prompt(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness."""
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
    )


GSM8K = LightevalTaskConfig(
    name="gsm8k",
    prompt_function="gsm8k",
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    metric=[Metrics.quasi_exact_match_gsm8k],
    generation_size=256,
    stop_sequence=["Question:", "Question"],
    few_shots_select="random_sampling_from_train",
)
MATH_TASKS = [
    LightevalTaskConfig(
        name=f"math:{subset}",
        prompt_function="math",
        hf_repo="lighteval/MATH",
        hf_subset=subset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[Metrics.quasi_exact_match_math],
        generation_size=256,
        stop_sequence=["Problem:", "Problem"],
        few_shots_select="random_sampling_from_train",
    )
    for subset in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
]

# 0-shot for common sense
COMMON_SENSE_REASONING_STRING = [(t, f"custom|{t.name}|0|1") for t in COMMON_SENSE_REASONING_TASKS]
_TASKS_STRINGS.extend(COMMON_SENSE_REASONING_STRING)
_TASKS_STRINGS.extend([(GSM8K, f"custom|{GSM8K.name}|5|1")])
_TASKS_STRINGS.extend([(t, f"custom|{t.name}|4|1") for t in MATH_TASKS])
_TASKS += COMMON_SENSE_REASONING_TASKS
_TASKS += [GSM8K] + MATH_TASKS


# ── MMLU ──────────────────────────────────────────────────────────────
class CustomMMLUEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="mmlu_prompt",
        hf_repo="lighteval/mmlu",
        hf_subset=None,
        metric=None,
        hf_avail_splits=None,
        evaluation_splits=None,
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        if metric is None:
            metric = [Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace]
        if evaluation_splits is None:
            evaluation_splits = ["test"]
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


MMLU_TASKS: list = []
MMLU_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
]

for answer_type in ("mc", "cloze"):
    prompt_function = f"mmlu_{answer_type}_prompt"
    generation_size = -1 if answer_type == "cloze" else 1
    for subset in MMLU_SUBSETS:
        MMLU_TASKS.append(
            CustomMMLUEvaluationTask(
                name=f"mmlu_{answer_type}:{subset}",
                prompt_function=prompt_function,
                hf_subset=subset,
                generation_size=generation_size,
            )
        )

MMLU_TASKS += [
    CustomMMLUEvaluationTask(
        name="mmlu_stem_mc",
        hf_repo="TIGER-Lab/MMLU-STEM",
        prompt_function="mmlu_mc_prompt",
        hf_subset="default",
        generation_size=1,
    ),
    CustomMMLUEvaluationTask(
        name="mmlu_stem_cloze",
        hf_repo="TIGER-Lab/MMLU-STEM",
        prompt_function="mmlu_cloze_prompt",
        hf_subset="default",
        generation_size=-1,
    ),
]


def mmlu_cloze_prompt(line, task_name: str = None):
    """MMLU prompt without letters."""
    topic = line["subject"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


def mmlu_mc_prompt(line, task_name: str = None):
    topic = line["subject"]
    query = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    query += "Answer:"

    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
        target_for_fewshot_sorting=[" A", " B", " C", " D"][gold_ix],
    )


MMLU_STRING = [(t, f"custom|{t.name}|0|1") for t in MMLU_TASKS]
_TASKS_STRINGS.extend(MMLU_STRING)
_TASKS += MMLU_TASKS

# ── Public exports ────────────────────────────────────────────────────
EARLY_SIGNAL_TASKS = ",".join(
    [t[1] for t in COMMON_SENSE_REASONING_STRING] + [t[1] for t in MMLU_STRING]
)

TASKS_TABLE = _TASKS
TASKS_GROUPS = {
    "early-signal": EARLY_SIGNAL_TASKS,
    "math": f"custom|{GSM8K.name}|5|1" + "," + ",".join([f"custom|{t.name}|4|1" for t in MATH_TASKS]),
}


__all__ = [
    "TASKS_TABLE",
    "TASKS_GROUPS",
    "EARLY_SIGNAL_TASKS",
    "COMMON_SENSE_REASONING_TASKS",
    "MMLU_TASKS",
    "GSM8K",
    "MATH_TASKS",
    "CustomMMLUEvaluationTask",
]
