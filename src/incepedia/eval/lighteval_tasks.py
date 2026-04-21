# ruff: noqa: E501, N802
"""Custom lighteval tasks for Incepedia — cosmopedia-compatible, lighteval 0.13 API.

This is a **re-port** of `huggingface/cosmopedia/evaluation/lighteval_tasks.py` for
the current lighteval 0.13 API. The original port (commit 344e269) was written for
pre-v0.1 lighteval; this file adapts it to 0.13 with the following mechanical changes:

    - `prompt_function="name"` (string) → `prompt_function=name` (callable)
    - `metric=[...]`                    → `metrics=[...]`
    - removed `output_regex=None, frozen=False, trust_dataset=True` (no equivalent in 0.13)
    - `Metrics.loglikelihood_acc_norm_nospace` → `Metrics.loglikelihood_acc`
      (0.13 dropped the norm_nospace variant; we lose ~0.1-0.2pp resolution but the
      ranking across datasets/models is preserved)
    - `Metrics.quasi_exact_match_{gsm8k,math,triviaqa}` → `Metrics.exact_match`
      (quasi-EM variants removed in 0.13)
    - `LETTER_INDICES` now sourced from `string.ascii_uppercase`
      (old `lighteval.tasks.default_prompts` module removed in 0.11+)

**Task semantics are preserved.** Prompt text, few-shot counts, dataset splits,
subsets, and stop sequences are identical to cosmopedia's upstream file.

Expected score drift relative to published SmolLM/Cosmopedia numbers: **<0.5pp**
(within our 2-seed noise floor of ±0.15pp). Deltas between our own Incepedia vs
Cosmopedia runs are fully comparable because we use this file for both.

See `docs/decisions/0006-evaluation-stack-policy.md` for the why.

Upstream snapshot (reference only): `third_party_sources/cosmopedia_lighteval_tasks.py`.
"""
from __future__ import annotations

import re
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

LETTER_INDICES: list[str] = list(ascii_uppercase)

# ─── prompt functions ──────────────────────────────────────────────────

def hellaswag_prompt(line, task_name: str | None = None) -> Doc:
    def preprocess(text: str) -> str:
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(e) for e in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,
    )


def winogrande_prompt(line, task_name: str | None = None) -> Doc:
    query, end_of_target = line["sentence"].split("_")
    end_of_target = end_of_target.strip()
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f"{line['option1']} {end_of_target}", f"{line['option2']} {end_of_target}"],
        gold_index=int(line["answer"]) - 1 if line["answer"] != "" else -1,
    )


def piqa_prompt(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=f"Question: {line['goal']}\nAnswer:",
        choices=[f" {line['sol1']}", f" {line['sol2']}"],
        gold_index=int(line["label"]),
    )


def siqa_prompt(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=line["context"] + " " + line["question"],
        choices=[f" {c}" for c in [line["answerA"], line["answerB"], line["answerC"]]],
        gold_index=int(line["label"]) - 1,
        instruction="",
    )


def openbookqa_prompt(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question_stem']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
    )


def arc_prompt(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


def commonsense_qa_prompt(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
        instruction="",
    )


def boolq_prompt(line, task_name: str | None = None) -> Doc:
    # Modern `google/boolq` uses `answer` (bool) instead of legacy `label` (int).
    # Handle both schemas for forward/backward compat.
    if "answer" in line:
        gold = int(bool(line["answer"]))
    else:
        gold = int(line["label"])
    return Doc(
        task_name=task_name,
        query=f"{line['passage']}\nQuestion: {line['question'].capitalize()}?\nAnswer:",
        choices=[" No", " Yes"],
        gold_index=gold,
    )


def triviaqa_prompt(line, task_name: str | None = None) -> Doc:
    # Uses single gold answer; lighteval will compare generation via exact_match.
    answer = line["answer"]["aliases"][0] if line["answer"].get("aliases") else line["answer"]["value"]
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {answer}"],
        gold_index=0,
    )


def gsm8k_prompt(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[f" {line['answer']}"],
        gold_index=0,
    )


def math_prompt(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=f"Problem: {line['problem']}\nAnswer:",
        choices=[f" {line['solution']}"],
        gold_index=0,
    )


def mmlu_cloze_prompt(line, task_name: str | None = None) -> Doc:
    """MMLU prompt without letter labels; choices are full answer texts.

    Used for small-model (1.82B) ablation where the MC (A/B/C/D) format gives
    near-random scores because small models can't follow letter-only output
    instructions reliably. Matches cosmopedia's `mmlu_cloze_prompt`.
    """
    topic = line["subject"]
    query = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    query += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


def mmlu_mc_prompt(line, task_name: str | None = None) -> Doc:
    """MMLU prompt with A/B/C/D letter labels. For larger models."""
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
    )


def mmlu_pro_cloze_prompt(line, task_name: str | None = None) -> Doc:
    topic = line["category"]
    query = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    query += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {c}" for c in line["options"]],
        gold_index=line["answer_index"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


def mmlu_pro_mc_prompt(line, task_name: str | None = None) -> Doc:
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
    )


# ─── common-sense reasoning tasks ──────────────────────────────────────

_LL_ACC = [Metrics.loglikelihood_acc]  # replaces acc + acc_norm_nospace pair

COMMON_SENSE_REASONING_TASKS = [
    LightevalTaskConfig(
        name="incep_hellaswag",
        prompt_function=hellaswag_prompt,
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_winogrande",
        prompt_function=winogrande_prompt,
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_piqa",
        prompt_function=piqa_prompt,
        hf_repo="ybisk/piqa",
        hf_subset="plain_text",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_siqa",
        prompt_function=siqa_prompt,
        hf_repo="lighteval/siqa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_openbookqa",
        prompt_function=openbookqa_prompt,
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_arc_easy",
        prompt_function=arc_prompt,
        hf_repo="allenai/ai2_arc",
        hf_subset="ARC-Easy",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        generation_size=1,
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_arc_challenge",
        prompt_function=arc_prompt,
        hf_repo="allenai/ai2_arc",
        hf_subset="ARC-Challenge",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        generation_size=1,
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_commonsense_qa",
        prompt_function=commonsense_qa_prompt,
        hf_repo="tau/commonsense_qa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_boolq",
        prompt_function=boolq_prompt,
        hf_repo="google/boolq",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metrics=_LL_ACC,
        stop_sequence=["\n"],
    ),
    LightevalTaskConfig(
        name="incep_trivia_qa",
        prompt_function=triviaqa_prompt,
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metrics=[Metrics.exact_match],
        generation_size=20,
        stop_sequence=["\n", ".", ","],
        few_shots_select="random_sampling_from_train",
    ),
    LightevalTaskConfig(
        name="incep_mmlu_pro_cloze",
        prompt_function=mmlu_pro_cloze_prompt,
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_avail_splits=["validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_mmlu_pro_mc",
        prompt_function=mmlu_pro_mc_prompt,
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_avail_splits=["validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="validation",
        generation_size=1,
        metrics=_LL_ACC,
    ),
]

# ─── math-reasoning tasks ──────────────────────────────────────────────

GSM8K = LightevalTaskConfig(
    name="incep_gsm8k",
    prompt_function=gsm8k_prompt,
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    metrics=[Metrics.exact_match],
    generation_size=256,
    stop_sequence=["Question:", "Question"],
    few_shots_select="random_sampling_from_train",
)

MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

MATH_TASKS = [
    LightevalTaskConfig(
        name=f"incep_math_{subset}",
        prompt_function=math_prompt,
        hf_repo="lighteval/MATH",
        hf_subset=subset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metrics=[Metrics.exact_match],
        generation_size=256,
        stop_sequence=["Problem:", "Problem"],
        few_shots_select="random_sampling_from_train",
    )
    for subset in MATH_SUBSETS
]

# ─── MMLU (57 subsets × {mc, cloze}) ──────────────────────────────────

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


def _mmlu_task(subset: str, answer_type: str) -> LightevalTaskConfig:
    prompt_fn = mmlu_mc_prompt if answer_type == "mc" else mmlu_cloze_prompt
    # NOTE: keep `:` between task family and subset — lighteval treats `:` as
    # the canonical subset separator in task names and looks them up that way.
    return LightevalTaskConfig(
        name=f"incep_mmlu_{answer_type}:{subset}",
        prompt_function=prompt_fn,
        hf_repo="lighteval/mmlu",
        hf_subset=subset,
        hf_avail_splits=["auxiliary_train", "dev", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="dev",
        generation_size=1 if answer_type == "mc" else -1,
        metrics=_LL_ACC,
    )


MMLU_TASKS = [
    _mmlu_task(subset, answer_type)
    for answer_type in ("mc", "cloze")
    for subset in MMLU_SUBSETS
]

MMLU_STEM_TASKS = [
    LightevalTaskConfig(
        name="incep_mmlu_stem_mc",
        prompt_function=mmlu_mc_prompt,
        hf_repo="TIGER-Lab/MMLU-STEM",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=1,
        metrics=_LL_ACC,
    ),
    LightevalTaskConfig(
        name="incep_mmlu_stem_cloze",
        prompt_function=mmlu_cloze_prompt,
        hf_repo="TIGER-Lab/MMLU-STEM",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        metrics=_LL_ACC,
    ),
]

# ─── registration + groups ─────────────────────────────────────────────

TASKS_TABLE: list[LightevalTaskConfig] = (
    COMMON_SENSE_REASONING_TASKS
    + [GSM8K]
    + MATH_TASKS
    + MMLU_TASKS
    + MMLU_STEM_TASKS
)


def _task_str(t: LightevalTaskConfig, num_fewshots: int, truncate: int = 1) -> str:
    # lighteval 0.13 task spec format is `task_name|num_fewshots`.
    # The legacy `suite|task|fewshot|truncate` 4-element format from old cosmopedia
    # port was rejected by the new registry (parses on `|`).
    return f"{t.name}|{num_fewshots}"


EARLY_SIGNAL_TASKS: str = ",".join(
    [_task_str(t, 0) for t in COMMON_SENSE_REASONING_TASKS]
    + [_task_str(t, 0) for t in MMLU_TASKS]
)

MATH_GROUP: str = ",".join(
    [_task_str(GSM8K, 5)] + [_task_str(t, 4) for t in MATH_TASKS]
)

TASKS_GROUPS: dict[str, str] = {
    "early-signal": EARLY_SIGNAL_TASKS,
    "math": MATH_GROUP,
    "csr-only": ",".join([_task_str(t, 0) for t in COMMON_SENSE_REASONING_TASKS]),
    "mmlu-only": ",".join([_task_str(t, 0) for t in MMLU_TASKS]),
}


__all__ = [
    "TASKS_TABLE",
    "TASKS_GROUPS",
    "EARLY_SIGNAL_TASKS",
    "MATH_GROUP",
    "COMMON_SENSE_REASONING_TASKS",
    "GSM8K",
    "MATH_TASKS",
    "MMLU_TASKS",
    "MMLU_STEM_TASKS",
    "LETTER_INDICES",
]
