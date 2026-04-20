# incepedia.eval

Evaluation layer. Mirrors the `huggingface/cosmopedia/evaluation/` setup so scores are
directly comparable to published Cosmopedia / SmolLM / FineWeb ablation numbers.

## Files

- `lighteval_tasks.py` — verbatim port of cosmopedia's task defs. Exports `TASKS_TABLE`
  (list of `LightevalTaskConfig`) and `TASKS_GROUPS` dict with pre-built task strings:
    - `"early-signal"` — CSR + MMLU(mc/cloze) + MMLU-Pro + BoolQ + TriviaQA
    - `"math"` — GSM8k(5-shot) + MATH 7 subsets(4-shot)

## Tasks at a glance

**Common-sense reasoning (0-shot)** — hellaswag, winogrande, piqa, siqa, openbookqa, arc:easy, arc:challenge, commonsense_qa, boolq, trivia_qa

**Knowledge / reasoning** — MMLU 57 subjects × {mc, cloze}, MMLU-Pro × {mc, cloze}, MMLU-STEM × {mc, cloze}

**Math / generation** — GSM8k (5-shot), MATH 7 subsets (4-shot)

**Metrics** — `loglikelihood_acc` + `loglikelihood_acc_norm_nospace` for CSR/MMLU; `quasi_exact_match` for GSM8k / MATH / TriviaQA.

## Running evaluations

The standard recipe (one model, early-signal group) is wrapped by `scripts/run_eval.sh`
(TBD). Until then, invoke lighteval directly:

```bash
MODEL=HuggingFaceTB/cosmo-1b
EARLY_SIGNAL=$(python -c 'from incepedia.eval.lighteval_tasks import TASKS_GROUPS; print(TASKS_GROUPS["early-signal"])')

accelerate launch --num_processes=8 --main_process_port=29600 \
  lighteval/run_evals_accelerate.py \
  --model_args="pretrained=$MODEL" \
  --custom_tasks src/incepedia/eval/lighteval_tasks.py \
  --output_dir experiments/exp_xxx/eval \
  --override_batch_size 16 \
  --tasks "$EARLY_SIGNAL"
```

For math / HumanEval add the `math` group / run `bigcode-evaluation-harness` separately.

## DO NOT modify task definitions

Any tweak breaks score comparability. If a task needs to change, record why in an ADR
(`docs/decisions/`) and create a **new** task variant rather than editing the original.
