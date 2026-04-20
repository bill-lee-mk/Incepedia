# 0006 · Evaluation stack policy — pin to lighteval latest

- **Status**: accepted
- **Date**: 2026-04-20
- **Deciders**: bill-lee-mk

- **Context**:
  - `huggingface/cosmopedia` 的 `evaluation/lighteval_tasks.py` 是 2024 年 Q1 针对 lighteval pre-v0.1 主分支写的。之后 lighteval 经历了多次 breaking API 变更:
    - v0.4+:`prompt_function` 从 `str` 改为 `Callable`
    - v0.5+:`suite`、`trust_dataset` 等字段语义调整
    - v0.8+:移除 `output_regex` / `frozen`
    - v0.11+:移除 `default_prompts.py`(`LETTER_INDICES` 不在了)
    - v0.13+:`metric=` → `metrics=`;删除 `quasi_exact_match_*`、`loglikelihood_acc_norm_nospace` 等变体;built-in 标准 task 改用 MC+exact_match 形式
  - Pin 到 v0.10.0 能最小改动地跑 cosmopedia 原文件,但随时间推移形成技术债:
    - 6 个月未更新,bug 自生自灭
    - 新 task(多语言、agent、code)无法白嫖社区贡献
    - 跟 torch / numpy / typer 等上游依赖的兼容空间越来越窄
    - 项目跨 12+ 月生命周期内只会越卡越重
  - Incepedia 是一个**长期、多版本、多人协作**的项目,代码健康度权重高于"初期省 4 小时"。

- **Decision**:
  **评测栈跟随 lighteval 最新 pypi 版(当前 0.13),不 pin 老版本。**

  我们自维护一份 `src/incepedia/eval/lighteval_tasks.py`,其中:
  - **Prompt 逻辑** 逐字节继承 cosmopedia 原版(不改 prompt 文本、不改 few-shot 数、不改数据集子集)
  - **API 层** 机械适配到 lighteval 当前版本:
    - `prompt_function=<callable>`(0.4+)
    - `metrics=[...]`(0.13)
    - 去掉 `output_regex` / `frozen`
    - `LETTER_INDICES` 改从 `string.ascii_uppercase` 取
  - **Metric 降级**(0.13 不再支持的):
    - `Metrics.loglikelihood_acc_norm_nospace` → `Metrics.loglikelihood_acc`(丢失 length-normalization,~0.1-0.2pp 噪声)
    - `Metrics.quasi_exact_match_{gsm8k,math,triviaqa}` → `Metrics.exact_match`
  - **Runner** 绕过 typer CLI(当前 broken),用 `from lighteval.main_accelerate import accelerate` 编程调用,外层包一层 `accelerate launch` 启动多 GPU

  预期:我方 Incepedia 跑出的 cosmopedia 基线分数与 SmolLM blog 公布值差 **<0.5pp**(2-seed 噪声为 ±0.15pp,完全在 delta 分析容忍范围内)。

  **不变量**:Incepedia vs Cosmopedia 的 delta 恒有效 — 因为两边用同一 lighteval、同一 port 文件、同一 seed 跑。

- **Alternatives considered**:
  1. **Pin lighteval v0.10.0 + 机械 patch cosmopedia 原文件**
     - 初期省 4 小时
     - 中长期:上游 bug 不修、CLI/typer 组合被锁、社区新 task 白嫖不到、12+ 月后可读性雪崩
     - 否决:项目周期太长,债务复利滚大
  2. **使用 lighteval v0.13 的内置 task(不写 port)**
     - 极简,跟上游
     - 但 v0.13 hellaswag/piqa/openbookqa 等用 MC+exact_match,对 1.82B 模型基本 random,**失去小模型 ablation 分辨率**
     - 否决:科学需求决定我们要 cloze-style loglikelihood
  3. **自己 fork 一份 lighteval**
     - 自由度最大
     - 维护成本最高(要跟上游合并),收益不匹配
     - 否决

- **Consequences**:
  - **Positive**:
    - 跟随 maintained 上游,未来长期收益(bug 修、新 task、新 metric)
    - 代码健康度随时间增长,不退化
    - 新贡献者读代码不用学"为什么 pin 老版本"
    - CLI 修复后可直接切换到 CLI 调用路径
  - **Negative / trade-offs**:
    - 与公开 SmolLM / Cosmopedia blog 数字有 <0.5pp 的绝对漂移(可接受)
    - 每次 lighteval minor release 有小概率再触发 API 适配(历史经验:每 6 个月一次,1-2 小时工作量)
    - 当前 typer CLI broken,runner 用 driver-script workaround(文档已注明,CLI 恢复后可切回)
  - **Follow-ups**:
    - 每次 lighteval 新版本发布,agent 跑一次 `pytest tests/eval/test_api_compat.py`(未建),若失败则 patch 我方 `lighteval_tasks.py`
    - 若上游 CLI 稳定,回写 runner 用 CLI 调用,以利用未来 CLI 增强(subprocess 更简单)
    - 可选:建立"每季度 lighteval 升级"cron check,写进 AGENTS.md

- **Related**:
  - ADR 0004 · 双轨评测协议(用本文定义的 port 作为双轨的共同评测器)
  - `src/incepedia/eval/lighteval_tasks.py`:本决定的实现
  - `src/incepedia/eval/runner.py`:driver-script workaround
  - `third_party_sources/cosmopedia_lighteval_tasks.py`:upstream 快照(read-only 参考)
