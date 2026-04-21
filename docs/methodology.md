# Incepedia · 方法论

> 本文档是项目的"北极星":所有实验决策最终都回到这里对齐。
> 语言:中文为主,核心 ADR 另存于 `docs/decisions/`。

---

## 1 · 问题定义

**做什么**:构造一个合成预训练数据集 `Incepedia`,在标准 ablation 协议下,下游 early-signal benchmark 分数**同时超越**:
- Cosmopedia v1 在"独立 corpus"场景下的分数(Cosmo-1B 路线)
- Cosmopedia v2 在"decay 调料"场景下的分数(SmolLM2 路线)

**为什么做两个场景都要**(详见 [ADR 0005](./decisions/0005-project-scope-standalone-plus-seasoning.md)):
- Cosmopedia v1 / v2 在产业中分别代表了"主力"和"调料"两种合成数据使用范式
- 不锁死单一场景,才能兑现"不管上游 2027 把合成数据定位在哪,Incepedia 都能打"

**不做什么**:
- 不追求单纯 token 规模。**信息密度 × 多样性 > token 总量**。
- 不做通用 SFT / RLHF 数据,只聚焦 pretraining corpus。

---

## 2 · 评判数据质量的唯一标尺:Ablation Training(双轨自适应)

详见 [ADR 0004](./decisions/0004-evaluation-protocol-dual-track-adaptive.md)。

### 2.1 操作化定义

> **数据质量 := 在 1.82B Llama2 架构下,通过 ablation 训练得到的 lighteval early-signal benchmark 平均分。**

任何"数据看起来高质量 / 信息密度高 / 可读性好"等主观判断都不算证据,**必须通过 ablation 训练验证**。

### 2.2 两条评测轨道

| | **Track 1 · Standalone** | **Track 2 · Seasoning** |
|---|---|---|
| 协议 | 1.82B Llama2 / 1.7B Qwen3 从头训 × 30B tokens × 2 seeds | 共享 backbone(FineWeb-Edu × 20B)+ cooldown-fork × 6B tokens |
| 答什么 | "数据集 X 作为独立 pretraining corpus,比 Y 强吗?" | "数据集 X 作为 decay 阶段调料,比 Y 强吗?" |
| 单次墙钟(Llama2) | ~4.2 天(2 seeds) | backbone 1.4 天 + 每 fork ~10 h |
| 单次墙钟(Qwen3) | ~4.0 天(2 seeds) | backbone 1.3 天 + 每 fork ~9 h |
| 典型用途 | 版本 milestone 验收 | 日常参数/配比 ablation |
| 对应论文 | Cosmo-1B / FineWeb / Phi-1 | SmolLM2 stage-4 decay、Hägele et al. 2024 |

### 2.3 双协议架构(详见 ADR 0007)

**两条评测轨道 × 两个架构 = 双协议矩阵**:

| 协议 | 架构 | 作用 | 频率 |
|---|---|---|---|
| **Protocol A** | Llama2-1.82B(全注意力 / RoPE θ=10000 / 无 bias) | **外部锚** — 验证 pipeline,对齐 SmolLM/FineWeb 公开数字 | 全项目 2 次(Cosmopedia v2 × 2 seeds) |
| **Protocol B** | Qwen3-style 1.7B(GQA 8 KV / RoPE θ=1e6 / QKV bias) | **工作架构** — 所有日常 ablation + Incepedia 版本演进 | Track 1 milestones 6 + Track 2 backbone 1 + forks ~33 |

**关键不变项**:两协议共用 Mistral-7B-v0.1 tokenizer、同一份 tokenized 数据、同 lr / batch / seeds、同 lighteval port。**唯一变量是架构本身**,delta 可解释为纯架构效应。

### 2.3 选择规则(自适应)

**每个 ablation 开始前,先问:"这实验想答的是独立能力还是调料能力?"**

- 版本级 milestone / 独立 corpus 对比 → **Track 1**
- 参数微调(prompt / persona / critic / 配比 / 结构 / 难度)→ **Track 2**
- **不设前置配额**,事后盘点比例作为结果记录

预估自然分布:Track 1 ~15%、Track 2 ~85%(约 6 + 33 ≈ 40 次 ablation)。

### 2.4 Early-signal benchmark 套件

**常识 (CSR, 0-shot)**:HellaSwag / Winogrande / PIQA / SIQA / OpenBookQA / ARC-Easy / ARC-Challenge / CommonsenseQA / BoolQ

**知识 / 推理**:MMLU(cloze + MC,57 subset)、MMLU-Pro(cloze + MC)、MMLU-STEM、TriviaQA(0/5-shot)

**数学 / 生成**:GSM8k(5-shot)、MATH 7 subset(4-shot)

**代码**:HumanEval(`bigcode-eval-harness`, T=0.2, top-p=0.95, n=20)

**Metric**:`loglikelihood_acc` + `loglikelihood_acc_norm_nospace`(CSR/MMLU);`quasi_exact_match`(GSM8k/MATH/TriviaQA);`pass@1`(HumanEval)

### 2.5 为什么这些具体数字

- **1.82B 参数**:FineWeb ablation 标准,给小模型benchmark 足够信号但成本可控
- **30B tokens(Track 1)**:约 Chinchilla 最优的 80%,分数在小 benchmark 上已 converge,相对排序稳定
- **20B backbone + 6B cooldown(Track 2)**:backbone 给 cooldown 足够的稳定起点(Hägele et al. 最小 15B);cooldown 占比 23%(6B/26B)在 trapezoidal 最优区间;总 26B token ≈ Track 1 的 30B,两轨 token 量级可比
- **2 seeds**:FineWeb 论文惯例,噪声降到 ±0.15pp

### 2.6 **不做** proxy 小模型

初期需要最高保真度信号,200M/400M 的 scaling 转手到 1.82B 不可靠。宁可每次多等几天。

---

## 3 · Incepedia 生成策略 — C1 到 C16

所有决策详见 [ADR 0003](./decisions/0003-multi-tier-generator-routing.md)(路由部分),生成器配置在 `configs/generators.yaml`。

### 3.1 基础策略(C1 – C4)

| # | 决策 | 作用 |
|---|---|---|
| **C1** | **强制 seed-grounding** — 每条 prompt 必带 Wikipedia / arXiv / FineWeb-Edu 段落 + 反幻觉指令 | 降低 Mixtral 时代的幻觉(Cosmopedia 公认痛点) |
| **C2** | **Persona 注入** — 从 `configs/personas.yaml`(P2 扩到 ~10k)按权重抽 | 反 L3 同质化(风格单调) |
| **C3** | **5 档难度** — elementary / middle / high_school / college / expert | 支撑课程式实验(C16) |
| **C4** | **12+ 结构模板** — textbook / blog / wikihow / story / Q&A / Socratic / case_study / debate / lecture_notes / glossary / cheatsheet / worked_example / misconception_correction | 反 L1 同质化(模板单一) |

### 3.2 生成器 & 生产线(C5 – C12)

| # | 决策 |
|---|---|
| **C5** | 双 OpenRouter key 轮换,提并发 / 降限流 |
| **C6** | `httpx.AsyncClient` + `aiolimiter` + `tenacity` 指数退避 |
| **C7** | Checkpoint:每 1000 条写一个 Parquet shard + 原子重命名 |
| **C8** | 实时监控:每 shard 统计成功率 / 平均 token / 成本 / dup 率 |
| **C9** | 成本闸门:每 10k 样本实际成本偏离预估 >20% → 自动暂停等人确认 |
| **C10** | Critic loop(仅 STEM / code / expert 档,~15% 样本触发) |
| **C11** | 输出 schema(每行含 generator / temperature / seed_text_hash / critic_score / revision_of 等完整元数据) |
| **C12** | 每批合成由 `batch.yaml` 完全定义(git 跟踪,可复现) |

### 3.3 关键超越点(C13 – C16,新增)

**这 4 条是 Incepedia 相对 Cosmopedia v2 的核心差异化**。

#### C13 · Instruction reversal(Phi-4 方法)
- 对每个 web 段落,让 generator 反推"这段话在回答什么问题",生成 `(question, answer)` 对
- 高质量的可以当 instruction-style pretrain 样本,也可输入到 textbook generator 做"Q&A 章节"
- 来源:Phi-4 技术报告明确列为核心方法
- 占 Incepedia 总量 ~10–15%

#### C14 · 风格反单调化(anti-monotone)
- 维护 `configs/disallowed_openings.yaml`:持续扩充的"禁用开头短语"列表
- Cosmopedia 著名案例:"Once upon a time"、"A few years back"、"In recent years"、"It was a sunny morning"…
- 每批生成前把这个列表注入 system prompt:*"Do not begin with any of: [list]"*
- 每 1000 条采样 50 条,统计开头短语 top-20,发现新的单调模式 → 自动加入禁用清单
- 成本极低(每 1000 条 <$0.1 审计),但对防塌方至关重要

#### C15 · Web rephrasing 主路径(Nemotron / WRAP / Phi-4 路线)
- **除了** "textbook from web seed"(Cosmopedia 式重建),新增 **"rephrase this web text into <style>"**(直接改写):
  - `rephrase_as_textbook_chapter`
  - `rephrase_as_qa_dialogue`
  - `rephrase_as_socratic_discussion`
  - `rephrase_as_lecture_notes`
- 这个路径**保留事实真实性**(因为是改写真实文本),大幅降低幻觉
- Phi-4 / Nemotron-Synth / WRAP 均有证据表明 rephrasing 质量 ≥ 凭空生成
- **和 Cosmopedia 式"seed 只当启发、内容自由发挥"的最大区别**
- 占 Incepedia 总量 ~40–50%(**主力路径**)

#### C16 · Synthetic 贯穿 pretrain 全程(Phi-4 路线)
- 这不是生成器决策,是**训练配方**决策
- Phi-4 报告:12 epochs synthetic > 1 epoch more unique web
- 对我们的含义:在"独立 corpus"场景(Track 1),Incepedia 应作为 100% 主力而非仅 cooldown;在"配料"场景(Track 2),评测其作为 decay 调料的 delta 增益
- **这恰好对齐两条 ablation 轨道**(ADR 0004)

---

## 4 · Incepedia 生成 pipeline(待实现)

```
 topics.yaml   ──┐
 personas.yaml ──┼→ PromptAssembler ─→ prompt(seed + persona + difficulty + audience + anti-repetition)
 web seeds     ──┤                          │
 RAG retriever ──┘                          │
                                            ▼
                       Router (by task tier, weighted, configs/generators.yaml)
                            │
                   midtier_bulk · frontier_reasoning · critic
                            │
                            ▼
                OpenRouter Async Client(httpx + aiolimiter + tenacity)
                            │
                            ▼
           Raw completion + metadata
                            │
                ┌───────────┴──────────┐
                ▼                      ▼
          (optional) Critic round   pass-through
           (C10: STEM/code only)
                │
                └───────────┬──────────┘
                            ▼
         Boilerplate cleanup + sanity filter
         (length / language / refusal / opening pattern — C14)
                            │
                            ▼
            Parquet shard writer (10M tokens / shard)
                            │
                            ▼
           data/raw_generations/batch_{YYYYMMDD}_{tag}/
                            │
                            ▼
         Rolling stats: cost / speed / diversity / opening-pattern audit
                            │
                            ▼
                Event: sync_to_nas.sh gen <batch_id>
```

---

## 5 · 三阶段路线图

| 阶段 | 时长估计 | 目标 | 成功标准 |
|---|---|---|---|
| **P1 · 复现** | 2–3 周 | 复现 Cosmopedia v2 在 Track 2 的 seasoning 基线 + Cosmopedia v1 的 Track 1 基线 | 分数对齐公开数字 ±0.5pp |
| **P2 · PoC** | 4–6 周 | 3B token Incepedia v0.1,两轨均持平或超越 Cosmopedia | 最小闭环:生成 → 训练 → 评测 → 对比 |
| **P3 · 迭代 / 扩容** | 2–3 月 | 10B–30B token Incepedia v1.0 | 两轨显著超越 Cosmopedia(v1 standalone + v2 seasoning),逼近 SmolLM2 |

每阶段结束写 retrospective + ADR。

---

## 6 · 关键决策(ADR 索引)

详见 `docs/decisions/README.md`。当前已定:

- **0001** · 三层存储防护(本地 + NAS + git)
- **0002** · Ablation 作为唯一质量指标(被 0004 部分扩充)
- **0003** · 多档生成器路由
- **0004** · 双轨自适应评测协议
- **0005** · 项目 scope:独立 corpus + decay 调料双场景
- **0006** · Evaluation stack policy(lighteval 跟随 latest)
- **0007** · 双协议架构(Llama2-1.82B 锚 + Qwen3-1.7B 主力)

---

## 7 · 命名与数据契约

- 实验命名:`exp_{YYYYMMDD}_{slug}_seed{N}`,`config.yaml` 必须带 `track: 1|2` 字段
- 数据集命名:`incepedia_v{major}.{minor}`
- Config 规范:见 `AGENTS.md § Config file format`
- INDEX.parquet 字段:见 `scripts/index_experiment.py` 顶部 `SCHEMA_COLS`

---

## 8 · 开放问题(持续跟进)

- [ ] 难度分层的粒度(5 档 vs 3 档?)—— P2 ablation 回答
- [ ] Instruction reversal 是否作为独立 split 训练 —— 待 Phi-4 细节进一步调研
- [ ] Curriculum ordering:cooldown 阶段纯 Incepedia vs 混 web 比例 —— P3 ablation
- [ ] Embedding 去重的 threshold(0.90 vs 0.92 vs 0.95)—— P2 检验
- [ ] 若 Track 1 和 Track 2 方向在某变量上冲突(一涨一降),优先哪个 —— 待实际出现再 ADR

---

© 2026 · Incepedia contributors · Apache-2.0
