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

**易混淆点:Track 1 有 stable=24B,为何共享 backbone 只训 20B(0.5B warmup + 19.5B stable),而不是 24B?**

两轨回答的是**不同科学问题**,分段数字**不是**同一套 LR 日程的简单切片,不能从「cooldown 最多 6B」反推 backbone 必须等于 30B−6B=24B。

- **Track 1(standalone)**:warmup 0.5B + **在目标语料(如 Cosmopedia)上** stable 24B + cooldown 5.5B = **30B**,测的是「纯该语料从头训满这一预算」;其中 24B 是同一条 run 里、**在该语料上保持峰值 LR 的平台段**。
- **Track 2(seasoning)**:backbone 只在 **FineWeb-Edu** 上跑 warmup 0.5B + stable 19.5B = **20B**,末端 **不设** LR 衰减(`cooldown_tokens=0`),在峰值 LR 结束处存盘;之后的 **6B** 在 **cooldown-fork** 上单独跑,才是「调料语料 + 线性衰减」阶段。
- **设计意图**:backbone 的 20B 与 Track 1 的 24B **不是**「少训了 4B stable」的疏忽,而是 **角色不同** —— 前者是 web 主干上的前缀终点,后者是合成语料 standalone 全程里的高 LR 平台。两轨用 **20B+6B=26B** 与 **30B** 做**总 token 量级可比**,而非要求 stable 段 token 数一一相等。

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
- Phi-4 报告:12 epochs synthetic > 1 epoch more unique web(对照同等训练 token 预算,4→8 epoch 显著增,8→12 平台,>12 倒退)
- 对我们的含义:在"独立 corpus"场景(Track 1),Incepedia 应作为 100% 主力而非仅 cooldown;在"配料"场景(Track 2),评测其作为 decay 调料的 delta 增益
- **这恰好对齐两条 ablation 轨道**(ADR 0004)
- **v0.1 epoch 选择**:10B uniq × 30B 训练 = **3 epoch**,处于 Phi-4 观察的"显著增"段,保守远离过拟合拐点。**E3 控制实验**用 Cosmopedia v2 随机 3B subset × 10 epoch,把 epoch 效应与数据质量效应拆开

### 3.4 P2 质量优先升级项(①–⑨)—— 在 C 基础上调参

C1–C16 确定"做什么",下面 9 项确定"P2 阶段做到什么规模 / 强度"。详见 `docs/codenames-cheatsheet.md §2.4`。

| # | 升级 | 基础策略 | P2 目标值 | 状态 |
|---|------|----------|-----------|------|
| **①** | uniq token 规模 | C16 | 3B → **10B**(epoch 从 10 降到 3,大幅降 memorization 混淆) | ✅ |
| **②** | frontier tier 占比 | ADR 0003 | 16% → **50%**(rephrase_* 全部升档,midtier 只留 wikihow/story) | ✅ |
| **③** | Best-of-N + critic pick | 新增 | midtier N=1 / frontier N=2 / STEM-expert N=3 | ✅ |
| **④** | critic 覆盖与轮数 | C10 | 15% × 1 轮 → **50% × 最多 2 轮**,score<4 强制改到 ≥4 | ✅ |
| **⑤** | seed 来源丰富度 | C1 | FineWeb-Edu + Wiki + arXiv abstract → **+ OpenStax + Stack-Edu + arXiv full-text** | ✅ |
| **⑥** | persona pool 规模 | C2 | 2k(PersonaHub 导入)→ **10k**(LLM 过滤均匀子集) | ✅ |
| **⑦** | 结构模板数量 | C4 | 12 → **20**(加 glossary / cheatsheet / misconception_correction / worked_example / case_study / debate / flashcard / concept_map) | ✅ |
| ⑧ | Multi-agent debate | 新增 | expert 档 20% 走 2-agent 辩论 → critic 裁决 | ⏳ v0.2 |
| ⑨ | 全局滚动 embed 索引 | C11 衍生 | FAISS 1M 滑窗实时拒 cos>0.95 | ⏳ v0.2 |

---

## 4 · Incepedia 生成 pipeline · P2 规范版

完整管线见 `docs/codenames-cheatsheet.md §4` 的剖面图。下面是分段概述。

### 4.1 一次性数据准备(W1)

- **seed 层**(⑤):`scripts/prepare_seeds.py` 从本地 FineWeb-Edu dedup(72 GB)+ HF Wikipedia + HF arXiv full-text + OpenStax CC + Stack-Edu 采样 300–800 token 段落,按 edu-score 加权,落到 `data/seeds/*.parquet`
- **topic 层**:`scripts/build_topic_tree.py` 合并 BISAC 34k + Wikipedia category 2 层(~50k)+ arXiv subject + **MMLU 57 subject 反向锚点**(不使用原题文,只对齐主题),产 `configs/topics.yaml` ~90k topic
- **persona 层**(⑥):`scripts/import_persona_hub.py` 从 HF `tencent-ailab/persona-hub` 200k 里 LLM 过滤到 **10k 均匀子集**,落 `configs/personas.yaml`
- **配方层**:`configs/batches/v0.1_pilot.yaml`(C12 批次声明)+ `configs/generators.yaml`(ADR 0003)+ `configs/disallowed_openings.yaml`(C14,手工种 30 条,W3–W5 自动扩)

### 4.2 每条样本的实时流水线(W2–W5)

**9 维度 PromptAssembler**:`seed_source × seed_text × topic × persona × difficulty × structure × path × generator_tier × anti_mono_constraints`

```
PromptAssembler
    → Router(ADR 0003,②:frontier 占比 50%)
    → OpenRouter AsyncClient(C5/C6,双 key)
    → Best-of-N(③:midtier=1 / frontier=2 / STEM-expert=3)
    → Critic loop(C10/④:50% 覆盖 × 最多 2 轮,score<4 必改)
    → Post-filter(语言 / 长度 / 拒答 / 首句 ∉ disallowed_openings C14)
    → Quality classifier(fineweb-edu-classifier score ≥ 3)
    → 批内 embed dup check(BGE-small 10k 滚动,cos>0.95 拒)
    → Writer(C7:1000 条 Parquet + 原子重命名)
    → Audit(C8 cost / 成功率 / dup;C14 首句 top-20 回写;C9 偏差 >20% 暂停)
    → data/raw_generations/batch_<id>/*.parquet
    → sync_to_nas.sh gen <batch_id>(事件驱动)
```

### 4.3 批量后处理 & tokenize(W6)

`scripts/dedup_and_filter.py` 四阶段:

1. MinHash dedup(n-gram=5,threshold=0.7,datatrove)—— 对齐 Cosmopedia 参数
2. Embed dedup(BGE-small-en-v1.5,cos>0.92)—— 超越点
3. Classifier filter 复核(edu-score ≥ 3)
4. Decontamination(10-gram + 0.5 ratio,覆盖 MMLU / MMLU-Pro / GSM8k / MATH / HumanEval / MBPP)

→ `scripts/tokenize_dataset.py` → `data/datasets/incepedia_v01_qwen/*.ds`(datatrove → nanotron shard)

### 4.4 下游训练单元(W6 收尾)

- **E2**(主打,G4 门):`exp_inc_v01_qwen3_seed{42,1337}` × Track 1 × 30B × 2 seeds
- **E3**(epoch 控制):`exp_ctrl_cosmo_v2_subset3B_qwen3_seed42` × Track 1 × 30B × 1 seed(Cosmopedia v2 随机 3B subset × 10 epoch)
- **E5**(epoch 扫描,事后可选):`exp_inc_v01_qwen3_ep{1,3,10}`
- **A1 / A2 / A5 / A6 / A8 / A10**(6 条重生成 ablation):每条各生成 2B uniq,挂 Track 2 cooldown-fork,~$40–60k + 15 天 GPU

详细代号映射与 G1–G4 门控规则见 `docs/codenames-cheatsheet.md`。

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
