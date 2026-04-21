---
title: "Incepedia · 项目思路汇报"
author: "bill-lee-mk"
date: "2026-04-21"
---

# Incepedia · 项目思路汇报

---

# 1. 当前 Cosmopedia 的局限性

1. **Mixtral 的事实幻觉**——Khan/AutoMathText 部分数学/科学事实有错。
2. **生成器是天花板**——风格、深度、推理链都受 Mixtral‑Instruct 限制(尽管 v2 验证更强模型也没救,说明瓶颈是**提示+种子**)。
3. **不能单独撑起大规模预训练**——SmolLM 阶段已经把 Cosmopedia 从"主菜"降为 ~10% 的提味剂。
4. **同质化风险**:同一生成器 + 同一种风格指令,即使 dedup 通过,**潜在分布重复**仍然存在(论文中 MinHash 只能查表层重复)。
5. **Benchmark 漏污染**:10‑gram + 0.5 ratio 是宽松阈值,**改述型污染**抓不到。
6. **English‑only**。
7. **音乐性、长链条推理、agent 场景几乎缺席**。
8. **审美/风格偏 Mixtral**——下游模型容易带"AI 味"。

---

# 2. Incepedia 项目愿景与终极目标

## 2.1 一句话定义

**构造一个合成预训练数据集 Incepedia,在相同 1.82B Llama2 ablation 协议下,同时以"独立 corpus"和"decay 调料"两种身份,下游 benchmark 分数超越 Cosmopedia。**

## 2.2 愿景(长 / 中 / 短期)

- **长期(12+ 个月)**:让 Incepedia 成为开源社区继 Cosmopedia 之后的首选合成预训练数据集,推动"合成数据是否为大模型训练新主力"议题的可验证进展。
- **中期(3–6 个月)**:发布 Incepedia v1.0(10B+ tokens),在 standalone 和 seasoning 两种场景下均显著超越 Cosmopedia v1/v2。
- **短期(3 个月,即本周期)**:完成 P1 基线复现 + P2 PoC(Incepedia v0.1,3B tokens)+ P3 迭代到 v1.0。

## 2.3 三阶段路线图

| 阶段 | 目标 | 成功标准 |
|---|---|---|
| **P1 · 复现** | 复现 Cosmopedia v2 在 Protocol A(Llama2-1.82B)和 Protocol B(Qwen3-1.7B)上的基线 | Protocol A 分数对齐 SmolLM/FineWeb 公开数字 ±0.5pp;Protocol B 自建基线 |
| **P2 · PoC** | 3B token Incepedia v0.1,Protocol B 两轨跑通 | 两轨均持平或超越 Cosmopedia |
| **P3 · 迭代+扩容** | 10B–30B token Incepedia v1.0,Protocol B 主跑;P3 末期 Protocol A 100B 终验 | 两架构两轨均显著超越,逼近 SmolLM2 |

## 2.4 双协议架构(ADR 0007)

| 协议 | 架构 | 作用 | 项目频率 |
|---|---|---|---|
| **Protocol A** | Llama2-1.82B(全注意力 / RoPE θ=1e4) | 外部锚 — 对齐 SmolLM/FineWeb 公开数字,验证 pipeline 正确性 | 2 次 |
| **Protocol B** | Qwen3-style 1.7B(GQA 8 KV / RoPE θ=1e6 / QKV bias) | 工作架构 — 承担所有 Incepedia 版本演进 + 配比 ablation | ~40 次 |

**唯一变量是架构**:tokenizer / 数据 / lr / batch / seeds / 评测脚本两协议完全一致。

---

# 3. Incepedia 执行方案总体战略

> 总体战略:**复现 → 替换 → 闭环**,科学方法论 = "只动一个变量,看分数变化"。

---

# 4. 第一步 · 百分百复现 Cosmopedia 的训练‑评测管线

目标:**搭一个"无新意但 100% 一致"的 reference pipeline**,后续所有实验都跑在它之上,排除非数据因素。

## 4.1 锁定 reference 配方(强烈建议直接对齐 SmolLM/Cosmo‑v2 的 ablation 设置,而不是 v1 的 6‑epoch 训练)

| 项 | 选择 | 理由 |
|---|---|---|
| 训练框架 | `nanotron` | Cosmopedia/SmolLM 官方栈,checkpoint 可分叉 |
| 模型 | Llama2‑1.82B(等同 FineWeb ablation 模型) | 行业事实基线;cosmo‑1b 同档;成本合理 |
| Tokenizer | **`mistralai/Mistral‑7B‑v0.1`** 的 tokenizer(v1)或 SmolLM 自有 tokenizer(v2) | 与目标 baseline 完全一致 |
| Token 量 | 30B(快速 ablation)+ 100B(中验证)+ 350B(终验) | 三档分别对应 ~$几k/几万/几十万的算力预算 |
| Seq len / GBS | 2048 / 1.3M tokens | 与 Cosmo‑1B、SmolLM 一致 |
| LR | 3e‑4 cosine(完全复现 v1)+ 备选 trapezoidal(v2 风格,可在 cooldown 阶段做配方实验) | trapezoidal 更适合 ablation,不需要重训整段 |
| Seeds | 至少 2 个(42, 1337) | 行业标准 |
| Eval | `lighteval` + `cosmopedia/evaluation/lighteval_tasks.py` 完全照搬 | 评测脚本零修改 |

## 4.2 落地 checklist(按顺序)

1. `git clone` 三件套到 `Incepedia/` 仓库:`nanotron`、`lighteval`、`huggingface/cosmopedia`(取 prompts/eval/decontamination 目录)。
2. 复刻数据预处理:用 `datatrove` 做 tokenization → `nanotron` shard 格式。
3. **冒烟基线 1**:用 **公开的 `cosmopedia‑v2`** 全量训 30B tokens × 2 seeds → 在 lighteval 早信号组上得到 reference 分数,与 SmolLM blog 中 Mixture11 的数字对齐(±0.5pp 以内才算对得上)。
4. **冒烟基线 2**:同样设置训 `FineWeb-Edu` 30B → 对齐 `HuggingFaceFW/ablation-model-fineweb-edu` 数字。
   - 这两个对齐过了,reference 管线才能宣告"可信"。
5. **Checkpoint 策略**:`nanotron` 默认每 N 步存一份;**强制每 2B token 存一次,并在 6/12/18/24/30B 五个节点上各做 lighteval**——这就是后续所有 Incepedia 分叉实验的"早期信号 dashboard"。
6. **去污染**:用 `cosmopedia/decontamination` 跑 10‑gram + difflib,**针对最终所有评测集**(含 GSM8k、MMLU‑Pro、HumanEval 的输入)。
7. **WandB / MLflow**:loss、grad norm、token throughput、每 2B token 的所有 lighteval 分数都进同一面板,后面方便对比。

> **关于"checkpoint 分叉"诉求的关键提醒**:`nanotron` 存的是优化器+权重的 full state,可以 resume。**所以一定要用 trapezoidal+cooldown,而不是 cosine**——cosine 一旦动末端 LR 就回不了头;trapezoidal 的 cooldown 是单独阶段,可以从同一个"主干 checkpoint"上**多次分叉做不同 cooldown 实验**(SmolLM 团队就是这么省算力的,见 Hägele et al. 2024)。

---

# 5. 第二步 · 生成 Incepedia

## 5.1 大方向(吸取 Cosmopedia 的教训 + 2025‑2026 最新经验)

把 Cosmopedia v2 当骨架,在 4 个维度做超越:

| 维度 | Cosmopedia v2 | **Incepedia 增强** |
|---|---|---|
| 生成器多样性 | 单一 Mixtral‑8x7B‑Instruct | **多模型混合**(GPT‑4o‑mini / Claude‑3.5‑Haiku / DeepSeek‑V3 / Llama‑3.3‑70B / Qwen2.5‑72B / Mistral‑Large 等,通过 OpenRouter 路由),且**每个 topic 至少 2 个生成器交叉**,降低单模偏置 |
| 主题表 | 34k BISAC topics | BISAC 34k + **Wikipedia category tree(top‑level + 2 层子类,~50k)** + **arXiv subject(STEM 强化)** + **MMLU 57 subject 的同主题反向覆盖**(注意这不是污染,是"主题对齐") |
| Seed 来源 | FineWeb 全文检索 | **FineWeb‑Edu(higher quality)+ Wikipedia + arXiv abstracts + Stack‑Edu + 已发布教科书目录**;每个 prompt **强制带 1 个 seed 段落**(避免凭空生成 → 减幻觉) |
| 内容形态 | textbook / blog / wikiHow / story | + **"问答对话(Socratic)" + "课后习题 + 解答" + "代码教程" + "概念辩论(multi‑agent)" + "instruction reversal"(给答案反推问题)**——直接借 Phi‑4 的方法 |

## 5.2 质量控制与去重

- **MinHash dedup**(`datatrove`,与 Cosmopedia 同参数:n‑gram=5,阈值 0.7)
- **Embedding‑level near‑dup**(BGE‑small / nomic‑embed,阈值 cos>0.92):**Cosmopedia 没做,这是关键超越点之一**
- **Quality classifier**:复用 `HuggingFaceFW/fineweb-edu-classifier` 给每条打分,只保留 score≥3 → 这是 FineWeb‑Edu 提分的核心 trick
- **Decontamination**:沿用 cosmopedia 的 10‑gram + 0.5 ratio,**新增对 GSM8k/MATH/HumanEval/MBPP 的覆盖**

---

# 6. 关于"量级是不是定 10B"

1. **解耦"数据集 size"和"训练消耗 token 数"**——这是评估合成数据最容易踩的坑。
   - 一个 5B 的高密度数据集,跑 6 epoch ≈ 30B 训练 token,可能比 30B 数据集跑 1 epoch 更优(Phi‑4 报告:**12 epoch 合成数据 > 1 epoch 更多 unique web token**)。
   - 因此"截断实验"应该是**两个轴的网格**:`{数据集 size: 1, 3, 5, 10, 15, 30B} × {训练 token: 30B, 100B}`,而不是单轴。

2. **定一个"涌现监测器"**:每次 ablation 都画 *训练 token vs 各 benchmark*。**涌现的标志是某个 benchmark 在某 token 数突然非线性跃升**。一旦观察到,**马上在那个区间加密采样**(比如 25–35B 之间每 2B 存 ckpt)。

3. **优先级建议**:
   - 第一阶段(2–4 周):先做到 **3B 高质量 Incepedia**,跑 30B token ablation,**和 Cosmopedia‑v2 同设置打成平手或更好**——这是 PoC,先证明"方法论有效"。
   - 第二阶段(1–2 月):扩到 10B + 多领域配比 ablation,目标是 **MMLU/GSM8k 显著超过 Cosmopedia‑v2**。
   - 第三阶段:扩到 30B + 与 FineWeb‑Edu 混合,目标超过 **SmolLM2 公开数字**。

4. **量级的"科学截断点"**怎么找:训练 loss 在 valid set 上的 **per‑domain perplexity** 曲线 + benchmark 分数曲线,**两者同时 plateau** 的点就是该尺寸的"够了"信号。

---

# 7. 什么是 Ablation Training?怎么用到 Incepedia 生成里

## 7.1 严格定义

**Ablation Training = "控制变量法"在数据集/训练 pipeline 上的应用。** 固定除一个变量外的所有东西(模型架构、初始化 seed、tokenizer、token 总数、超参、评测脚本),**只动要研究的那一个变量**,看下游模型 benchmark 的差异。

## 7.2 它和"训完一个模型测一下"的本质区别

- 单次训练得到的是**绝对分数**——可能是数据好,可能是 lr 好,可能是 seed 走运
- ablation 得到的是**因果增量**——"加了 persona pool,MMLU 涨了 0.6pp,可信度 95%"

## 7.3 用到 Incepedia 生成的具体场景

10 个**当前最该跑的 ablation 实验**(每个 30B token × 2 seed,8×H100 上 1‑2 天一组):

| 编号 | 变量 | 对照组 A | 对照组 B | 想回答的问题 |
|---|---|---|---|---|
| A1 | seed grounding | 无 seed,纯 topic prompt | 强制注入 web seed 段落 | 幻觉/分数 trade‑off |
| A2 | 生成器档位 | 100% DeepSeek‑V3 | 100% Claude‑3.5‑Sonnet | 高档生成器值不值 |
| A3 | 生成器混合 | 单生成器 | 4 模型混合 | 多样性是否兑现为分数 |
| A4 | persona pool | 无 persona | 注入 10k persona | persona 边际收益 |
| A5 | 难度分层 | 全 college 档 | 5 档均匀 | 课程结构是否有效 |
| A6 | 自批改 | 无 critic | Claude critic + 重写 | 多 agent 是否提升 |
| A7 | RAG 注入 | 无 RAG | Wikipedia top‑3 RAG | 真实性是否变现 |
| A8 | 结构模板 | 仅 textbook | 12 模板均匀 | 结构多样性收益 |
| A9 | 配比:STEM 占比 | 30% | 60% | STEM 倾斜对 MMLU‑STEM 影响 |
| A10 | 配比:故事/常识占比 | 5% | 20% | 常识倾斜对 HellaSwag 影响 |

## 7.4 两个降本关键技巧

1. **Proxy ablation(代理消融)**:**先用 360M / 30B token 做粗筛**,1‑2 个候选再用 1.82B / 30B 验证,1‑2 个胜出再用 1.82B / 100B 做最终验证。三层漏斗,算力开销降 5‑10 倍。

2. **Trapezoidal cooldown 分叉**:训完一个 1.82B 主干 ckpt 后,**在 cooldown 阶段(只占总训练 20%)**用不同数据配比分叉跑多个实验,共享主干算力。这是 SmolLM 团队的官方降本姿势(参考 Hägele et al. 2024)。

---

# 8. 当前在做的:复现 Cosmopedia v2 reference baseline

> **双协议架构**(ADR 0007):每个 Track 的训练在两个架构上分别跑——Protocol A(Llama2-1.82B)只作外部锚 2 次,Protocol B(Qwen3-1.7B)是日常工作架构,承担所有 Incepedia 版本演进。Tokenizer 两协议同用 Mistral-7B-v0.1,确保唯一变量是架构。

## 8.1 Track 1 · Standalone(独立 corpus 能力)

| 项 | 值 |
|---|---|
| Protocol A 协议 | 1.82B Llama2(全注意力 / RoPE θ=1e4 / 无 bias)从头训 × 30B tokens × 2 seeds |
| Protocol B 协议 | 1.7B Qwen3-style(GQA 8 KV / RoPE θ=1e6 / QKV bias)从头训 × 30B tokens × 2 seeds |
| 单次墙钟(A / B) | ~2.1 天 / ~2.0 天(8×H100)× 2 seeds = ~4.2 / ~4.0 天 |
| Milestone scale-up | 1.82B / 1.7B × 100B × 1 seed = ~7 / ~6.5 天 |
| 适用问题 | "数据集 X 作为独立 pretraining corpus,和数据集 Y 相比如何?" |
| 论文血脉 | Cosmopedia‑v1 / Cosmo‑1B / FineWeb ablation 协议的直接继承(Protocol A);Qwen3 论文(Protocol B) |

## 8.2 Track 2 · Seasoning(decay 调料能力)

| 项 | 值 |
|---|---|
| 协议 | **共享 backbone + cooldown‑fork** |
| 协议 · backbone | 1.82B Llama2 × 20B tokens FineWeb‑Edu(trapezoidal LR,warmup+stable 阶段结束保存 ckpt)—— **一次性投入,之后所有 fork 共用** |
| 协议 · fork | 从 backbone ckpt 分叉,LR 从 peak 线性衰减到 0,喂 6B tokens 的测试数据集 |
| 单次墙钟 | backbone 1.4 天(一次);每 fork ~10 h(1 seed)/ ~20 h(2 seeds) |
| 适用问题 | "在 SmolLM2 式 decay 阶段,用数据集 X 精调,和用 Y 精调相比如何?" |
| 论文血脉 | SmolLM2 论文评判 Cosmopedia v2 的真实协议 |

## 8.3 当前阶段的具体实验单

**Track 1 · Protocol A(Llama2-1.82B 外部锚,仅 2 次):**

| exp_id | 协议 | seeds | 墙钟 | 目的 |
|---|---|---|---|---|
| `exp_ref_cosmopedia_v2_seed42` | Llama2-1.82B × 30B × 纯 Cosmopedia v2 | 1 | 2.1 天 | 外部锚 seed1 |
| `exp_ref_cosmopedia_v2_seed1337` | Llama2-1.82B × 30B × 纯 Cosmopedia v2 | 1 | 2.1 天 | 外部锚 seed2,与 seed42 平均 = 对齐 SmolLM |

**Track 1 · Protocol B(Qwen3-1.7B 工作架构,承担所有版本演进):**

| exp_id | 协议 | seeds | 墙钟 | 目的 |
|---|---|---|---|---|
| `exp_ref_cosmopedia_v2_qwen3_seed42` | Qwen3-1.7B × 30B × 纯 Cosmopedia v2 | 1 | 2.0 天 | Protocol B 基线 seed1 |
| `exp_ref_cosmopedia_v2_qwen3_seed1337` | Qwen3-1.7B × 30B × 纯 Cosmopedia v2 | 1 | 2.0 天 | Protocol B 基线 seed2 |
| `exp_inc_v01_qwen3_seed{42,1337}` | Qwen3-1.7B × 30B × Incepedia v0.1 | 2 | 4.0 天 | 首次对标 |
| `exp_inc_v02_qwen3_*` | Qwen3-1.7B × 30B × Incepedia v0.2 | 2 | 4.0 天 | 第二轮迭代 |
| `exp_inc_v03_qwen3_*` | Qwen3-1.7B × 30B × Incepedia v0.3 | 2 | 4.0 天 | 第三轮 |
| `exp_inc_v10_qwen3_*` | Qwen3-1.7B × 30B × Incepedia v1.0 | 2 | 4.0 天 | 最终版本 |
| `exp_inc_v10_qwen3_scale` | Qwen3-1.7B × 100B × Incepedia v1.0 | 1 | 6.5 天 | 放大验收 |

**Track 2 · Protocol B(Seasoning,共享 backbone + cooldown forks):**

| exp_id | 协议 | seeds | 墙钟 | 目的 |
|---|---|---|---|---|
| `backbone_fineweb_edu_qwen3` | Qwen3-1.7B × 20B 纯 FineWeb-Edu | 1 | 1.3 天 | **一次性**,之后所有 fork 共用 |
| `fork_cosmopedia_v2_qwen3` | cooldown 6B × Cosmopedia v2 | 2 | 18 h | Track 2 基线 |
| `fork_inc_v01_qwen3` | cooldown 6B × Incepedia v0.1 | 2 | 18 h | 首次配料对比 |
| `fork_*_variant_*` × ~30 | cooldown 6B × 各种变量 | 1 | 9 h each | C1–C16 每条 1–2 次 + 配比扫描 |

**项目训练总墙钟**:Protocol A ~4.2 天 + Protocol B ~40 天 ≈ **~44 天**(8×H100,3 个月日历内有充裕余量)

---

# 9. 修正后的 Step C 决策清单(C1 – C16)

## 9.1 基础策略(C1 – C4)

| # | 决策 | 作用 |
|---|---|---|
| **C1** | **强制 seed-grounding** — 每条 prompt 必带 Wikipedia / arXiv / FineWeb-Edu 段落 + 反幻觉指令 | 降低 Mixtral 时代的幻觉(Cosmopedia 公认痛点) |
| **C2** | **Persona 注入** — 从 `configs/personas.yaml`(P2 扩到 ~10k)按权重抽 | 反 L3 同质化(风格单调) |
| **C3** | **5 档难度** — elementary / middle / high_school / college / expert | 支撑课程式实验(C16) |
| **C4** | **12+ 结构模板** — textbook / blog / wikihow / story / Q&A / Socratic / case_study / debate / lecture_notes / glossary / cheatsheet / worked_example / misconception_correction | 反 L1 同质化(模板单一) |

## 9.2 生成器 & 生产线(C5 – C12)

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

## 9.3 关键超越点(C13 – C16,新增)

**这 4 条是 Incepedia 相对 Cosmopedia v2 的核心差异化**。

### C13 · Instruction reversal(Phi-4 方法)
- 对每个 web 段落,让 generator 反推"这段话在回答什么问题",生成 `(question, answer)` 对
- 高质量的可以当 instruction-style pretrain 样本,也可输入到 textbook generator 做"Q&A 章节"
- 来源:Phi-4 技术报告明确列为核心方法
- 占 Incepedia 总量 ~10–15%

### C14 · 风格反单调化(anti-monotone)
- 维护 `configs/disallowed_openings.yaml`:持续扩充的"禁用开头短语"列表
- Cosmopedia 著名案例:"Once upon a time"、"A few years back"、"In recent years"、"It was a sunny morning"…
- 每批生成前把这个列表注入 system prompt:*"Do not begin with any of: [list]"*
- 每 1000 条采样 50 条,统计开头短语 top-20,发现新的单调模式 → 自动加入禁用清单
- 成本极低(每 1000 条 <$0.1 审计),但对防塌方至关重要

### C15 · Web rephrasing 主路径(Nemotron / WRAP / Phi-4 路线)
- **除了** "textbook from web seed"(Cosmopedia 式重建),新增 **"rephrase this web text into <style>"**(直接改写):
  - `rephrase_as_textbook_chapter`
  - `rephrase_as_qa_dialogue`
  - `rephrase_as_socratic_discussion`
  - `rephrase_as_lecture_notes`
- 这个路径**保留事实真实性**(因为是改写真实文本),大幅降低幻觉
- Phi-4 / Nemotron-Synth / WRAP 均有证据表明 rephrasing 质量 ≥ 凭空生成
- **和 Cosmopedia 式"seed 只当启发、内容自由发挥"的最大区别**
- 占 Incepedia 总量 ~40–50%(**主力路径**)

### C16 · Synthetic 贯穿 pretrain 全程(Phi-4 路线)
- 这不是生成器决策,是**训练配方**决策
- Phi-4 报告:12 epochs synthetic > 1 epoch more unique web
- 含义:在"独立 corpus"场景(Track 1),Incepedia 应作为 100% 主力而非仅 cooldown;在"配料"场景(Track 2),评测其作为 decay 调料的 delta 增益
- **这恰好对齐两条 ablation 轨道**(ADR 0004)

---

# 10. 为什么两条 track 互补,一个都不能少

| 问题 | 只有 Track 2(cooldown-fork)能回答吗? | 只有 Track 1(full)能回答吗? |
|---|---|---|
| "Incepedia 作为独立 corpus 比 Cosmopedia v2 强吗?" | ❌ 不能,答的是配料场景 | ✅ 能 |
| "Incepedia 作为 SmolLM2 式 decay 调料更强吗?" | ✅ 能,这就是 SmolLM2 官方协议 | ⚠️ 间接(只能测纯 corpus,不测配料) |
| "改了 persona pool,分数有变化吗?" | ✅ 能,快 | ✅ 能,但慢 5× |
| "Incepedia v0.X 相对 v0.Y 进步了多少?" | ✅ 能 | ✅ 能 |
| "独立 corpus 和调料能力,哪个方向更有前景?" | ❌ 缺独立数据 | ❌ 缺配料数据 |

→ **Track 1 + Track 2 一起 = 完整覆盖**(独立 corpus 与 decay 调料双场景)。少任何一个都不对等。

## 10.1 协议选择规则(自适应,非配额)

每个 ablation 开始前问:"这个实验想答的是独立能力还是调料能力?"

- 独立能力 / 版本级 milestone → **Track 1**
- 参数微调 / 配比扫描 / 单变量 ablation → **Track 2**
- **不设前置配额**,事后盘点比例作为项目记录
- 预估自然分布:Track 1 ~15%、Track 2 ~85%(约 6 + 33 ≈ 40 次 ablation)
