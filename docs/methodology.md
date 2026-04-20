# Incepedia · 方法论

> 本文档是项目的"北极星":所有实验决策最终都回到这里对齐。
> 语言:中文为主,核心 ADR 另存于 `docs/decisions/`。

---

## 1 · 问题定义

**做什么**:构造一个合成预训练数据集 `Incepedia`,在 1.82B Llama2 / 30B token 的标准 ablation 协议下,下游 early‑signal benchmark 分数**持续超越 Cosmopedia v2**。

**为什么做**:
- Cosmopedia 是当前最可验证的开源合成数据范式,但其 2024 年设计存在明显 gap(单生成器、粗聚类、无 embedding 去重、10‑gram 浅去污染、无多 agent)。
- 2025–2026 路线已迁到"web‑grounded synthetic + multi‑agent + 难度分层",**有机会以相同算力做出更强数据**。

**不做什么**:
- 不追求单纯 token 规模超越。**信息密度 × 多样性 > token 总量**。
- 不做通用 SFT / RLHF 数据,只聚焦 pretraining corpus。

---

## 2 · 评判数据质量的唯一标尺:Ablation Training

### 2.1 操作化定义

> **数据质量 := 在 1.82B Llama2 × 30B token × 固定超参 × 2 seeds 下,lighteval early‑signal benchmark 的平均分。**

任何"数据看起来高质量 / 信息密度高 / 可读性好"等主观判断都不算数据质量证据,**必须通过 ablation 训练验证**。

### 2.2 Reference 协议(完全对齐 FineWeb / Cosmopedia / SmolLM)

| 项 | 值 |
|---|---|
| 架构 | Llama2 ~1.82B |
| 序列长度 | 2048 |
| Global batch | 1.31M tokens |
| 训练 token(快速 ablation) | 30B |
| 训练 token(中验证) | 100B |
| 训练 token(终验) | 350B |
| LR | 3e-4 cosine 或 trapezoidal + 20% cooldown |
| Seeds | 2(42, 1337),分数取平均 |
| 评测 | `lighteval` + 从 cosmopedia 复刻的 task 定义 |

### 2.3 Early‑signal benchmark 套件

**常识 (CSR, 0-shot)**:HellaSwag / Winogrande / PIQA / SIQA / OpenBookQA / ARC‑Easy / ARC‑Challenge / CommonsenseQA / BoolQ

**知识 / 推理**:MMLU(cloze + MC,57 subset)、MMLU‑Pro(cloze + MC)、MMLU‑STEM、TriviaQA(0/5‑shot)

**数学 / 生成**:GSM8k(5-shot)、MATH 7 subset(4-shot)

**代码**:HumanEval(`bigcode-eval-harness`, T=0.2, top-p=0.95, n=20)

**Metric**:`loglikelihood_acc` + `loglikelihood_acc_norm_nospace`(CSR/MMLU);`quasi_exact_match`(GSM8k/MATH/TriviaQA);`pass@1`(HumanEval)

### 2.4 降本技巧(Proxy ablation)

三级漏斗:
1. **粗筛**:360M 模型 × 30B token → 筛掉大批候选
2. **中验**:1.82B × 30B token
3. **终验**:1.82B × 100B / 350B token

通常一个变量只需粗筛 + 中验即可得出 95% 信心的结论。

---

## 3 · Incepedia 相对 Cosmopedia v2 的 4 个超越点

### 3.1 多生成器混合(对抗 L1/L3 同质化)
- 至少 4 个家族交叉(OpenAI / Anthropic / DeepSeek / Llama)
- 每个 (topic, format) 切片必须触达 ≥ 2 家族
- 按任务档位路由:常识走 midtier_bulk,推理/数学走 frontier_reasoning

### 3.2 Web‑grounded RAG + seed 强制注入(反幻觉,抗 L4)
- 每条 prompt 必须绑定 1–3 段外部事实(Wikipedia / arXiv abstract / 教科书段)
- Prompt 模板含 *"All factual claims MUST be derivable from the seed text or general knowledge; if uncertain, write 'I am not sure'"*

### 3.3 多 agent 自批改(Phi-4 路线)
- 对 STEM / 代码高价值样本:generator → critic → reviser 三步
- critic 用 frontier 模型,判断事实 + 推理 + 完整性
- 低分样本重写,极低分丢弃

### 3.4 三层去污染(超越 Cosmopedia 的 10-gram 浅检测)
- **Layer 1**:13-gram + `difflib.SequenceMatcher` ratio > 0.3
- **Layer 2**:sentence-transformer embedding 相似度 cos > 0.85
- **Layer 3**:LLM-as-judge(对 Layer 2 命中对儿做二分类)
- **Layer 4(审计)**:训练后 Min-K% Prob membership inference

覆盖评测集:MMLU / MMLU-Pro / MMLU-STEM / HellaSwag / PIQA / SIQA / Winogrande / OpenBookQA / ARC(E/C) / CommonsenseQA / BoolQ / TriviaQA / GSM8k / MATH / HumanEval / MBPP。

### 3.5 (额外)Embedding-level 近重复去重
Cosmopedia 只做了 MinHash。我们加 BGE-M3 + FAISS,cos > 0.92 视为近重复删除 — 这一层抓的是"同一概念不同句式"。

---

## 4 · Incepedia 生成 pipeline(要实现)

```
 topics.yaml  ──┐
 personas.yaml ─┼─→  PromptAssembler  ─→  OpenRouter (按 tier 路由) ─→  raw_output.parquet
 web seeds    ──┤                                │
 RAG retriever ─┘                                ├─→ (optional) critic round
                                                 │
                                     ↓ boilerplate cleanup
                                     ↓ MinHash dedup
                                     ↓ embedding dedup
                                     ↓ 3-layer decontam
                                     ↓ quality classifier (FineWeb-edu)
                                     ↓ tokenization (datatrove)
                               ─→  data/datasets/incepedia_v0.x/
```

---

## 5 · 三阶段路线图

| 阶段 | 时长估计 | 目标 | 成功标准 |
|---|---|---|---|
| **P1 · 复现** | 2–3 周 | 复现 Cosmopedia v2 reference baseline | 1.82B/30B ablation 分数对齐 SmolLM 公布数字 ±0.5pp |
| **P2 · PoC** | 4–6 周 | 3B token Incepedia v0.1 | 同 ablation 下平均分 ≥ Cosmopedia v2 |
| **P3 · 迭代 / 扩容** | 2–3 月 | 10B–30B token Incepedia v1.0 | 显著超越 Cosmopedia v2,逼近 SmolLM2 |

每阶段结束写一份 retrospective + ADR。

---

## 6 · 关键决策(ADR 索引)

- [`0001-three-layer-storage.md`](./decisions/0001-three-layer-storage.md) · 本地 + NAS 冷备 + git 元数据
- [`0002-ablation-as-quality-metric.md`](./decisions/0002-ablation-as-quality-metric.md) · 数据质量 = ablation 分数
- [`0003-multi-tier-generator-routing.md`](./decisions/0003-multi-tier-generator-routing.md) · 生成器按任务档位路由

新决策按 `docs/decisions/TEMPLATE.md` 起草。

---

## 7 · 命名与数据契约

- 实验命名:`exp_{YYYYMMDD}_{slug}_seed{N}`
- 数据集命名:`incepedia_v{major}.{minor}`
- Config 规范:见 `AGENTS.md § Config file format`
- INDEX.parquet 字段:见 `scripts/index_experiment.py` 顶部 `SCHEMA_COLS`

---

## 8 · 开放问题(持续跟进)

- [ ] 难度分层的粒度(5 档 vs 3 档?)—— 待 P2 ablation 回答
- [ ] 是否把 instruction reversal 作为独立 split 训练 —— 待 Phi-4 细节进一步调研
- [ ] Curriculum ordering:cooldown 阶段纯 Incepedia vs 混 web 比例 —— 待 P3 ablation
- [ ] Embedding 去重的 threshold(0.90 vs 0.92 vs 0.95)—— 待 P2 检验

---

© 2026 · Incepedia contributors · Apache-2.0
