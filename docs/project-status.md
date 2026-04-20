# Incepedia · 项目状态简报

| 字段 | 值 |
|---|---|
| 更新频率 | 每周一 / 周四,2 次/周 |
| 本次更新 | 2026-04-20 |
| 当前阶段 | **P1 · 复现基线**(筹备阶段) |
| Owner | bill-lee-mk |

> 本文档是项目的"一页纸"视图。细节见 `docs/methodology.md` 与 `docs/decisions/`。

---

## 1. 一句话定义

**构造一个合成预训练数据集 Incepedia,在相同 1.82B Llama2 ablation 协议下,同时以"独立 corpus"和"decay 调料"两种身份,下游 benchmark 分数超越 Cosmopedia。**

---

## 2. 愿景

- **长期(12+ 个月)**:让 Incepedia 成为开源社区继 Cosmopedia 之后的首选合成预训练数据集,推动"合成数据是否为大模型训练新主力"议题的可验证进展。
- **中期(3–6 个月)**:发布 Incepedia v1.0(10B+ tokens),在 standalone 和 seasoning 两种场景下均显著超越 Cosmopedia v1/v2。
- **短期(3 个月,即本周期)**:完成 P1 基线复现 + P2 PoC(Incepedia v0.1,3B tokens)+ P3 迭代到 v1.0。

---

## 3. 具体目标(按阶段)

| 阶段 | 时长 | 目标 | 成功标准 | 状态 |
|---|---|---|---|---|
| **P1 · 复现** | 2–3 周 | 复现 Cosmopedia v2 在 Track 2 + Cosmopedia v1 在 Track 1 的基线 | 分数对齐公开数字 ±0.5pp | 🟡 进行中 |
| **P2 · PoC** | 4–6 周 | 3B token Incepedia v0.1 | 两轨均持平或超越 Cosmopedia | ⏳ 未开始 |
| **P3 · 迭代+扩容** | 2–3 月 | 10B–30B token Incepedia v1.0 | 两轨显著超越,逼近 SmolLM2 | ⏳ 未开始 |

---

## 4. 方法 —— 科学判据(ADR 0004)

**数据质量 = 下游模型在固定评测套件上的分数。** 不接受主观判断。

| 要素 | 固定配置 |
|---|---|
| 模型 | Llama2 ~1.82B(FineWeb ablation 标准) |
| Tokenizer | `mistralai/Mistral-7B-v0.1` 或 SmolLM |
| 训练 token | 30B(快 ablation)/ 100B(中验证)/ 350B(终验) |
| Seeds | 2 个(42, 1337),分数平均 |
| 评测 | `lighteval` early-signal 套件(CSR + MMLU + GSM8k) |
| 协议 | Track 1(standalone)/ Track 2(cooldown-fork),自适应选择 |

---

## 5. 策略 —— C1 至 C16(ADR 0003 + 方法论 §3)

**三组差异化设计**:

| 组 | 决策 | 核心 |
|---|---|---|
| **基础(C1–C4)** | seed-grounding + persona + 5 档难度 + 12+ 结构模板 | 反幻觉 + 反同质化 |
| **工程(C5–C12)** | 双 key 轮换 + 异步并发 + checkpoint + 实时监控 + 成本闸门 + critic loop | 吞吐 / 质量 / 可控 |
| **超越点(C13–C16)** | instruction reversal + 风格反单调 + **web rephrasing 主路径** + synthetic 贯穿全程 | 相对 Cosmopedia 的核心差异化 |

---

## 6. 存储 & 资源架构

**三层存储防护**(ADR 0001):

| 层 | 位置 | 内容 | 状态 |
|---|---|---|---|
| L1 · 热 | 本地 NVMe(8.2 TB 可用) | 全量数据 + ckpt + aim 日志 | ✅ |
| L2 · 冷备 | Lambda NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/`(3.1 PB) | 事件驱动 rsync 冷镜像 | ✅ |
| L3 · 元数据 | GitHub `bill-lee-mk/Incepedia` | config / metrics / ADR / INDEX.parquet | ✅ |

**算力**:单节点 8× H100 80GB HBM3(未来可扩节点)。

**API 预算**:OpenRouter(已配 2 把 key,目标 ~$22k 全项目合成成本)。

---

## 7. 当前进度(截至 2026-04-20)

| 模块 | 状态 | 备注 |
|---|---|---|
| 仓库骨架(README / ADR / AGENTS.md) | ✅ | 10 个 commit |
| 存储三层 + 同步脚本 | ✅ | event-driven rsync + audit log |
| 参考数据 · Cosmopedia v2 | ✅ | 114 GB / 104 shards,双副本 |
| 参考数据 · FineWeb-Edu(32 shards ~30B tokens) | ✅ | 72 GB,双副本 |
| 参考数据 · Cosmopedia v1 | ⏳ | P3 阶段再拉 |
| Conda env + 依赖 | 🟡 | 准备切到 Python 3.11 + lighteval 0.13 |
| 评测层(port + runner) | 🟡 | 正在适配 lighteval 0.13 API |
| 训练层(backbone / standalone / cooldown) | ⏳ | 代码待写 |
| Tokenized 数据 | ⏳ | datatrove pipeline 待运行 |
| 合成 pipeline(OpenRouter 路由) | ⏳ | P2 前完成 |

---

## 8. 最新评测结果

> 每次评测完成后填入下表。双轨分别列出。

### Track 1 · Standalone

| 实验 | 模型 | 训练数据 | Tokens | HellaSwag | MMLU cloze | ARC-C | GSM8k | 平均 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | 尚无结果 |

### Track 2 · Seasoning (cooldown-fork)

| 实验 | Backbone ckpt | Cooldown 数据 | HellaSwag | MMLU cloze | ARC-C | GSM8k | 平均 | Δ vs Cosmo-v2 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | 尚无结果 |

---

## 9. 本周期下一步

**本周(4 月 21–27)**:
- [ ] 环境切到 Python 3.11 + lighteval 0.13
- [ ] 重写评测 port 到 v0.13 API + 写 ADR 0006
- [ ] Smoke test:用 `cosmo-1b` 跑 early-signal,验证 pipeline
- [ ] 写训练层代码(backbone / standalone / cooldown-fork launchers)
- [ ] Tokenize Cosmopedia v2 和 FineWeb-Edu 参考数据

**下周(4 月 28 – 5 月 4)**:
- [ ] 训练共享 backbone(1.82B × 20B FineWeb-Edu)
- [ ] 跑 `exp_ref_cosmopedia_v2`(Track 1 基线 milestone)
- [ ] 对齐 SmolLM 公开分数 ±0.5pp

---

## 10. 风险 & 开放问题

| 风险 / 问题 | 影响 | 缓解措施 |
|---|---|---|
| lighteval 0.13 API 重写可能引入隐性分数漂移 | Medium | smoke test 对比 cosmo-1b 公开分数 ±3pp |
| numpy 1.26 vs 2.x 依赖冲突 | Low | 已锁 1.26,实测运行 OK |
| flash_attn 编译耗时 30–60 min | Low | 推迟到训练开始前再装 |
| Track 1 与 Track 2 分数方向在某变量上冲突 | Medium | 出现时新 ADR 定取舍优先级 |
| OpenRouter rate limit | Low | 已配双 key 轮换 |

---

## 11. 关键文档索引

- **方法论详解** → `docs/methodology.md`
- **ADR 列表** → `docs/decisions/README.md`
- **AI agent 守则** → `AGENTS.md`
- **人类贡献指南** → `CONTRIBUTING.md`
- **存储使用手册** → `README.md` + NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/README.md`
- **实验索引** → `INDEX.parquet`(待首个实验完成后生成)

---

## 变更历史

| 日期 | 要点 |
|---|---|
| 2026-04-20 | 创建,P1 筹备阶段 |
