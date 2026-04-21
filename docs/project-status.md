# Incepedia · 项目状态简报

| 字段 | 值 |
|---|---|
| 更新频率 | 每周一 / 周四,2 次/周 |
| 本次更新 | 2026-04-20(链式执行进行到 Phase 4 中途) |
| 当前阶段 | **P1 · 基础设施搭建**(~80% 完成) |
| Owner | bill-lee-mk |

> 本文档是项目的"一页纸"视图。细节见 `docs/methodology.md` 与 `docs/decisions/`。

---

## 1. 一句话定义

**构造一个合成预训练数据集 Incepedia,在相同 1.82B Llama2 ablation 协议下,同时以"独立 corpus"和"decay 调料"两种身份,下游 benchmark 分数超越 Cosmopedia。**

---

## 2. 愿景

- **长期(12+ 个月)**:让 Incepedia 成为开源社区继 Cosmopedia 之后的首选合成预训练数据集。
- **中期(3–6 个月)**:发布 Incepedia v1.0(10B+ tokens),standalone 与 seasoning 两种场景均显著超越 Cosmopedia v1/v2。
- **短期(3 个月,即本周期)**:P1 基线复现 → P2 PoC → P3 迭代到 v1.0。

---

## 3. 具体目标(按阶段)

| 阶段 | 时长 | 目标 | 成功标准 | 状态 |
|---|---|---|---|---|
| **P1 · 复现** | 2–3 周 | 复现 Cosmopedia v2 在 Track 2 + Cosmopedia v1 在 Track 1 基线 | 分数对齐公开数字 ±0.5pp | 🟡 80% 基础设施就绪,待 GPU 空闲做 smoke test |
| **P2 · PoC** | 4–6 周 | 3B token Incepedia v0.1 | 两轨均持平或超越 Cosmopedia | ⏳ 未开始 |
| **P3 · 迭代+扩容** | 2–3 月 | 10B–30B token Incepedia v1.0 | 两轨显著超越 | ⏳ 未开始 |

---

## 4. 方法(科学判据,ADR 0004)

**数据质量 = 下游模型在固定评测套件上的分数。**

| 要素 | 固定配置 |
|---|---|
| 模型 | Llama2 ~1.82B |
| Tokenizer | `mistralai/Mistral-7B-v0.1` |
| 训练 token | 30B(快 ablation)/ 100B(中验证)/ 350B(终验) |
| Seeds | 2 个(42, 1337),分数平均 |
| 评测 | `lighteval` 0.13 + 自维护 port(ADR 0006) |
| 协议 | Track 1(standalone)/ Track 2(cooldown-fork),自适应选择 |

---

## 5. 策略 —— C1 至 C16(methodology.md §3)

**三组差异化设计**:

| 组 | 决策 | 核心 |
|---|---|---|
| 基础(C1–C4) | seed-grounding + persona + 5 档难度 + 12+ 结构模板 | 反幻觉 + 反同质化 |
| 工程(C5–C12) | 双 key 轮换 + 异步并发 + checkpoint + 实时监控 + 成本闸门 + critic loop | 吞吐 / 质量 / 可控 |
| 超越点(C13–C16) | instruction reversal + 风格反单调 + **web rephrasing 主路径** + synthetic 贯穿全程 | 相对 Cosmopedia 的核心差异化 |

---

## 6. 存储 & 资源架构

**三层存储防护**(ADR 0001):

| 层 | 位置 | 内容 | 状态 |
|---|---|---|---|
| L1 · 热 | 本地 NVMe(8.2 TB 可用) | 全量数据 + ckpt + aim 日志 | ✅ |
| L2 · 冷备 | Lambda NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/` | 事件驱动 rsync 冷镜像 | ✅ 参考数据已镜像 |
| L3 · 元数据 | GitHub `bill-lee-mk/Incepedia` | config / metrics / ADR / INDEX.parquet | ✅ 15 个 commit |

**算力**:单节点 8× H100 80GB HBM3(**共享服务器,目前被其他用户 Qwen3 训练占满**)。
**API 预算**:OpenRouter(已配 2 把 key,目标 ~$22k 全项目合成成本)。

---

## 7. 当前进度(截至 2026-04-20 ~13:05 UTC)

| 模块 | 状态 | 备注 |
|---|---|---|
| 仓库骨架 | ✅ | 16 个 commit |
| 存储三层 + 同步脚本 | ✅ | event-driven rsync + audit log |
| 参考数据 · Cosmopedia v2(raw) | ✅ | 114 GB / 104 shards,双副本 |
| 参考数据 · FineWeb-Edu 32 shards(raw) | ✅ | 72 GB,双副本 |
| 参考数据 · Cosmopedia v1(raw) | ⏳ | P3 阶段再拉 |
| **Conda env + 依赖**(Python 3.11 + lighteval 0.13) | ✅ | 8.5 GB env,所有 import 通过,CUDA OK |
| **评测层**(port + runner, lighteval 0.13 API) | ✅ | 136 tasks 注册,driver-script 绕 CLI |
| **训练层**(config schema + launcher + orchestrator) | ✅ | pydantic + nanotron YAML 渲染 + 主编排,dry-run 通过 |
| **首个 reference 实验 config**(Track 1 standalone) | ✅ | `experiments/exp_ref_cosmopedia_v2_seed42/config.yaml` |
| **Backbone 共享实验 config**(Track 2) | ✅ | `experiments/backbone_fineweb_edu/config.yaml` |
| **Tokenized · Cosmopedia v2** | ✅ | **58.76 GiB / 31.5B tokens / 16 .ds shards** · 9.2 min 完成 |
| Tokenized · FineWeb-Edu backbone | 🟡 | 脚本就绪,待 Shell 恢复后启动(~5 min 即可完成) |
| Smoke test · cosmo-1b early-signal | 🛑 | **GPU 冲突**:8×H100 被其他用户 Qwen3 训练占满(已持续 3+ 小时) |
| 合成 pipeline(OpenRouter 路由) | ⏳ | P2 前完成 |

---

## 8. 最新评测结果

> 每次评测完成后填入下表。双轨分别列出。

### Track 1 · Standalone
| 实验 | 模型 | 训练数据 | Tokens | HellaSwag | MMLU cloze | ARC-C | GSM8k | 平均 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | 等 Smoke test + P1 实验 |

### Track 2 · Seasoning
| 实验 | Backbone | Cooldown 数据 | ... | — | 等 backbone 训练 |

---

## 9. 本周期下一步

**本周结束前**:
- [ ] 等 GPU 空闲,跑 cosmo-1b smoke test 验证评测 pipeline
- [ ] 完成 FineWeb-Edu tokenize(~5 min CPU 工作)
- [ ] 安装 flash-attn(为训练做准备,30-60 min 编译)
- [ ] 跑共享 backbone 训练(1.4 天 × 8 H100)

**下周**:
- [ ] 跑 `exp_ref_cosmopedia_v2_seed42`(Track 1 milestone 基线)
- [ ] 对齐 SmolLM 公开分数 ±0.5pp
- [ ] 开始 Incepedia 合成 pipeline 编码(C13–C16)

---

## 10. 风险 & 开放问题

| 风险 / 问题 | 影响 | 当前状态 |
|---|---|---|
| GPU 资源共享(opsd 用户 Qwen3 训练占满) | High | 🟡 延后 smoke test + backbone 训练,等对方跑完 |
| Agent Shell 子进程偶发挂死(本会话已发生 2 次) | Medium | 需重启 Cursor agent 会话恢复 |
| lighteval 0.13 API 重写可能引入分数漂移 | Medium | 待 smoke test 验证 |
| flash_attn 编译耗时 30–60 min,训练前必装 | Low | 已规划 |
| numpy 1.26 vs 2.x 依赖冲突 | Low | lighteval 0.13 拉回 numpy 2.4,实测 OK |

---

## 11. 本次会话(2026-04-20)链式执行记录

| Phase | 描述 | 状态 | Commit |
|---|---|---|---|
| 0 | 环境重建(Python 3.11 + lighteval 0.13) | ✅ | 环境,未 commit |
| 1 | Port + Runner 重写到 0.13 API + ADR 0006 | ✅ | `5696b4f`, `d7ec352` |
| 2 | Smoke test cosmo-1b | 🛑 GPU 冲突 | — |
| 3 | 训练层代码(schema + launcher + orchestrator) | ✅ | `d2a6ecd` |
| 4a | Tokenize Cosmopedia v2 | ✅ | `c746d96`(脚本);数据未 commit(.gitignored) |
| 4b | Tokenize FineWeb-Edu | 🟡 脚本就绪,Shell 挂死未启动 | — |
| 5 | 终报告 | 🟡 本文件即是 | — |

**本会话 commits**(≤60 字符规则):
```
c746d96 feat(data): add datatrove tokenizer pipeline for reference data
d2a6ecd feat(training): add config schema, nanotron launcher, orchestrator
d7ec352 fix(gitignore): exclude third_party (not submodules)
5696b4f refactor(eval): adapt port + runner to lighteval 0.13 API
```

---

## 12. 关键文档索引

- **方法论详解** → `docs/methodology.md`
- **ADR 列表** → `docs/decisions/README.md`(已有 6 条 ADR)
- **AI agent 守则** → `AGENTS.md`
- **人类贡献指南** → `CONTRIBUTING.md`
- **存储使用手册** → `README.md` + NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/README.md`
- **实验索引** → `INDEX.parquet`(待首个实验完成后生成)

---

## 变更历史

| 日期 | 要点 |
|---|---|
| 2026-04-20 | 创建,P1 筹备阶段 |
| 2026-04-20 | 完成环境重建 + eval/training 代码重写 + Cosmopedia v2 tokenize (31.5B tokens)。Smoke test 阻塞于 GPU 冲突;FineWeb tokenize 未启动(Shell 挂死) |
