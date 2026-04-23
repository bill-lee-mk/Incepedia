# Incepedia · 项目代号对照表(一页纸)

> 遇到任何 `P*` `C*` `T*` `R*` `ADR*` `Track*` `Protocol*` `L*` 等代号,这里一次说清。

---

## 1. 层级关系总览

```
                                   Incepedia 项目
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
      【项目阶段】                 【架构决策】                  【工程执行】
         P1 / P2 / P3             ADR 0001 – 0009               TODO T1 – T5
                                        │
                                        │ 展开细节
                                        │
            ┌───────────┬──────────────┼───────────┬────────────┐
            │           │              │           │            │
       存储(L1-L3)  评测(Track)   架构(Protocol)  后端  生成器策略(C1-C16)
        ADR 0001    ADR 0004      ADR 0007    ADR 0006/0008/0009  ADR 0003
                                                                   │
                                                              methodology.md §3

            【训练轮次(conversational)】
                   R0(sanity)  → R1(正式 30B)  → 未来 R2 R3...
```

## 2. 代号总表(按首字母字母序)

| 代号 | 全称 / 含义 | 所在阶段 / 管线 | 验证 / 消融目标 | 权威来源 |
|------|-------------|-----------------|-----------------|----------|
| **ADR 0001** | Three-layer storage defense(本地 NVMe + NAS 冷镜像 + git 元数据) | 基础设施 | 任意一层失败数据不丢 | `docs/decisions/0001-three-layer-storage.md` |
| **ADR 0002** | Ablation as quality metric(ablation 分数 = 数据质量唯一指标) | 方法论基石 | 排除主观判断 | `docs/decisions/0002-ablation-as-quality-metric.md` |
| **ADR 0003** | Multi-tier generator routing(按任务路由到 midtier / frontier / critic 档 LLM) | 生成管线 P2 | 算力 × 质量效率比 | `docs/decisions/0003-multi-tier-generator-routing.md` |
| **ADR 0004** | Dual-track adaptive evaluation(Track 1 + Track 2 双轨评测) | 评测管线 | standalone 能力 + seasoning 能力 | `docs/decisions/0004-evaluation-protocol-dual-track-adaptive.md` |
| **ADR 0005** | Project scope:standalone + seasoning 双场景 | 项目 scope | 覆盖 Cosmopedia v1 / v2 两种用法 | `docs/decisions/0005-project-scope-standalone-plus-seasoning.md` |
| **ADR 0006** | Evaluation stack policy — pin to lighteval latest | 评测工具链 | 工具漂移管理 | `docs/decisions/0006-evaluation-stack-policy.md` |
| **ADR 0007** | Dual-protocol architecture(Llama2-1.82B 锚 + Qwen3-1.7B 工作) | 训练架构 | 外部可比 + modern 架构双保险 | `docs/decisions/0007-dual-protocol-architecture.md` |
| **ADR 0008** | FlashAttention-3 可选 backend(默认 OFF) | 训练后端 | 速度优化(实测 1.7B 无端到端增益) | `docs/decisions/0008-flash-attn-3-optional.md` |
| **ADR 0009** | nanotron → HF 转换必经之路 | train→eval 管线 | 评测/发布标准化 | `docs/decisions/0009-nanotron-to-hf-conversion.md` |
| **C1** | 强制 seed-grounding(每条 prompt 必带 Wikipedia/arXiv/FineWeb-Edu 段落) | 生成管线 P2 | 降低 Mixtral 时代幻觉 | `docs/methodology.md` §3.1 |
| **C2** | Persona 注入(~10k personas 按权重抽) | 生成管线 P2 | 反 L3 风格同质化 | `docs/methodology.md` §3.1 |
| **C3** | 5 档难度(elementary → expert) | 生成管线 P2 | 支持 C16 课程实验 | `docs/methodology.md` §3.1 |
| **C4** | 12+ 结构模板(textbook/blog/Q&A/Socratic/...) | 生成管线 P2 | 反 L1 模板单一 | `docs/methodology.md` §3.1 |
| **C5** | 双 OpenRouter key 轮换 | 生产线 | 并发/限流 | `docs/methodology.md` §3.2 |
| **C6** | `httpx.AsyncClient + aiolimiter + tenacity` | 生产线 | 异步批量 + 指数退避 | `docs/methodology.md` §3.2 |
| **C7** | Checkpoint:每 1000 条 Parquet + 原子重命名 | 生产线 | 崩溃重启不丢数据 | `docs/methodology.md` §3.2 |
| **C8** | 实时监控(成功率 / 成本 / dup 率) | 生产线 | 质量预警 | `docs/methodology.md` §3.2 |
| **C9** | 成本闸门(偏差 >20% 自动暂停) | 生产线 | 预算保护 | `docs/methodology.md` §3.2 |
| **C10** | Critic loop(STEM/code/expert ~15% 触发) | 生产线 | 对困难样本二次质检 | `docs/methodology.md` §3.2 |
| **C11** | 输出 schema(含 generator/seed_hash/critic_score/...) | 生产线 | 追溯性 + 后续去重 | `docs/methodology.md` §3.2 |
| **C12** | 每批合成由 `batch.yaml` 完全定义 | 生产线 | 可复现 | `docs/methodology.md` §3.2 |
| **C13** | **Instruction reversal**(Phi-4 方法,web→问答对) | 生成管线差异化 | 超越 Cosmopedia 的核心策略 #1 | `docs/methodology.md` §3.3 |
| **C14** | **风格反单调化**(`disallowed_openings.yaml` 动态扩容) | 生成管线差异化 | 超越 Cosmopedia 的核心策略 #2 | `docs/methodology.md` §3.3 |
| **C15** | **Web rephrasing 主路径**(Nemotron / WRAP / Phi-4) | 生成管线差异化 | 超越 Cosmopedia 的核心策略 #3(40-50% 量) | `docs/methodology.md` §3.3 |
| **C16** | **Synthetic 贯穿 pretrain 全程**(Phi-4) | 训练配方 | 超越 Cosmopedia 的核心策略 #4 | `docs/methodology.md` §3.3 |
| **L1 / L2 / L3** | 三层存储(本地 NVMe / NAS 冷镜像 / git 元数据)| 基础设施 | 最坏情况 2 层挂仍保配置和分数 | ADR 0001 / README § 三层存储防护 |
| **P1** | Phase 1 · Reproduce(复现) | 当前阶段 | 与 SmolLM/FineWeb 公开数字 ±0.5pp | README / methodology.md §5 |
| **P2** | Phase 2 · PoC(概念验证,3B Incepedia v0.1) | 下一阶段 | 最小闭环:生成→训→评 | README / methodology.md §5 |
| **P3** | Phase 3 · Iterate & scale(10-30B Incepedia v1.0 + HF 发布) | 终局 | 跨 benchmark 超 Cosmopedia + 论文 | README / methodology.md §5 |
| **Protocol A** | Llama2-1.82B 外部锚(RoPE 10000,无 bias,Mistral tokenizer) | 训练架构 | 与 SmolLM/Cosmopedia 严格可比 | ADR 0007 |
| **Protocol B** | Qwen3-1.7B 工作架构(GQA,RoPE 1e6,QKV bias,Qwen tokenizer) | 训练架构 | 日常 ablation + 发布 | ADR 0007 |
| **R0** | ("rerun-0") 1.5B sanity-eval round | conversational 指代 | 验证 eval pipeline 端到端 | R1 前的 launcher bug 产物 |
| **R1** | ("rerun-1") 30B 正式训练 round(当前!) | conversational 指代 | 拿到可与 Cosmopedia 比的真实分数 | 当前 seed42 30B 跑到 48% |
| **T1** | **FA3 switch**(`NANOTRON_USE_FA3=1`) | 工程 TODO ✅ 已完成 | 验证 FA3 vs FA2 数值等价 + 速度 | `docs/project-status.md` § TODO |
| **T2** | **nanotron→Aim metrics 桥接**(sidecar 实现,待升级到直连) | 工程 TODO 🟡 sidecar done,直连 待办 | 实时训练曲线监控 | 同上 |
| **T3** | **本地 patches 自动应用**(bootstrap_env.sh 内 idempotent) | 工程 TODO ✅ 已完成 | 环境重建无手工步骤 | 同上 |
| **T4** | **lint_nanotron_yaml.py**(launcher yaml 对齐 nanotron 默认值) | 工程 TODO ✅ 已完成 | 防止 `ignore_sanity_checks=False` 类暗坑 | 同上 |
| **T5** | **nanotron 原生直连 Aim/W&B**(替代当前 sidecar) | 工程 TODO 🟡 待 R1 完成后做 | 彻底告别 log-tail 妥协 | 同上 |
| **Track 1** | Standalone 全量训(1.82B/1.7B × 30B × 2 seeds,纯数据集) | 评测协议 | 数据集作"主力"的能力 | ADR 0004 |
| **Track 2** | Seasoning cooldown-fork(共享 backbone 20B + 调料 6B) | 评测协议 | 数据集作"调料"的能力 | ADR 0004 |

---

## 3. 阶段 × 管线 × 代号 的二维图

```
─────────┬─────────────┬──────────────┬──────────────┬──────────────┬──────────────
 阶段    │ 生成管线     │ 训练管线      │ 评测管线      │ 存储          │ 监控
─────────┼─────────────┼──────────────┼──────────────┼──────────────┼──────────────
         │             │              │              │              │
 P1 复现 │  (N/A)      │ ADR 0007      │ ADR 0004     │ ADR 0001     │ T2 sidecar
         │             │ Protocol A/B  │ Track 1/2    │ L1/L2/L3     │ (T5 future)
         │             │ ADR 0008 FA3  │ ADR 0006     │              │
         │             │ ADR 0009 HF   │ ADR 0009 HF  │              │
         │             │ T1/T3/T4      │              │              │
         │             │              │              │              │
         │             │ R0 → R1 ★    │ R0 → R1 ★   │              │
         │             │              │              │              │
─────────┼─────────────┼──────────────┼──────────────┼──────────────┼──────────────
         │             │              │              │              │
 P2 PoC  │ ADR 0003    │ (同上,不变)  │ (同上,不变) │ (同上,不变) │ (同上)
         │ C1-C12      │              │              │              │
         │ C13-C16★    │              │              │              │
         │             │              │              │              │
─────────┼─────────────┼──────────────┼──────────────┼──────────────┼──────────────
         │             │              │              │              │
 P3 扩容 │ (C13-C16   │ (+ Protocol A │ (跨多 seed   │ (+ HF Hub    │ (+ W&B 云端
  发布   │   成熟化)   │ 锚验证,      │   平均 + 多   │ 发布层)       │ 跨机监控)
         │             │ ~5 天 GPU)    │ benchmark)   │              │
         │             │              │              │              │
─────────┴─────────────┴──────────────┴──────────────┴──────────────┴──────────────

★ = 关键推进点
```

---

## 4. 按"我想问什么"反查

### "现在 R1 是什么?"
= 当前跑的 seed42 第 2 次正式训练(第 1 次有 launcher bug 只训 1.5B = R0)。
一个 30B token 训练,ETA 23h。

### "P1 什么时候结束?"
P1 完成条件 = Cosmopedia v2 × {Protocol A × 2 seeds,Protocol B × 2 seeds} 都训完
且每个 seed 都过 cosmopedia-full eval,分数与公开数字 ±0.5pp。
当前:Protocol B seed42 R1 48% done;seed1337 / Protocol A 未开始。

### "C13-C16 和 C1-C12 的区别?"
- C1-C12:**生成的基础设施**(怎么让 LLM 稳定产出结构化数据)
- C13-C16:**超越 Cosmopedia 的 4 条差异化策略**(真正创新点)

### "Track 1 用 Cosmopedia v1 还是 v2?"
方法论说 Track 1 standalone 对应 Cosmopedia v1(纯数据集)场景,
但我们实际用 v2 做 P1 reference baseline(v1 未 tokenize 尚未获取)。
P2 开始时需要补上 v1 数据。**(已进"缺口表"第 5 项)**

### "ADR 0007 和 Protocol A/B 是一回事?"
ADR 0007 是**决策文档**,决定了"用两套架构跑";
Protocol A/B 是**这个决策产物的两个实例**。
读 ADR 0007 即理解 Protocol A/B。

### "T5 为什么要做?"
当前 Aim 监控曲线走 sidecar(tail train.log → 正则 → push Aim),1-2 步延迟、
正则脆。T5 是让 nanotron 在训练内直接调 `aim.Run.track()`,0 延迟、无解析。
R1 跑完后立即做。

### "L1 L2 L3 和 P1 P2 P3 冲突吗?"
完全不同两个轴。
- **L** = Layer(存储层,基础设施维度)
- **P** = Phase(项目阶段,时间维度)

### "R1 后面还有 R2 R3 吗?"
`R*` 是对话里我用的非正式序号,主要用于同一 seed 的**重跑**(rerun)。
正式项目语境里更重要的是:seed42 + seed1337 × Protocol A/B × milestones。
未来不会频繁提到 R2 / R3。

---

## 5. 维护约定

- **新增 ADR** → 在 `docs/decisions/` 新建 `0010-*.md` + 更新 `decisions/README.md` 索引。
- **新增 TODO (T6+)** → 加到 `docs/project-status.md § TODO 队列`,更新此表。
- **新增 C 策略(C17+)** → 加到 `docs/methodology.md §3`,更新此表。
- **阶段推进** → 更新 `README.md` 首屏「当前阶段」+ `docs/project-status.md` 变更历史。

---

© 2026 · Incepedia contributors · Apache-2.0
