# Incepedia · 项目代号对照表(一页纸)

> 遇到任何 `P*` `C*` `E*` `A*` `G*` `W*` `T*` `R*` `ADR*` `Track*` `Protocol*` `L*` `①–⑨` 等代号,这里一次说清。
>
> 本文档是**代号 + 生成管线**的单一真相源。`docs/methodology.md` 讲方法论,`docs/incepedia-overview.md` 讲叙事,`docs/project-status.md` 讲进度,三者与本文件一致时以本文件为准。

---

## 1. 层级关系总览

```
                                   Incepedia 项目
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
      【项目阶段】                 【架构决策】                  【工程执行】
        P1 / P2 / P3              ADR 0001 – 0009              TODO T1 – T5
         │      │
         │      │
         │      └──── P2/P3 衍生的"可执行单元":
         │              ├─ E1–E5    v0.1 实验组(Track 1)
         │              ├─ A1–A10   单变量 ablation(Track 2)
         │              ├─ G1–G4    go/no-go 门控
         │              └─ W1–W6    6 周实施周次
         │
         └──── 横切维度:
                 ├─ 存储层   L1 / L2 / L3                    (ADR 0001)
                 ├─ 评测轨道 Track 1 / Track 2               (ADR 0004)
                 ├─ 架构协议 Protocol A / Protocol B          (ADR 0007)
                 ├─ 生成策略 C1 – C16 + ①–⑨ 质量升级         (ADR 0003, methodology §3)
                 └─ 训练轮次 R0(sanity)→ R1(正式 30B)→ R2...
```

---

## 2. 管线 × 阶段 × 代号(二维)

> 读法:**5 个管线横排表头**(生成 / 训练 / 评测 / 存储 / 监控),**3 个阶段竖排第一列**(P1 / P2 / P3)。每格列出**该阶段该管线下实际落地的所有代号**。与 §3 各分组表一一对齐(§3.3 的 C、§3.4 的 ①–⑨、§3.5 的 E、§3.6 的 A、§3.7 的 G、§3.8 的 W、§3.9 的 L、§3.10 的 Protocol、§3.11 的 Track、§3.12 的 R、§3.13 的 T)。★ = 该格的关键推进点。

### 2.1 细粒度矩阵

| 阶段 \ 管线 | **生成管线**<br>(ADR 0003) | **训练管线**<br>(ADR 0007 / 0008 / 0009) | **评测管线**<br>(ADR 0004 / 0006 / 0009) | **存储**<br>(ADR 0001) | **监控** |
|---|---|---|---|---|---|
| **P1 复现**<br>🟡 R1 进行中,40% | — N/A,复用 Cosmopedia v2 全量 28B uniq(不自产数据) | • **架构**:**Protocol A**(Llama2-1.82B,Mistral-7B tokenizer,**仅用 2 次**)+ **Protocol B**(Qwen3-1.7B,Qwen 151k,工作架构)<br>• **后端**:FA2 默认 + ADR 0008 FA3 switch(**T1**,默认 OFF)<br>• **数据**:`cosmo_v2_qwen3` tokenized dataset<br>• **转换**:ADR 0009 nanotron → HF(eval 前置必经)<br>• **工程 TODO**:**T3** bootstrap_env.sh 自动 apply 4 个 nanotron 补丁 · **T4** lint_nanotron_yaml.py YAML 审计<br>• **运行轨迹**:**R0**(弃)→ ★**R1** seed42 30B(40%)→ **R2** seed1337 30B → **E1 = R1+R2** 基线入账 | • **轨道**:Track 1 / Track 2 双轨协议就位(ADR 0004),P1 主用 Track 1 跑通端到端<br>• **工具链**:lighteval pinned(ADR 0006)<br>• **输入**:ADR 0009 HF ckpt(通过 `scripts/convert_nanotron_qwen2_to_hf.py`)<br>• **任务组**:**`cosmopedia-full`**(129 任务,与 SmolLM `eval.slurm` 对齐,发表级)+ `early-signal`(debug,不作为对外声明) | **L1** 本地 NVMe `/home/ubuntu/lilei/projects/Incepedia/`(single source of truth)<br>**L2** NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/`(事件驱动 rsync 冷镜像)<br>**L3** git remote(<100MB 结构化产物兜底) | • **T2** sidecar:`scripts/tail_train_log_to_aim.py`(train.log → Aim)<br>• **T5** nanotron 原生直连 Aim / W&B 🟡 R1 完成后替代 sidecar |
| **P2 PoC**<br>⏳ R1/R2 结束后启动 | **【配置层(W1 产物)】**<br>• `configs/topics.yaml`(~90k,BISAC 34k + Wiki 2 层 + arXiv + MMLU 57 反锚)<br>• `configs/personas.yaml`(10k,**⑥** 由 PersonaHub 过滤)<br>• `configs/generators.yaml`(ADR 0003,midtier / frontier / critic)<br>• `configs/disallowed_openings.yaml`(**C14**,动态扩容)<br>• `configs/batches/v0.1_pilot.yaml`(**C12**,批配方完整声明)<br>• `data/seeds/*.parquet`(**⑤** 新增 OpenStax + Stack-Edu + arXiv full)<br><br>**【生成(W2–W5)】**<br>• 基础 16 条方针:**C1–C12**(seed grounding / persona / 难度 / 模板 / 异步 / audit ...)<br>• ★超越点:**C13** instruction reversal · **C14** 反单调 · **C15** rephrase 主路径 · **C16** synthetic 贯穿<br>• 质量升级 **①–⑦**(10B uniq / frontier 50% / Best-of-N / critic 50%×2 / 多源 seed / persona 10k / 模板 20)<br><br>**【后处理(W6)】**<br>Stage 1 MinHash dedup · Stage 2 BGE embed dedup · Stage 3 fineweb-edu classifier ≥3 · Stage 4 decontam(MMLU / MMLU-Pro / GSM8k / MATH / HumanEval / MBPP)<br><br>**【时间线】**W1 基建 5d → W2 pilot+G1/G2 → W3–W5 全量 **10B uniq** → ★W6 dedup+tokenize | **Track 1(从头训,ADR 0004)**<br>• **E1** 收尾(= R1+R2,基线本体)<br>• ★**E2** `exp_inc_v01_qwen3_seed{42,1337}`(10B uniq × ~3 ep × 30B × 2 seeds,Incepedia v0.1 主打)<br>• **E3** `exp_ctrl_cosmo_v2_subset3B_qwen3_seed42`(3B × 10 ep,剥离 epoch 效应)<br>• **E5** `exp_inc_v01_qwen3_ep{1,3,10}`(epoch 边际收益扫描,Phi-4 式)<br><br>**Track 2(共享 backbone + cooldown-fork,ADR 0004)**<br>• **R3** backbone 20B(一次性,~1.3 天)<br>• P2 批准 6 组**重生成 ablation**:**A1** seed-ground / **A2** rephrase path / **A5** 难度 / **A6** generator 家族 / **A8** STEM 配比 / **A10** reversal(各 2B uniq × 6B cooldown × 2 seeds)<br>• 4 组 post-filter 模拟(不重生成):**A3** persona / **A4** 模板 / **A7** critic / **A9** embed dedup | **双轨并行产出**<br>• Track 1 = E* 系列的全 benchmark 结果<br>• Track 2 = A* 系列的 Δ 分布<br><br>**门控 G(P2 流程开关,§3.7)**<br>• **G1** 人工抽检 10k(1 天,均分 ≥4.0)<br>• **G2** 自动化指标(edu-score 通过率 ≥85% / dup / decontam)<br>• **G3** 1.5B pilot × cooldown-fork × 2 seeds(18h,**规则:无论结果都进 G4**,避免 false-negative)<br>• ★**G4** 10B uniq × 30B × 2 seeds(= E2 本体,最终裁决:E2 ≥ E1 avg,任一 benchmark 显著胜) | (同 P1 的 L1 / L2 / L3)<br>+ `scripts/sync_to_nas.sh gen <batch_id>` 事件驱动备份生成数据<br>+ `scripts/sync_to_nas.sh hf_ckpt`(R1 产物上行) | (同 P1,期望 T5 已完成)<br>+ **生成侧实时监控**(C8 成功率 / 成本 / dup · C9 成本闸门,偏差 >20% 自动暂停) |
| **P3 扩容发布**<br>⏳ | • ★**v0.2 增强配方上线**:**C17** cross-doc / **C18** multi-turn / **C19** DSPy / **C20** RAG<br>• ★**⑩** 任务路由式生成器(本地 SmolLM2 + Vertex Batch + OR 三层)<br>• **⑧** multi-agent debate(expert 档)<br>• **⑨** 全局滚动 embed 索引<br>• uniq 10B → **30B v1.0 冻结**<br>• C1–C20 全开 | • ★**Protocol A 锚验证**(Llama2-1.82B,~5 天 GPU)→ 对齐 SmolLM 公开数字 ±0.5pp<br>• **Protocol B** v1.0 多 seed 主跑(30B uniq × N seeds)<br>• 同 P2 的 FA3 / HF 转换 / 补丁 基础设施延续 | • 跨多 benchmark + emergence 窗口<br>• 多 seed 均值 + 置信区间<br>• 外部复现包:HF model card + `configs/eval/` 可一键运行<br>• P3 仍复用 `cosmopedia-full` + 可能追加更重的 benchmark<br>• ★**A12-A15** 验证 C17-C20 各自贡献 | (同 P2)<br>+ HF Hub 公开发布层(数据集 / 模型 / tokenizer / model card) | (同 P2)<br>+ **W&B 云端跨机**(取代或补充本地 Aim,便于外部协作者旁观) |

### 2.2 关键衔接点(P1 → P2 → P3)

```
P1 ────────────────────────► P2 ─────────────────────────► P3
│                            │                             │
R1 (seed42 30B) ──┐          W1 基建 ──► W2 pilot+G1/G2 ──► ⑧⑨ 升级
R2 (seed1337 30B)─┤                           │            + Protocol A 锚
       │          │                           ▼            + 30B uniq v1.0
       │          └─► E1 基线入账              G3 (pilot    + HF Hub 发布
       │                                       cooldown)
       │                                        │ (规则:
       │                                        │  无论 pass/fail
       │                                        │  都进 G4)
       ▼                                        ▼
   T5 sidecar → 原生 Aim               W3–W5 全量 10B gen
   ADR 0008 FA3 可选 benchmark         W6 dedup+tokenize
   ADR 0009 HF 转换稳定                  │
                                          ▼
                                    E2 + E3 + E5 主训练
                                    A1/A2/A5/A6/A8/A10 fork
                                          │
                                          ▼
                                      ★ G4 裁决
                                    (E2 ≥ E1 avg?)
```

★ = 关键推进点 · G3 采 P2 规则:无论结果都进 G4。

---

## 3. 代号总表(按族分组,族内按编号排序)

### 3.1 项目阶段 · `P`

| 代号 | 全称 | 目标 | 成功标准 | 状态 |
|------|------|------|----------|------|
| **P1** | Phase 1 · Reproduce(复现) | 复现 Cosmopedia v2 基线 + 双协议 pipeline 验证 | Protocol A 对齐 SmolLM 公开数字 ±0.5pp;Protocol B 自建基线 | 🟡 40%(R1 在跑) |
| **P2** | Phase 2 · PoC(概念验证) | Incepedia v0.1(**10B uniq token**,质量优先稳健版)打平或胜 Cosmopedia | E1/E2/E3 出分 + A1/A2/A5/A6/A8/A10 完成 | ⏳ 待 R1/R2 结束后启动 |
| **P3** | Phase 3 · Iterate & scale | 10–30B Incepedia v1.0 + HF Hub 发布 + 论文 | 两轨跨 benchmark 显著超 Cosmopedia | ⏳ |

### 3.2 架构决策记录 · `ADR`

| 代号 | 主题 | 所在管线 | 权威来源 |
|------|------|----------|----------|
| **ADR 0001** | Three-layer storage defense(L1/L2/L3) | 基础设施 | `decisions/0001-three-layer-storage.md` |
| **ADR 0002** | Ablation 作为数据质量唯一指标(部分被 0004 扩充) | 方法论基石 | `decisions/0002-ablation-as-quality-metric.md` |
| **ADR 0003** | Multi-tier generator routing(midtier / frontier / critic) | **生成管线** | `decisions/0003-multi-tier-generator-routing.md` |
| **ADR 0004** | Dual-track adaptive evaluation(Track 1 + Track 2) | 评测管线 | `decisions/0004-evaluation-protocol-dual-track-adaptive.md` |
| **ADR 0005** | Project scope:standalone + seasoning 双场景 | 项目 scope | `decisions/0005-project-scope-standalone-plus-seasoning.md` |
| **ADR 0006** | Evaluation stack policy — pin to lighteval latest | 评测工具链 | `decisions/0006-evaluation-stack-policy.md` |
| **ADR 0007** | Dual-protocol architecture(Llama2-1.82B 锚 + Qwen3-1.7B 工作) | 训练架构 | `decisions/0007-dual-protocol-architecture.md` |
| **ADR 0008** | FlashAttention-3 可选 backend(默认 OFF) | 训练后端 | `decisions/0008-flash-attn-3-optional.md` |
| **ADR 0009** | nanotron → HF 转换必经之路 | train→eval 管线 | `decisions/0009-nanotron-to-hf-conversion.md` |

### 3.3 生成策略决策 · `C`(静态方针,不是实验)

| 代号 | 决策 | 分组 | 对应 ablation |
|------|------|------|---------------|
| **C1** | 强制 seed-grounding(Wikipedia/arXiv/FineWeb-Edu 段落 + 反幻觉指令) | 基础 | A1 |
| **C2** | Persona 注入(P2 起 10k,①–⑥) | 基础 | A3 |
| **C3** | 5 档难度(elementary → expert) | 基础 | A5 |
| **C4** | 结构模板(P2 起 20 种,①–⑦) | 基础 | A4 |
| **C5** | 双 OpenRouter key 轮换 | 生产线 | — |
| **C6** | `httpx.AsyncClient + aiolimiter + tenacity` | 生产线 | — |
| **C7** | 每 1000 条 Parquet + 原子重命名 | 生产线 | — |
| **C8** | 实时监控(成功率 / 成本 / dup 率) | 生产线 | — |
| **C9** | 成本闸门(偏差 >20% 自动暂停) | 生产线 | — |
| **C10** | Critic loop(P2 起 50% × 2 轮,①–④) | 生产线 | A7 |
| **C11** | 输出 schema(generator / seed_hash / critic_score / revision_of / ...) | 生产线 | — |
| **C12** | 每批合成由 `batch.yaml` 完全定义 | 生产线 | — |
| **C13** | **Instruction reversal**(Phi-4 方法,web→问答对) | 超越点 | A10 |
| **C14** | **风格反单调化**(`disallowed_openings.yaml` 动态扩容) | 超越点 | (事后 audit,非 ablation 变量) |
| **C15** | **Web rephrasing 主路径**(Nemotron/WRAP/Phi-4,40–50%) | 超越点 | A2 |
| **C16** | **Synthetic 贯穿 pretrain 全程**(Phi-4:多 epoch > 更多 uniq web) | 训练配方 | E2 / E3 / E5 联合验证 |
| **C17** | **Cross-document synthesis**(2-3 个同主题 seed → 一份综合教程) | 超越 FinePhrase | A12 |
| **C18** | **Multi-turn refinement**(outline → draft → refine,Phi-4 隐式采用) | 超越 FinePhrase | A13 |
| **C19** | **DSPy prompt 自动优化**(FinePhrase 明确列为 next) | 超越 FinePhrase | A14 |
| **C20** | **Retrieval-augmented grounding**(每个 seed 辅助 2-3 同主题文档作背景) | 超越 FinePhrase | A15 |

### 3.4 质量升级项 · `①–⑨`(P2 生成配方,在 C 基础上加强)

| 代号 | 升级 | P2 是否采纳 | 增量成本 |
|------|------|-------------|----------|
| **①** | uniq token 3B → **10B** | ✅ | +$6–15k(线性) |
| **②** | frontier tier 占比 16% → **50%** | ✅ | +$15–30k |
| **③** | **Best-of-N**(midtier=1,frontier Best-of-2,STEM/expert Best-of-3) | ✅ | +30% 该部分 |
| **④** | **critic 覆盖 15% → 50% × 最多 2 轮**,score<4 强制改到 ≥4 | ✅ | +$4–8k |
| **⑤** | seed 多源(+OpenStax + Stack-Edu + arXiv full-text) | ✅ | 一次性 1 天工程 |
| **⑥** | persona 2k → **10k**(PersonaHub 过滤子集) | ✅ | +$200 |
| **⑦** | 结构模板 12 → **20** | ✅ | $0 工程时间 |
| ⑧ | Multi-agent debate(expert 档 5-20%,C20 关联) | ⏳ v0.2 候选 | +$3–5k |
| ⑨ | 全局滚动 embed 索引(FAISS 1M 滑窗) | ⏳ v0.2 候选 | +2 GPU |
| **⑩** | **任务路由式生成器**(rephrase 用本地 SmolLM2,math/code/sci 用 frontier,critic 用 Opus)| ✅ **v0.2 锁定** | API 总额 **降 60%**(rephrase 不走 API)|
| ⑪ | **Cross-document synthesis(C17,5%)** | ⏳ v0.3 候选 | +$2-3k(frontier 生成)|
| ⑫ | **Multi-turn refinement(C18,frontier 30%)** | ⏳ v0.3 候选 | +30% frontier API 成本 ≈ +$8k |
| ⑬ | **DSPy prompt 优化(C19,W1 一次性)** | ⏳ v0.3 候选 | $200(扫 4 个 prompt 各 100 样本)|
| ⑭ | **Retrieval-augmented grounding(C20,20% 路径)** | ⏳ v0.3 候选 | input token +30%(成本 +5%)|

**v0.2 保守版(已锁定)**:只开启 C1–C16 + ①–⑩,扣掉所有 ⑪–⑭ 新 exploration。依据:C17–C20 暂无实验证据支撑,留 A12–A15 ablation 验证后再进 v0.3。

### 3.5 v0.1 实验组合 · `E`(Track 1,**P2 回答"v0.1 好不好"**)

**两阶段设计**:先在 Track 2(A11 ratio scan,见 §3.6)用 cooldown-fork 快速扫 mix ratio,**选出 30 / 50 中最优档** → 再在 Track 1 跑 **E2-100(纯独立)+ E2-mix(最优 mix)+ E3(epoch 控制)**。省 2 次 Track 1 run ≈ 4 天 GPU。

| 代号 | 实验 ID | 数据(uniq + mix) | 训练 token | epochs(Inc 部分) | 对比 | 回答问题 |
|------|---------|-------------------|------------|-------------------|------|----------|
| **E1** | `exp_ref_cosmopedia_v2_qwen3_seed{42,1337}` | Cosmopedia v2 全量(28B uniq) | 30B | ~1.07 | 基线本体 | 基线分数 |
| **E2-100** | `exp_inc_v01_100_qwen3_seed{42,1337}` | **100% Incepedia v0.1**(10B uniq) | 30B | **3** | vs E1 | "Incepedia 能独当一面?"(**野心叙事**)|
| **E2-mix** | `exp_inc_v01_mix{30或50}_qwen3_seed{42,1337}` | **Inc 10B + FineWeb-Edu-HQ mix-in**(比例由 A11 scan 决定) | 30B | 1–2 | vs E1 / vs FinePhrase | "Incepedia 作为 SOTA 式 mix-in 赢?"(**SOTA 可比**)|
| **E3** | `exp_ctrl_cosmo_v2_subset3B_qwen3_seed42` | Cosmopedia v2 随机 3B subset(100%) | 30B | 10 | vs E2-100 的 3B×10ep 变体 | **单独数据质量**增量(剥离 epoch) |
| ~~E4~~ | ~~(已删,与 E1 冗余)~~ | | | | | |
| **E5** | `exp_inc_v01_qwen3_ep{1,3,10}` | Incepedia v0.1(100%,10B uniq) | 10 / 30 / 100B | 1 / 3 / 10 | 内部扫描 | epoch 边际收益曲线(Phi-4 式,事后可选) |

**mix-in 数据选择**:默认 **FineWeb-Edu-HQ**(已 tokenize,零阻塞)。W3–W5 期间**后台并行** tokenize DCLM,若 W6 前就绪则 E2-mix 改用 DCLM(对齐 FinePhrase 结论:DCLM 作 HQ 源的 mix-in 略优于 FW-Edu-HQ)。

### 3.5.1 v0.2 保守版 · path_mix 配比(**已锁定**,2026-04-24)

**设计原则**:只用有明确文献证据的路径,扣掉所有未验证的 exploration(Cosmopedia-style / C17-C20)。projected 21B 分数 **~0.18(FinePhrase +6%)**,置信度高。激进版(+C17-C20 + Cosmopedia-style)留 v0.3 做。

| # | 路径 | 占比 | 量(10B)| 生成器 | 证据 |
|---|------|-----|--------|--------|------|
| 1 | rephrase_as_tutorial | 15.0% | 1.50B | 本地 SmolLM2 | FinePhrase 4 赢家 |
| 2 | rephrase_as_faq | 15.0% | 1.50B | 本地 SmolLM2 | FinePhrase 4 赢家 |
| 3 | rephrase_as_table | 15.0% | 1.50B | 本地 SmolLM2 | FinePhrase 4 赢家(ARC 冠军) |
| 4 | rephrase_as_math_word_problem | 15.0% | 1.50B | 本地 SmolLM2 | FinePhrase 4 赢家(GSM8K 冠军) |
| 5 | math_worked_example(凭空 + 解)| 10.0% | 1.00B | Vertex Gemini 2.5 Pro / DeepSeek-R1 | Phi-4 实证 |
| 6 | code_tutorial(凭空)| 10.0% | 1.00B | Vertex Claude 4 Sonnet / Qwen3-Coder | Phi-4 / DeepSeek-Coder 实证 |
| 7 | scientific_reasoning(长链 CoT) | 5.0% | 0.50B | Vertex DeepSeek-R1 / o1 / Gemini Pro | 长链推理必须 frontier |
| 8 | instruction_reversal(C13) | 15.0% | 1.50B | Vertex Claude 4 Sonnet / GPT-4o | Phi-4 明确 10-15% 占比 |
| **合计** | | **100.0%** | **10B** | | |

**扁平求和核验**:15 × 4 + 10 × 2 + 5 + 15 = **60 + 20 + 5 + 15 = 100% ✓**

**生成成本**(Vertex Batch 50% off · 本地 vLLM $0):
- 路径 1-4(60%,6B):**本地 SmolLM2 + 8×H100 vLLM,$0 API**,~2 天
- 路径 5-7(25%,2.5B):**Vertex Batch · Gemini Pro / Claude Sonnet / DeepSeek-R1**,~$20k,~48h
- 路径 8(15%,1.5B):**Vertex Batch · Claude Sonnet + GPT-4o**,~$11k,~48h
- **总计 ~$31k + 0 GPU 电费,3-4 天 calendar**(并行)

**比 v0.2 激进版省 $45k + 3 个新 C 决策风险**,代价:projected 分数 0.18 vs 0.19(差 +1pp 置信期望值)。

### 3.6 单变量消融 · `A`(Track 2,**P2 回答"哪条 C 决策值多少分"**)

> **⚠️ 本表是 P2 批准的规范版**,与 `docs/incepedia-overview.md §7.3` 的早期 A 表编号冲突,以下为准。overview.md 保留作历史。

| 代号 | 变量 | Control | Treatment | 对应 C | 需重生成? | 预期 Δ | **P2 批准** |
|------|------|---------|-----------|--------|------------|--------|-------------|
| **A1** | Seed grounding | 无 seed | web seed + 反幻觉(C1) | C1 | **是** | +0.8–1.5pp | ✅ |
| **A2** | Path 主路径 | 100% textbook_from_web(Cosmopedia 式) | rephrase_* 占 50%(C15) | C15 | **是** | +0.5–1.0pp | ✅ |
| A3 | Persona pool | 无 | 10k persona(C2) | C2 | 否 | +0.3–0.6pp | (post-filter 模拟) |
| A4 | 结构模板数 | 1 | 20(C4) | C4 | 否 | +0.3–0.5pp | (post-filter 模拟) |
| **A5** | 难度分层 | 100% college | 5 档均匀(C3) | C3 | **是** | CSR +0.4 / MMLU-STEM −0.2 | ✅ |
| **A6** | 生成器家族 | 100% DeepSeek-V3 | 5 家族按 tier(ADR 0003) | ADR 0003 | **是** | +0.2–0.5pp | ✅ |
| A7 | Critic loop | 关 | 50% × 2 轮(C10,④) | C10 | 否 | +0.3–0.7pp STEM | (post-filter 模拟) |
| **A8** | STEM 配比 | 30% | 60% | topics.ratios | **是** | MMLU-STEM +1–2pp / HellaSwag −0.3 | ✅ |
| A9 | Embed dedup | 只 minhash | +BGE cos>0.92 | — | 否 | +0.2–0.4pp | (post-filter 模拟) |
| **A10** | Instruction reversal | 关 | 10% 占比(C13) | C13 | **是** | +0.5pp MMLU | ✅ |
| **A11** | **Mix-ratio scan**(驱动 E2-mix 档位决策) | 100% Inc | **30% 或 50% Inc + 余 FW-Edu-HQ** | — | 否(复用主批 10B)| 扫出最优 ratio | 🆕 **决策性,必跑** |
| **A12** | **Cross-document synthesis** | 关(单 seed)| 5% 路径走多 seed 融合(C17) | C17 | **是**(2B uniq 重生成)| +0.3-0.5pp 知识连贯 | 🆕 v0.3 |
| **A13** | **Multi-turn refinement** | 单 pass | frontier 30% 走 outline→draft→refine(C18) | C18 | **是**(frontier 部分重生成)| +0.2-0.4pp | 🆕 v0.3 |
| **A14** | **DSPy 自动 prompt 优化** | 手写 prompt | DSPy 扫出的优化版 prompt(C19) | C19 | **是**(全 rephrase 重生成)| +0.1-0.3pp | 🆕 v0.3 |
| **A15** | **RAG 多 doc grounding** | 单 seed | 2-3 同主题 doc 拼成上下文(C20) | C20 | **是**(20% 路径重生成)| +0.2-0.4pp | 🆕 v0.3 |

✅ = P2 批准的 6 组**重生成 ablation**(成本 ~$40–60k,GPU ~15 天)。其余 4 组可在主批 10B uniq 数据上 post-filter 模拟,不需另出 API 钱。

🆕 **A11 特殊**:它不是事后报告 Δ 的 validation ablation,而是 **E2-mix 启动前的决策扫描**。3 档(100 / 50 / 30)× seed42 only × 6B cooldown-fork = **27 GPU-h ≈ 1.1 天**,结果直接决定 E2-mix 的 ratio 取值。

### 3.7 go/no-go 门控 · `G`(P2 流程开关)

| 代号 | 门 | 规模 | 训练? | 用时 | 回答 | 通过条件 |
|------|----|----|-------|------|------|----------|
| **G1** | 人工抽检 + 分布统计 | 10k 条(~$30) | 否 | 1 天 | 首句 top-20<15%、refusal<1%、schema 合法 | 人工均分 ≥4.0 |
| **G2** | 自动化指标 | 同 G1 | 否 | 0.5 天 | edu-score≥3 比、BGE dedup 率、decontam 命中 | edu-score 通过率 ≥85% |
| **G3** | Track 2 cooldown-fork | **1.5B uniq pilot** × 6B cooldown × 2 seeds | 是 | 18h | CSR 相对排序、MMLU 方向 | **P2 规则:无论结果都进 G4**(避免 false-negative) |
| **G4** | Track 1 full(= E2 本体,两档 E2-100 + E2-mix)| **10B uniq** × 30B × 2 seeds × **2 档** = **4 runs** | 是 | 8 天 | 全 benchmark,含 emergence 窗口 | P2 收尾:E2-100 或 E2-mix 任一 ≥ E1 avg,任一 benchmark 显著胜 |

### 3.8 6 周实施周次 · `W`(P2 时间线)

| 代号 | 任务 | 产物 | 依赖 |
|------|------|------|------|
| **W1** | 基建 + topic/persona/seed 数据层 | `src/incepedia/generation/` 16 模块 + `configs/batches/v0.1_pilot.yaml` + 10k persona + 90k topic + 5 类 seed | R1/R2 不阻塞 |
| **W2** | pilot 1.5B uniq + G1/G2/G3 | `data/raw_generations/batch_pilot/*.parquet` + G3 fork 分数 | W1 |
| **W3–W5** | v0.1 **10B uniq** 全量生成(+ **后台** tokenize DCLM 作备用 mix-in)| `data/raw_generations/batch_v0.1/*.parquet`(~4M 条)+ 可选 `data/datasets/dclm_qwen/*.ds` | G3 过(**规则**:无论结果) |
| **W6** | dedup + tokenize → **A11 ratio scan**(1.1d)→ 启动 **E2-100 + E2-mix + E3**(G4)| `data/datasets/incepedia_v01_qwen/*.ds` + `fork_inc_v01_ratio{100,50,30}_*/` + `exp_inc_v01_{100,mix}_*/` + `exp_ctrl_cosmo_v2_subset3B_*/` | W5 |

### 3.9 存储层 · `L`(ADR 0001)

| 代号 | 位置 | 内容 |
|------|------|------|
| **L1** | 本地 NVMe `/home/ubuntu/lilei/projects/Incepedia/` | single source of truth |
| **L2** | NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/` | 事件驱动 rsync 冷镜像 |
| **L3** | git remote | <100MB 结构化产物兜底 |

### 3.10 架构协议 · `Protocol`(ADR 0007)

| 代号 | 架构 | 参数 | Tokenizer | 项目频率 | 用途 |
|------|------|------|-----------|----------|------|
| **Protocol A** | Llama2(全注意力 / RoPE θ=1e4 / 无 bias / 24 层) | 1.82B | Mistral-7B-v0.1(32k) | **仅 2 次** | 外部锚,对齐 SmolLM/Cosmopedia 公开数字 |
| **Protocol B** | Qwen3-style(GQA 8 KV / RoPE θ=1e6 / QKV bias / 24 层) | 1.7B | Qwen 151k | ~40 次 | 工作架构,所有 P2 主线 ablation |
| **Protocol C** | Qwen2-style(GQA 8 KV / RoPE θ=**1e4** / **无 bias** / **28 层** / intermediate **6144**) | 1.7B | hynky/Llama-3.2-1B-no-bos(128k) | 2-3 次 | **FinePhrase 论文严格复刻**,Figure 1 直接对比;WSD schedule 1%/89%/10%,seq=4096,GBS=512×4096 |

### 3.11 评测轨道 · `Track`(ADR 0004)

| 代号 | 协议 | 回答 | 单次墙钟(Qwen3) |
|------|------|------|-------------------|
| **Track 1** | 从头训 × 30B × 2 seeds | "独立 corpus 能力" | ~4.0 天 |
| **Track 2** | 共享 backbone 20B + cooldown-fork 6B × 2 seeds | "调料能力" | backbone 1.3 天(一次性)+ fork 18h |

### 3.12 训练运行 · `R`(对话指代)

| 代号 | 含义 | 状态 |
|------|------|------|
| **R0** | seed42 第 1 次(有 launcher bug,只训 1.5B,算作 sanity) | ❌ 已弃 |
| **R1** | seed42 第 2 次 30B 正式训练(= E1 一半,Protocol B) | ❌ **崩溃于 step 20931 / 91%**(2026-04-24 03:07 UTC),**项目决定丢弃**;不补救 |
| **R2** | seed1337 30B(= E1 另一半,Protocol B) | ❌ **未启动,随 R1 一起暂搁** |
| **R-C-FP** | **FinePhrase 复刻 · Protocol C × 21B from-scratch × (FinePhrase 50% + FW-Edu-HQ 50%)** | 🟢 **进行中**(2026-04-24 起,ETA ~33h)|
| **R3+** | 按 exp_id 逐个排 | ⏳ |

### 3.13 工程 TODO · `T`(`docs/project-status.md`)

| 代号 | 任务 | 状态 |
|------|------|------|
| **T1** | FA3 switch | ✅ 实现完成(benchmark 待跑) |
| **T2** | nanotron → Aim sidecar | ✅ 完成 |
| **T3** | patches → bootstrap_env.sh | ✅ 完成 |
| **T4** | `scripts/lint_nanotron_yaml.py` | ✅ 完成 |
| **T5** | nanotron 原生直连 Aim/W&B(替代 sidecar) | 🟡 R1 完成后做 |

---

## 4. 生成管线详解(P2 主管线的"剖面图")

```
【W1 · 一次性数据准备】────────────────────────────────────────────────────
                                                                            
  HuggingFace / 本地 reference                 本地处理                      
  ─────────────────────────────              ─────────────                   
  FineWeb-Edu dedup (本地 72GB)       ──►  scripts/prepare_seeds.py          
  Wikipedia (HF wikimedia/wikipedia)  ──►    采样 300–800 tok 段落           
  arXiv full (HF ccdv/arxiv)          ──►    按 edu-score 加权               
  OpenStax CC (⑤ 新增)                ──►  data/seeds/*.parquet             
  Stack-Edu (⑤ 新增)                  ──►                                   
                                                                            
  BISAC 34k + Wikipedia 2 层 + arXiv  ──►  scripts/build_topic_tree.py       
    + MMLU 57 反向锚点                      configs/topics.yaml (~90k topic)
                                                                            
  PersonaHub (HF tencent-ailab)       ──►  scripts/import_persona_hub.py     
    → LLM 过滤到 10k 均匀子集 (⑥)           configs/personas.yaml (10k)     
                                                                            
  configs/generators.yaml (ADR 0003)         (已存在,无改动)                
  configs/disallowed_openings.yaml           (手工种 30 条 + W3-W5 自动扩)  
  configs/batches/v0.1_pilot.yaml (C12)      (本批配方完整声明)             

【W2–W5 · 每条样本的实时流水线】────────────────────────────────────────────

       ┌────────────────────── 9 维度采样(PromptAssembler)─────────────────┐
       │                                                                    │
       │   seed_source  topic  persona  difficulty  structure               │
       │       │         │       │         │           │                   │
       │   seed_text     │       │         │           │                   │
       │       ▼         ▼       ▼         ▼           ▼                   │
       │   path ∈ {rephrase_*×4 (C15), textbook_from_web, wikihow,          │
       │           story, math_worked_example, code_tutorial,                │
       │           scientific_reasoning, instruction_reversal (C13),         │
       │           multi_agent_debate (⑧ 留 v0.2) }                         │
       │       │                                                             │
       │   anti_mono constraints(注入 disallowed_openings 当前 list,C14)   │
       │       │                                                             │
       └───────┼─────────────────────────────────────────────────────────────┘
               ▼
       ┌────────────────────── Router(ADR 0003)─────────────────────┐
       │   task_map → tier ∈ {midtier_bulk, frontier_reasoning, critic}│
       │   tier 内按权重抽 model                                       │
       │   ②:frontier 占比从 16% 提到 50%                              │
       └───────┬───────────────────────────────────────────────────────┘
               ▼
       ┌──────── OpenRouter AsyncClient(C5/C6)─────────┐
       │   双 key 轮换 + aiolimiter + tenacity          │
       │   per-model rate-limit header 感知              │
       └───────┬────────────────────────────────────────┘
               ▼
       ┌──────── Best-of-N(③)───────────────────────────┐
       │   midtier:  N=1                                  │
       │   frontier: N=2  → critic 打分挑最佳              │
       │   STEM/expert: N=3                               │
       └───────┬──────────────────────────────────────────┘
               ▼
       ┌──────── Critic loop(C10,④)────────────────────┐
       │   trigger: task ∈ {math,code,sci,reversal}       │
       │   coverage 50%;最多 2 轮重写                      │
       │   score<4 必改;flag revision_of 血缘             │
       └───────┬──────────────────────────────────────────┘
               ▼
       ┌──────── Post-filter ─────────────────────────────┐
       │   language == en                                  │
       │   length ∈ [200, 3000]                            │
       │   拒答检测                                        │
       │   首句 ∉ disallowed_openings(C14)                │
       └───────┬──────────────────────────────────────────┘
               ▼
       ┌──────── Quality classifier ──────────────────────┐
       │   HuggingFaceFW/fineweb-edu-classifier           │
       │   score ≥ 3 才接受                                │
       └───────┬──────────────────────────────────────────┘
               ▼
       ┌──────── 批内实时 embed dup check ────────────────┐
       │   BGE-small 滚动 10k 索引,cos > 0.95 拒绝         │
       │   (全局 1M 滑窗 ⑨ 留 v0.2)                       │
       └───────┬──────────────────────────────────────────┘
               ▼
       ┌──────── Writer(C7)+ Audit(C8/C14)────────────┐
       │   每 1000 条 Parquet + 原子重命名               │
       │   每 shard 统计成功率 / cost / dup              │
       │   每 1000 条采 50 条统计首句 top-20 → 回写 C14  │
       │   每 10k 条对账成本(C9,偏差>20% 暂停)         │
       └───────┬──────────────────────────────────────────┘
               ▼
       data/raw_generations/batch_<batch_id>/*.parquet
               │
               ├──► sync_to_nas.sh gen <batch_id>(事件驱动)
               │
               ▼
       【W6 · 批量后处理】(scripts/dedup_and_filter.py)
               │
               ├── Stage 1: MinHash dedup(n-gram=5, threshold=0.7,datatrove)
               ├── Stage 2: Embed dedup(BGE-small,cos>0.92)
               ├── Stage 3: Classifier filter(已在生成时做,此处复核)
               └── Stage 4: Decontamination
                     10-gram + 0.5 ratio,覆盖
                     MMLU / MMLU-Pro / GSM8k / MATH / HumanEval / MBPP
               │
               ▼
       data/datasets/incepedia_v01_qwen/*.ds(datatrove → nanotron shard)
               │
               ├──► A11: fork_inc_v01_ratio{100,50,30}_qwen3_seed42 (Track 2 ratio scan, 1.1d)
               │         │
               │         ▼  挑出最优档 (30 或 50)
               │
               ├──► E2-100: exp_inc_v01_100_qwen3_seed{42,1337}     (G4,Track 1)
               ├──► E2-mix: exp_inc_v01_mix{best}_qwen3_seed{42,1337}(G4,Track 1,mix-in = FW-Edu-HQ 或 DCLM)
               ├──► E3:     exp_ctrl_cosmo_v2_subset3B_qwen3_seed42
               ├──► E5:     exp_inc_v01_qwen3_ep{1,3,10}(Incepedia 自己 epoch 扫描,事后可选)
               └──► A1/A2/A5/A6/A8/A10 分别生成各自 2B uniq,挂 cooldown-fork
```

---

## 5. 按"我想问什么"反查

### "现在 R1 是什么?"
seed42 第 2 次 30B 正式训练,当前 40%。R1 + R2(seed1337)合起来构成 **E1**。

### "P1 / P2 / P3 怎么衔接?"
- P1 = R1/R2 出分 → E1 入账
- P2 = W1 基建 → W2 pilot + G3 → W3–W5 生成 10B → W6 训 E2/E3 = G4 决策
- P3 = E2 胜出 → 放大到 30B uniq v1.0 + ⑧⑨ 升级 + Protocol A 锚验证

### "C / E / A 之间什么关系?"
- **C** = 静态方针(我们决定这么生成,16 条)
- **E** = Track 1 实验(v0.1 整体好不好,5 组)
- **A** = Track 2 ablation(每条 C 值多少分,10 组;P2 批准 6 组重生成)
- **G** = 门控,把以上流程分阶段卡住

### "C13 跟 A10 是一样的吗?"
不一样。**C13 是我们做 instruction reversal 这件事**(生成配方里 10% 走这条路径),**A10 是验证 C13 值多少分**(关闭 reversal vs 开启,对比 Δ benchmark)。C13 是因,A10 是证据。C1↔A1、C2↔A3、C3↔A5、C4↔A4、C10↔A7、C15↔A2 同理。

### "①–⑨ 跟 C 什么关系?"
① 调 uniq 总量、② 调 frontier 占比、③ 调 Best-of-N、④ 调 critic 覆盖与轮数、⑤ 调 seed 来源、⑥ 调 persona 规模、⑦ 调 结构数量 —— **都是在现有 C 决策上调参数,不是新策略**。C 是定性,①–⑨ 是定量。

### "G3 打输怎么办?"
P2 规则:**无论 G3 结果都进 G4**,避免 cooldown-fork 的 false-negative(Hägele et al. 报告相关系数 0.88,小 gap 有翻盘概率)。G3 只是"早期信号",G4 才是最终裁决。

### "E2 要训多久?"
10B uniq × 30B 训练 × 2 seeds × Qwen3-1.7B ≈ **4 天**(8×H100,对齐 R1 实测 45–55h/seed × 2 seeds)。加上 E3(3B × 10ep × 1 seed,~2 天)= W6 约 6 天 GPU。

### "W1–W6 跟 R1 冲突吗?"
不冲突。W1 是纯工程 + 数据准备 + 少量 API 开销,W2 pilot 的 G3 fork 需要 backbone(需要 R3 先跑,~1.3 天),所以**顺序**:R1(主训练机)→ R2 → R3(backbone)→ W2 fork。W1 本身不抢 GPU,可与 R1/R2 并行做。

### "L1 L2 L3 和 P1 P2 P3 冲突吗?"
完全不同两个轴。**L** = Layer(存储层),**P** = Phase(时间)。

### "ADR 0007 和 Protocol A/B 是一回事?"
ADR 0007 是**决策文档**,Protocol A/B 是**决策产物**。读 ADR 0007 即理解 Protocol A/B。

### "我要新加一条生成策略 C17,怎么办?"
见下面维护约定。

---

## 6. 维护约定

- **新增 ADR** → `docs/decisions/0010-*.md` + 更新 `decisions/README.md` 索引 + 本表 §3.2。
- **新增 C 策略(C21+)** → 加到 `docs/methodology.md §3` + 本表 §3.3。
- **新增 E 实验(E6+)** → 加到本表 §3.5 + 创建 `experiments/exp_*/config.yaml`。
- **新增 A 消融(A16+)** → 加到本表 §3.6 + 对应 C 决策在 §3.3 的"对应 ablation"列更新。
- **新增 G 门(G5+)** → 加到本表 §3.7,写明规模 / 训练与否 / 通过条件。
- **新增 ⑮ 升级项** → 加到本表 §3.4,在对应 W 周次里安排落地。
- **新增 TODO(T6+)** → `docs/project-status.md § TODO 队列` + 本表 §3.13。
- **阶段推进** → 更新 `README.md` 首屏「当前阶段」+ `docs/project-status.md` 变更历史 + 本表 §3.1 状态列。
- **A1–A15 编号歧义**:本表 §3.6 是规范版,`docs/incepedia-overview.md §7.3` 为历史版,不再同步更新,但保留原文。

---

© 2026 · Incepedia contributors · Apache-2.0
