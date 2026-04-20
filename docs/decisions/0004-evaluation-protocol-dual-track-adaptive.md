# 0004 · Dual-track adaptive evaluation protocol

- **Status**: accepted
- **Date**: 2026-04-20
- **Deciders**: bill-lee-mk
- **Supersedes**: `0002-ablation-as-quality-metric.md`(部分——评测指标不变,训练协议从"纯 1.82B × 30B standalone"扩为"双轨")

- **Context**:
  - ADR 0002 把"ablation 分数"锁定为唯一数据质量判据,但当时隐含假设是 Cosmo-1B 年代(2024-02)的协议:**纯数据集 × 1.82B × 30B token**
  - 核查 SmolLM2 论文(arXiv:2502.02737)发现,2025 版 Cosmopedia v2 **从不作为独立 corpus**——它只在 stage 4 decay 阶段占 4%(约 40B/11T token)。用老协议评它,**评的问题和作者想证明的不一致**
  - 本项目目标(ADR 0005)是 Incepedia 同时具备"独立 corpus"和"decay 调料"两种能力,单协议无法覆盖
  - Proxy 小模型(200M)方案被否决:scaling 转手不可靠,项目初期错误方向代价太大

- **Decision**:
  采用 **双轨 + 自适应选择** 的评测协议。

  ### Track 1 · Standalone(独立 corpus 能力)
  - 协议:**1.82B Llama2 从头训 × 30B tokens × 2 seeds**,纯数据集(零 backbone)
  - 单次墙钟:~2.1 天(8×H100)× 2 seeds = ~4.2 天
  - Milestone scale-up:1.82B × 100B × 1 seed = ~7 天
  - 适用问题:"数据集 X 作为独立 pretraining corpus,和数据集 Y 相比如何?"
  - Cosmopedia-v1 / Cosmo-1B / FineWeb ablation 协议的直接继承

  ### Track 2 · Seasoning(decay 调料能力)
  - 协议:**共享 backbone + cooldown-fork**
    - Backbone:1.82B Llama2 × 20B tokens FineWeb-Edu(trapezoidal LR,warmup+stable 阶段结束保存 ckpt)—— **一次性投入,之后所有 fork 共用**
    - Fork:从 backbone ckpt 分叉,LR 从 peak 线性衰减到 0,喂 6B tokens 的测试数据集
  - 单次墙钟:backbone 1.4 天(一次);每 fork ~10 h(1 seed)/ ~20 h(2 seeds)
  - 适用问题:"在 SmolLM2 式 decay 阶段,用数据集 X 精调,和用 Y 精调相比如何?"
  - SmolLM2 论文评判 Cosmopedia v2 的真实协议

  ### 选择规则(自适应,非配额)
  每个 ablation 开始前问:"这个实验想答的是独立能力还是调料能力?"
  - 独立能力 / 版本级 milestone → **Track 1**
  - 参数微调 / 配比扫描 / 单变量 ablation → **Track 2**
  - **不设前置配额**,事后盘点比例作为项目记录
  - 预估自然分布:Track 1 ~15%、Track 2 ~85%(约 6 + 33 ≈ 40 次 ablation)

- **Alternatives considered**:
  1. **只 Track 1**:独立能力证据充分,但完全不测 SmolLM2 式配料场景,与 ADR 0005 项目目标不匹配
  2. **只 Track 2**:配料证据充分,但答不到"Incepedia 作为独立 corpus 是否比 Cosmopedia 强",和老 Cosmo-1B 路线失联
  3. **Proxy 200M / 400M Llama2 小模型**:scaling 转手到 1.82B 不可靠,项目初期需要最高保真度信号,否决
  4. **固定配额(15%/85%、30%/70% 等)**:rigid,不如"按问题选"自然

- **Consequences**:
  - **Positive**:
    - 两个科学问题都有一手 1.82B 信号
    - Track 2 单次 10 h 支持日内迭代节奏
    - 自适应避免了"为凑配额硬跑"的浪费
    - Backbone 一次性投入摊销到 ~33 次 fork,效率最高
  - **Negative / trade-offs**:
    - Backbone ckpt 成为单点依赖资产,必须备份到 NAS 双副本
    - 维护两条训练/评测流水线,代码复杂度 +1
    - Track 1 / Track 2 分数不可直接数值比较(语义不同),报告时必须分别标注
  - **Follow-ups**:
    - 实现 `src/incepedia/training/backbone.py`(20B FineWeb-Edu 训练)
    - 实现 `src/incepedia/training/cooldown_fork.py`(从 ckpt 分叉做 6B cooldown)
    - 实现 `src/incepedia/training/standalone.py`(纯数据集 × 30B)
    - Backbone ckpt(~7 GB)生成后立即双副本(local NVMe + NAS)
    - 每次新 ablation 的 `config.yaml` 必须带 `track: 1|2` 字段

- **Tokens / 成本归纳**(8×H100 单节点,3 个月项目)
  | 项 | Tokens | 墙钟 |
  |---|---|---|
  | Track 1 (6 milestone runs) | 280B | ~28 天 |
  | Track 2 (1 backbone + 33 fork) | ~212B | ~16 天 |
  | 训练合计 | ~492B | ~44 天 |

- **Related**:
  - ADR 0002(评测指标 → 超集:仍用同一套 lighteval 任务)
  - ADR 0005(项目 scope → 双轨协议恰好覆盖双 scope)
  - `docs/methodology.md` §2
  - Hägele et al. 2024(arXiv:2405.18392)——trapezoidal cooldown-fork 可替代从头训的证据
  - SmolLM2 paper(arXiv:2502.02737)stage 4 decay 协议
