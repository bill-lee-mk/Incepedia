# 0007 · Dual-protocol architecture — Llama2-1.82B (ref anchor) + Qwen3-1.7B (working)

- **Status**: accepted
- **Date**: 2026-04-21
- **Deciders**: bill-lee-mk
- **Extends**: ADR 0004(双轨评测协议)—— 0004 定义 Track 1/Track 2 训练协议;0007 在每条 track 内引入第二个架构维度

- **Context**:
  - ADR 0004 规定 ablation 训练用 1.82B Llama2 架构,这继承自 FineWeb / Cosmopedia / SmolLM 的 2024 标准
  - 2026 时点,Llama2 是 2-年-旧的架构;现代主流(Qwen3 / Llama-3 / Gemma-3)使用 GQA、改良 RoPE、更优 init,**有更高的下游 benchmark headroom**
  - 用老架构跑 ablation,有可能让 Incepedia 的数据质量优势在 benchmark 上"被压扁"(MMLU 已贴近 random,GSM8k 几乎全 0)
  - 但完全抛弃 Llama2 又会失去与 SmolLM blog / FineWeb 论文公开数字的"外部锚",外部可信度受损
  - 评估候选:Llama-3.2-1B / Qwen3-1.7B / OLMo-2-1B / Gemma-3-1B
    - **Qwen3-1.7B** 综合评分最高:2026 社区相关性 + 信号强度 + 工程成本可接受
    - 工程成本 = nanotron 半天 patch(QKV bias / RoPE θ / GQA),其它框架(TorchTitan / Megatron / OLMo-core)代价更大

- **Decision**:
  采用**双协议**(dual-protocol)训练 + 评测策略:

  ### Protocol A · Llama2-1.82B(reference anchor)
  - **架构**:Llama2,hidden 2048 / layers 24 / heads 16 / 全注意力 / RoPE θ=10000 / 无 bias
  - **训练框架**:`nanotron`(原生支持)
  - **Tokenizer**:`mistralai/Mistral-7B-v0.1`(32k vocab)
  - **作用**:仅作"信任锚"——验证我方 pipeline 在已知公开协议上能复现 SmolLM/FineWeb 公布数字 ±0.5pp
  - **频率**:全项目 **2 次**(Cosmopedia v2 baseline × 2 seeds),不再扩展到任何 Incepedia 版本

  ### Protocol B · Qwen3-style 1.7B(working architecture)
  - **架构**:Qwen3-style,hidden 2048 / layers 24 / heads 16 / GQA(8 KV heads)/ MLP intermediate 8192 / RoPE θ=1000000 / **QKV bias=True**
  - **训练框架**:`nanotron`,**直接用内置的 `Qwen2Config` / `Qwen2Model`**(实测发现 nanotron 已原生支持 Qwen 架构含 `attention_bias`,无需写 patch)。`src/incepedia/training/nanotron_qwen.py` 只保留 spec 字典 + `is_qwen2_config: True` 触发内置 Qwen2Model。
  - **Tokenizer**:**用 `Qwen/Qwen2.5-1.5B` 原生 tokenizer(151k BBPE vocab)**。原计划沿用 Mistral 已在 commit `c427011` 落地,但用户决策后改为"各架构用自己的 tokenizer"让 Qwen3 发挥其 vocab 设计。**代价**:Cosmopedia v2 + FineWeb-Edu 需用 Qwen tokenizer 重 tokenize(~10 min 后台任务);**收益**:Qwen 架构在自己的 tokenizer 上的 benchmark 上限更高,且 vocab embedding 层不会成为"伪学习对象"。
  - **作用**:所有日常 ablation + Incepedia 版本演进 + 最终发布数字
  - **频率**:Track 1 milestone × 6 + Track 2 backbone(1)+ Track 2 forks(~33)

  ### 双协议组合的 narrative
  ```
  Cosmopedia v2 baseline:  Protocol A = X     (matches SmolLM blog ±0.5pp ✅ pipeline 可信)
                           Protocol B = Y     (我方自建基线)
  Incepedia v1.0:          Protocol A = X+Δa  (+Δa pp on standard 老协议)
                           Protocol B = Y+Δb  (+Δb pp on modern 协议)
  → "Incepedia > Cosmopedia 在两个不同架构下均成立"
  ```

  ### 不变项
  - **训练框架**:两协议同用 nanotron(Llama2 用内置 LlamaConfig,Qwen3 用内置 Qwen2Config)
  - **评测**:两协议同用 lighteval + 我方 port(`src/incepedia/eval/lighteval_tasks.py`)
  - **数据内容**:Cosmopedia v2 + FineWeb-Edu 的源数据完全相同,只是 tokenize 到两种 vocab
  - **训练超参**:lr / global batch / seeds / seq_len / schedule 两协议严格同设置
  - **评测口径**:两协议的 lighteval 任务 / metric / few-shot 数 / sample 数完全相同

  ### 变项(按 ADR 决策)
  - **架构**:Llama2-1.82B vs Qwen3-style 1.7B(本 ADR 主要决策)
  - **Tokenizer**:Mistral-7B-v0.1 vs Qwen/Qwen2.5-1.5B(子决策,2026-04-21 更新)
    - 架构 + 对应 native tokenizer 成对变化,delta 解释为"整体架构包"效应
    - 好处:每个架构发挥设计最优
    - 代价:delta 不能拆解为"纯架构"与"纯 tokenizer"的贡献,但这个细分在我们的科学问题下并不关键(我们只关心 Incepedia 是否在每种部署组合下都胜过 Cosmopedia)

- **Alternatives considered**:
  1. **单 Llama2-1.82B**(原方案,ADR 0004 默认):稳定可对齐,但 narrative 偏弱,modern 相关性低。否决。
  2. **单 Qwen3-1.7B**:modern 但失外部锚,Cosmopedia 基线无法对齐 SmolLM,可信度需自建。否决。
  3. **Llama2 ref + Llama-3.2-1B 主力**:工程最简(0 patch),但 Llama-3.2-1B 信号弱于 Qwen3,2026 社区相关性低于 Qwen3。降级至备选。
  4. **Llama2 ref + OLMo-2-1B 主力**:能拿"双外部锚"(SmolLM + AllenAI),但需迁到 OLMo-core 框架,3-5 天工程,代价不匹配。否决。
  5. **Llama2 ref + Qwen2.5-1.5B 主力**:可作 Protocol B 替代,如 Qwen3 patch 出问题时 fallback。备用。

- **Consequences**:
  - **Positive**:
    - 双外部对齐 + 内部 ablation 兼得,publish-ready narrative
    - "Incepedia 跨架构稳健"是最强的科学 claim
    - 工程改动 ~半天 + 训练墙钟 +15%(相对单 A 协议从 ~28 天 → ~32 天)
  - **Negative / trade-offs**:
    - 维护两套实验配置(架构 spec + experiment yaml)
    - Track 1 milestone 数量翻倍(6 个 Llama2 → 但实际只跑 2 个 Llama2 + 6 个 Qwen3 = 8 总)
    - 当 Protocol A 与 Protocol B 在某个 ablation 上结论不一致(罕见)需新 ADR 处理
  - **Follow-ups**:
    - 实现 `src/incepedia/training/nanotron_qwen.py`(本 ADR 通过后立即写,未来训练前要测试)
    - 写 4 个新 reference experiment config:
      - `experiments/exp_ref_cosmopedia_v2_seed42/`(已有,Llama2)
      - `experiments/exp_ref_cosmopedia_v2_seed1337/`(新增,Llama2 第二 seed)
      - `experiments/exp_ref_cosmopedia_v2_qwen3_seed42/`(新增)
      - `experiments/exp_ref_cosmopedia_v2_qwen3_seed1337/`(新增)
      - `experiments/backbone_fineweb_edu_qwen3/`(新增,Qwen3 共享 backbone)
    - 项目末期发布数字时,**Cosmopedia 与 Incepedia 在两套架构上的对照表都要展示**
    - 若 Qwen3 patch 实测有性能或正确性问题,fallback 到 Qwen2.5-1.5B(备用决策路径已留)

- **Wall-clock budget**(8 × H100,基于 Cosmo-1B 公开吞吐反推)
  | 实验类型 | Protocol A | Protocol B | 备注 |
  |---|---|---|---|
  | 单次 30B token 训练(1 seed) | ~50h(2.1 天) | ~47h(2.0 天) | Qwen3-1.7B 略小 |
  | Cosmopedia v2 ref(2 seeds) | ~4.4 天 | ~4.0 天 | 包含 ~45 min eval |
  | Track 1 milestones(6 × 2 seeds) | — | ~24 天 | 仅 Protocol B 跑 |
  | Track 2 backbone | — | ~1.4 天 | Qwen3 一次 |
  | Track 2 forks(33 × 1 seed × 6B) | — | ~13 天 | 共享 backbone |
  | **项目训练总墙钟** | **~4.4 天** | **~40 天** | 双协议合计 ~44 天 |
  | nanotron Qwen 集成工程 | — | **0**(原生支持) | 原计划半天 patch,实测不需要 |
  | Qwen tokenizer 重 tokenize | — | ~10 min | 后台任务,不占训练墙钟 |

- **Related**:
  - ADR 0002(ablation 作为唯一质量指标)
  - ADR 0004(双轨评测协议 — Track 1 standalone / Track 2 seasoning)
  - ADR 0005(项目 scope:standalone + seasoning 双场景)
  - ADR 0006(评测栈策略 — lighteval latest)
  - `src/incepedia/training/nanotron_qwen.py`(本决定的实现)
  - `src/incepedia/training/launcher.py`(`ARCHITECTURE_SPECS` 字典)
