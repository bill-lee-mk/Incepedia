# 0005 · Project scope — standalone corpus AND decay seasoning

- **Status**: accepted
- **Date**: 2026-04-20
- **Deciders**: bill-lee-mk

- **Context**:
  - Cosmopedia v1(2024-02)用法:**独立 pretraining corpus**,Cosmo-1B 直接用它做主力训练数据(25B synthetic + 5B code/stories = 180B 训练 token)
  - Cosmopedia v2(2024-07 SmolLM)用法:**~10% 配料**(28B Cosmo-v2 + 220B FineWeb-Edu + 4B Python-Edu)
  - Cosmopedia v2(2025 SmolLM2)用法:**decay 阶段 4% 调料**,仅在 stage 4 / 10T-11T 之间出现
  - 趋势:合成数据在 scale up 过程中从"主力"降为"调料"。但这不代表"主力"场景无价值——Phi-4(2024-12)用 ~400B synthetic 作为 pretrain 全程主力,证明合成数据仍有作为主力的空间
  - Incepedia 必须选定目标场景

- **Decision**:
  Incepedia 设计目标 = **同时覆盖两种场景**:
  1. **独立 corpus 场景** — 作为 100% 主力训练数据,对比 Cosmopedia v1 / Phi-1.5 数据
  2. **decay 调料场景** — 作为 SmolLM2 式 stage 4 / cooldown 调料,对比 Cosmopedia v2

  具体含义(对数据内容的约束):
  - **信息密度高**(独立场景需要 benchmark 信号饱和):充分的 STEM + 知识 + 推理覆盖
  - **事实真实性强**(调料场景尤其需要,模型 cooldown 阶段"定型"):强制 seed-grounding + web rephrasing 主路径
  - **多样性充分**(独立场景需要不塌模式):多生成器 × persona × 结构 × 难度
  - **难度分层完整**(调料场景要配合冷模型):elementary → expert 5 档

  **反面**:明确**放弃**"纯独立"或"纯调料"两种狭窄 scope。

- **Alternatives considered**:
  1. **🅰 只做独立 corpus(Cosmo-1B 路线)**:测试方法简单,但 SmolLM2 时代已证明合成数据规模上限在独立场景约 28–40B,**天花板较低**
  2. **🅱 只做 decay 调料(SmolLM2 对位)**:紧跟 2025 主流路线,但放弃了"Incepedia 能否独当一面"的野心,**故事不够大**
  3. **🆎 双场景完整验证**:本决定的升级版,最终 v1.0 在两种 full-size 协议上都做终极验收
  4. **🆑 务实双场景**(本决定):P2/P3 常规 ablation 覆盖两种场景,v1.0 发布时同时给出两种分数

- **Consequences**:
  - **Positive**:
    - 不管上游 2027 年再把合成数据定位在哪种场景,Incepedia 都能兑现
    - 两种评测 track(ADR 0004)都有对应证据
    - C1–C16 生成策略同时服务两种目标,无内部冲突
  - **Negative / trade-offs**:
    - 设计复杂度略高于单场景:某些配比可能需要在两种场景间权衡(比如 stories 占比对独立场景 HellaSwag 友好,对调料场景收益递减)
    - 总 ablation 数量增加(两种 track 各要跑一次 baseline)
  - **Follow-ups**:
    - P3 末期做 v1.0 双场景终极验收:`exp_inc_v10`(Track 1)+ `fork_inc_v10`(Track 2)
    - 若两种场景方向出现冲突(比如某变量对 Track 1 +1pp 但 Track 2 -1pp),提 ADR 做取舍

- **Related**:
  - ADR 0004(双轨评测协议,为本 scope 提供工具)
  - `docs/methodology.md` §1、§3
