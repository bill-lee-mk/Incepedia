# 0002 · Ablation training as the sole data-quality metric

- **Status**: accepted
- **Date**: 2026-04-20
- **Deciders**: bill-lee-mk
- **Context**:
  - 合成数据的主观"可读性"、"信息密度"与下游效果关联弱(见 Phi 系列 vs 社区复现的争论)
  - FineWeb (arXiv:2406.17557) / Cosmopedia / SmolLM 已确立行业范式:**ablation training + fixed eval suite**
  - 需要可复现、可 diff、可审计的判据
  - 成本:一次 1.82B × 30B token ablation ≈ 24h on 8×H100

- **Decision**:
  **唯一质量指标**:在固定协议下 ablation 训练得到的 lighteval early-signal 平均分。协议:
  - 模型:Llama2 ~1.82B(参考 FineWeb ablation)
  - 训练:30B token / 100B token / 350B token 三档
  - 超参:对齐 SmolLM / FineWeb 公布配方(lr 3e-4 cosine 或 trapezoidal)
  - 评测脚本:从 `huggingface/cosmopedia/evaluation/lighteval_tasks.py` 复刻,不改一行
  - Seeds:2 个(42, 1337),分数取平均

- **Alternatives considered**:
  1. **Perplexity on held-out**:对 cross-corpus 不可比,放弃
  2. **Quality classifier 打分**(FineWeb-Edu classifier):用于生成阶段的过滤器,但**不作为最终质量指标**(会 reward hacking)
  3. **LLM-as-judge on sampled docs**:只作为辅助 debug 信号

- **Consequences**:
  - Positive:有确定性、有历史可比;避免"看起来好"但下游不 work 的数据设计
  - Negative:每次变量验证成本 ≈ $3k(30B token)~ $30k(350B token);Proxy ablation(360M)降本优先
  - Follow-ups:实现 `src/incepedia/eval/` 与 cosmopedia 完全一致;reference baseline 对齐 SmolLM 公布数字 ±0.5pp

- **Related**:
  - `docs/methodology.md § 2`
  - `src/incepedia/eval/`
  - `huggingface/cosmopedia/evaluation/`
