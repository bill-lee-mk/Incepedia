# 0003 · Multi-tier generator routing via OpenRouter

- **Status**: accepted
- **Date**: 2026-04-20
- **Deciders**: bill-lee-mk
- **Context**:
  - Cosmopedia v2 实验显示:在 "textbook from web seed" 任务上,换更大生成器(Llama3-70B / Mixtral-8x22B / Qwen1.5-72B → Mixtral-8x7B)无显著提升
  - 但在数学/推理/代码任务上,生成器质量与下游效果高度相关(Phi-4 / Nemotron-CC-Math / DeepSeek-Math 证据)
  - OpenRouter 提供统一 API 访问 OpenAI / Anthropic / DeepSeek / Llama / Mistral / Qwen 等

- **Decision**:
  按任务档位路由生成器:
  - `midtier_bulk`:常识 / textbook / wikihow / story / blog / Socratic → DeepSeek-V3, Llama-3.3-70B, Mistral-Large, Qwen2.5-72B, GPT-4o-mini
  - `frontier_reasoning`:数学 / 推理 / 代码 / 多步 → Claude-3.5-Sonnet, GPT-4o, DeepSeek-R1, Claude-3.5-Haiku
  - `critic`(多 agent 自批改的 judge):Claude-3.5-Sonnet / GPT-4o

  每个 tier 强制至少 2 个家族,按权重抽样。`configs/generators.yaml` 是唯一路由表。

- **Alternatives considered**:
  1. **单生成器堆量**:Cosmopedia 的做法,已知同质化严重
  2. **所有任务都用 frontier**:成本爆炸(~10×)且在 bulk 任务无收益
  3. **所有任务都用 midtier**:在数学/推理任务上质量不足

- **Consequences**:
  - Positive:多样性提升(多家族 → 风格/偏置不集中);成本可控(frontier 只用在高价值任务);降低单模偏置
  - Negative:路由逻辑需维护;不同模型的 API 格式/长度限制/rate-limit 需封装;审计时要记录每条样本的生成模型
  - Follow-ups:在每条生成样本的 Parquet 里记录 `generator` 字段(P2 实现);定期跑 leave-one-model-out ablation 验证各模型边际贡献

- **Related**:
  - `configs/generators.yaml`
  - `docs/methodology.md § 3.1`
  - `src/incepedia/generation/`
