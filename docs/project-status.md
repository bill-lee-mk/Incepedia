# Incepedia · 项目状态简报

| 字段 | 值 |
|---|---|
| 更新频率 | 每周一 / 周四,2 次/周 |
| 本次更新 | 2026-04-21 12:00 UTC(Qwen tokenizer 重 tokenize 完成 / methodology §2.5 易混淆点 / P1 数据就绪,待你放行后开训) |
| 当前阶段 | **P1 · 基础设施搭建**(数据与管线就绪;**正式训练需你显式确认**) |
| Owner | bill-lee-mk |

> 本文档是项目的"一页纸"视图。细节见 `docs/methodology.md` 与 `docs/decisions/`。

---

## 1. 一句话定义

**构造一个合成预训练数据集 Incepedia,在 Llama2-1.82B(外部锚)与 Qwen3-1.7B(工作架构)双协议下,同时以"独立 corpus"和"decay 调料"两种身份,下游 benchmark 分数超越 Cosmopedia。**

---

## 2. 当前阶段进度(P1)

| 模块 | 状态 | 备注 |
|---|---|---|
| 仓库骨架 | ✅ | |
| 存储三层 + 同步脚本 | ✅ | event-driven rsync + audit log |
| 参考数据 · Cosmopedia v2(raw) | ✅ | 114 GB / 104 shards,本地+NAS 双副本 |
| 参考数据 · FineWeb-Edu 32 shards(raw) | ✅ | 72 GB,本地+NAS 双副本 |
| **Conda env**(Python 3.11 + lighteval 0.13 + datasets 3.6 + flash-attn 2.8.3) | ✅ | CUDA OK |
| **评测层**(port + runner) | ✅ | lighteval 0.13 适配已合入;`datasets>=3.6,<4` 已 pin |
| **训练层**(config + launcher + orchestrator,**支持双架构**) | ✅ | dry-run 通过;Llama2-1.82B + Qwen3-1.7B;Aim 注入 |
| Reference experiment configs · Protocol A(Llama2-1.82B × 2 seeds) | ✅ | Mistral tokenizer |
| Reference experiment configs · Protocol B(Qwen3-1.7B × 2 seeds) | ✅ | seed42 + seed1337;Qwen tokenizer |
| Backbone experiment config · Qwen3 | ✅ | `backbone_fineweb_edu_qwen3` |
| **Qwen3 nanotron 集成** | ✅ | 内置 `Qwen2Config` / `is_qwen2_config: True`,无自定义 patch |
| **Smoke test · cosmo-1b** | ✅ | CSR early-signal 端到端验证 |
| **eval 数据集预下载** | ✅ | `data/hf_cache/` 隔离缓存,降低 HF 429 |
| **Qwen tokenizer 数据重 tokenize** | ✅ | Cosmopedia v2 **102.26 GiB** + FineWeb-Edu **97.18 GiB**,各 16 `.ds` shards |
| **Tokenized · Cosmopedia v2(Mistral)** | ✅ | 31.5B tokens / ~60 GiB |
| **Tokenized · FineWeb-Edu(Mistral)** | ✅ | 29B tokens / ~55 GiB |
| **flash-attn** | ✅ | 已装 |
| 合成 pipeline(OpenRouter 路由) | ⏳ | P2 前完成 |

---

## 3. 资源现状

| 项 | 占用 / 说明 |
|---|---|
| `data/datasets/`(四套 tokenized 并存) | **约 316 GiB**(Mistral×2 + Qwen×2) |
| 本地 NVMe | 根分区约 **8.0 TiB 可用**(见 `df`) |
| NAS 冷备 | `INCEPEDIA_NAS_ROOT` 默认可写;**建议**将 `cosmopedia_v2_reference_qwen` 与 `fineweb_edu_backbone_qwen` 同步上 NAS(见 §6) |
| GPU | 以机器实时状态为准(`nvitop`) |
| API 花费 | 合成未大规模启动前 ≈ $0 |

---

## 4. lighteval 0.13 适配要点(归档)

| # | 问题 | 处理 |
|---|---|---|
| 1 | `pretrained=` → `model_name=` | `eval/runner.py` |
| 2 | 自定义 task 与内置同名 | `incep_` 前缀 |
| 3 | MMLU subset 分隔符 | `incep_mmlu_mc:subset` |
| 4 | task 串格式 | `task\|fewshot` |
| 5 | `datasets 4.x` 脚本数据集 | pin `datasets>=3.6,<4` + `HF_DATASETS_TRUST_REMOTE_CODE=1` |

---

## 5. 开训前例行

- `bash scripts/check_setup.sh`(应 **0 失败**;`.env` 建议 `chmod 600`)
- 你**显式放行**后再启动 nanotron 长训
- 首次 Protocol B 跑前确认 `experiments/.../config.yaml` 中 `dataset.path` 指向 Qwen 目录

---

## 6. NAS 同步命令(按需)

```bash
# Qwen tokenized(新增 ~200 GiB — Protocol B 训练反复读取)
bash scripts/sync_to_nas.sh dataset cosmopedia_v2_reference_qwen
bash scripts/sync_to_nas.sh dataset fineweb_edu_backbone_qwen

# 已有 Mistral 版(若尚未同步)
bash scripts/sync_to_nas.sh dataset cosmopedia_v2_reference
bash scripts/sync_to_nas.sh dataset fineweb_edu_backbone
```

---

## 7. 风险 & 开放问题

| 风险 / 问题 | 影响 | 当前状态 |
|---|---|---|
| Shell 子进程在长任务后挂死 | Medium | 仅影响 agent;用户终端不受影响 |
| GPU 资源共享 | High | 训前确认独占或配额 |
| lighteval API 漂移 | Medium | 已 pin 版本 + 自定义 port |

### TODO 队列(按优先级)

| # | 任务 | 触发时机 | 备注 |
|---|---|---|---|
| **T1** | **FA3 接入 nanotron**(加 `NANOTRON_USE_FA3=1` 开关,保留 FA2 作为 rotary/layer_norm/varlen 的回退;先做 10-step FA2/FA3 对照 benchmark,确认 loss 一致后再推广) | **seed42 跑完后、seed1337 开跑前** | 环境已装 `flash-attn-3 3.0.0`,但 nanotron 0.4 源码只有 FA2 路径,需要改 `nn/attention.py`。FA2 目前 ~99 TFLOPs/GPU 已算健康水位,FA3 预期再 +10-20%。结论写进 ADR 0006 |
| T2 | nanotron → Aim metrics 桥接 | FA3 开关落地同一批改动 | 目前 Aim UI 是空的,曲线先靠 `grep "iteration:" train.log`;需要把 nanotron logging 接到 `aim/` repo |
| T3 | 本地 nanotron patch 自动化(rotary / datatrove0.9 / consumption_stats) 写进 `scripts/bootstrap_env.sh` | 下次环境重建前 | 已有 `patches/*.patch`,目前需要手动 `git apply` |

---

## 8. 关键文档索引

- **方法论详解** → `docs/methodology.md`(含 Track 1「24B stable」vs Track 2 backbone「20B」说明)
- **ADR 列表** → `docs/decisions/README.md`(含 **0007** 双协议)
- **AI agent 守则** → `AGENTS.md`
- **存储** → `README.md` + NAS `README.md`

---

## 变更历史

| 日期 | 要点 |
|---|---|
| 2026-04-20 | 创建,P1 筹备 |
| 2026-04-21 | 双协议(ADR 0007)+ Qwen tokenizer 数据就绪 |
| 2026-04-21 12:00 UTC | **Qwen 重 tokenize 完成**;methodology 增补 §2.5;status 去陈旧「未 commit」条目 |
| 2026-04-21 10:25 UTC | **seed42 正式开跑**:8×H100、~65K tok/s、~99 TFLOPs/GPU、ETA ≈ 6h20m;路上修了 8 个兼容性坑(nanotron 入口 / YAML schema / vocab / rotary-FA28 / datatrove09 / consumption_stats / ZeRO-0 / grouped_gemm+pybind11),3 条 nanotron 本地 patch 放在 `patches/`;登记 TODO T1 FA3 接入 |
