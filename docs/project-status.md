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
| 合成 pipeline(OpenRouter 路由) | ⏳ | **P2 规范化完成**(见 `docs/codenames-cheatsheet.md §4`),W1 基建待 R1/R2 结束后开工 |
| **P2 代号族固化** | ✅ | E1–E5 / A1–A10 / G1–G4 / W1–W6 / ①–⑨ 全部进 `codenames-cheatsheet.md` |
| **P2 质量升级 ①–⑦** | ✅ 方案定稿 | 10B uniq / frontier 50% / Best-of-N / critic 50%×2 / 多源 seed / 10k persona / 20 结构;⑧⑨ 留 v0.2 |

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

| # | 任务 | 状态 | 备注 |
|---|---|---|---|
| **T1** | FA3 接入 nanotron(`NANOTRON_USE_FA3=1` 开关,默认 OFF;FA2 作为 rotary/layer_norm/varlen 的后端不变) | ✅ **实现完成**(ADR 0008、`patches/nanotron_fa3_optional_switch.patch`、import 测试通过);**对照 benchmark 待跑**(seed42 baseline 跑完后再做) | 验收项:50-step FA2 vs FA3 loss 差 ≤1e-3 相对、吞吐 ≥ +10%。验收前禁止把默认翻成 FA3 |
| T2 | nanotron → Aim metrics 桥接 | ✅ **实现完成**(orchestrator 自动拉起 `scripts/tail_train_log_to_aim.py` 侧车,每实验独立 Aim Run) | 正式 nanotron 侧原生桥接待上游实现,目前用 log-tail 方式,已足够画曲线 |
| T3 | 三条本地 nanotron patch 写进 `scripts/bootstrap_env.sh` | ✅ **实现完成**(idempotent `git apply --check`,不会重复打) | `patches/nanotron_{rotary_flash_attn28,datatrove09_dataset,consumption_stats_local,fa3_optional_switch}.patch` |
| T4 | `scripts/lint_nanotron_yaml.py`:launcher YAML vs nanotron 默认值 diff 审计 | ✅ **实现完成**(allow-list 35 项,`exit=0` 当无意外非默认字段) | 每次升级 nanotron / 改 launcher 的 PR 内必跑 |
| **T5** | **替换 sidecar 为 nanotron 直连 Aim/W&B**(在 nanotron `train_step_logs()` 末尾加 ~10 行 `aim.Run.track()` 或 `wandb.log()`,实时无延迟无解析) | 🟡 **待办** — **R1 seed42 跑完后立即做** | 当前 sidecar 是临时妥协(tail train.log → 解析正则 → push Aim);直连后:1) 0 延迟 2) metric 名称由 nanotron 决定不会漂 3) 跨实验对比更稳。工程量 ~30 min `git apply patches/nanotron_aim_direct.patch` + 测试 |

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
| 2026-04-21 ~10:50 UTC | **py-spy 定位慢路径**:`ignore_sanity_checks=False` 让每步调一次 `torch.testing.assert_close` 跨 rank → ~50% 吞吐损失。修回默认 True(commit `d7d0059`) |
| 2026-04-21 ~11:30 UTC | **T1-T4 四条 TODO 全部落地**:FA3 switch(`NANOTRON_USE_FA3=1`,默认 OFF;ADR 0008;patch 留档)、Aim sidecar 接入 orchestrator、bootstrap 自动 apply patches + 装 grouped_gemm/pybind11、`scripts/lint_nanotron_yaml.py`(35 非默认字段全 allow-listed) |
| 2026-04-22 08:00 UTC | **R1 seed42 30B 启动**(launcher batch_accum bug 已修;FRESH_START 跳过 1.5B-era 旧 ckpt;auto-resume 兜底;ETA ~45h) |
| 2026-04-23 02:30 UTC | **R1 进度 40%**(iter 9111/22888,11.9B/30B,lm_loss 1.43,281 TFLOPs/GPU,28% MFU);**TODO T5 登记**(seed42 跑完后做 nanotron→Aim 直连,告别 sidecar 临时妥协) |
| 2026-04-23 19:00 UTC | **P2 生成管线规范化**:新增 `docs/codenames-cheatsheet.md § 2.4–2.8` 五个代号族(①–⑨ / E1–E5 / A1–A10 / G1–G4 / W1–W6),新增 §4 生成管线剖面图;`methodology.md` 增补 §3.4(P2 质量升级表)与 §4 P2 规范版;**P2 方案定稿**(10B uniq / frontier 50% / Best-of-N / critic 50%×2 / 多源 seed / 10k persona / 20 结构);批准 E3 控制实验 + 6 条重生成 ablation(A1/A2/A5/A6/A8/A10)+ G3 无条件进 G4 规则;W1 待 R1/R2 结束后开工 |
| 2026-04-23 21:00 UTC | **FinePhrase 报告调研**(HF `finephrase` playbook,90 实验 / 486B token / 12.7 GPU 年)→ P2 增量修订:(1) **E 系列扩为两阶段设计**:新增 **A11 mix-ratio scan**(Track 2,seed42,100/50/30 三档 × 6B cooldown,~1.1d)驱动决策,Track 1 跑 **E2-100**(独立叙事) + **E2-mix**(最优 ratio,SOTA 可比) + E3(epoch 控制);mix-in 默认 FineWeb-Edu-HQ,W3-W5 后台并行 tokenize DCLM 作备;(2) G4 改为 2 档 4 runs 共 8 天;(3) **② frontier 50% 保持不变**(等 A2 对照实验思路成熟后再议);(4) 修订 A(本地 SmolLM2)暂不采纳,留观察 |
