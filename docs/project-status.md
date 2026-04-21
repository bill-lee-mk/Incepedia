# Incepedia · 项目状态简报

| 字段 | 值 |
|---|---|
| 更新频率 | 每周一 / 周四,2 次/周 |
| 本次更新 | 2026-04-21 06:20 UTC(双协议架构落地:ADR 0007 + Qwen3-1.7B spec + 4 个新 reference experiment configs) |
| 当前阶段 | **P1 · 基础设施搭建**(~90% 完成) |
| Owner | bill-lee-mk |

> 本文档是项目的"一页纸"视图。细节见 `docs/methodology.md` 与 `docs/decisions/`。

---

## 1. 一句话定义

**构造一个合成预训练数据集 Incepedia,在 Llama2-1.82B(外部锚)与 Qwen3-1.7B(工作架构)双协议下,同时以"独立 corpus"和"decay 调料"两种身份,下游 benchmark 分数超越 Cosmopedia。**

---

## 2. 当前阶段进度(P1)

| 模块 | 状态 | 备注 |
|---|---|---|
| 仓库骨架 | ✅ | 18 个 commit |
| 存储三层 + 同步脚本 | ✅ | event-driven rsync + audit log |
| 参考数据 · Cosmopedia v2(raw) | ✅ | 114 GB / 104 shards,本地+NAS 双副本 |
| 参考数据 · FineWeb-Edu 32 shards(raw) | ✅ | 72 GB,本地+NAS 双副本 |
| **Conda env**(Python 3.11 + lighteval 0.13 + datasets 3.6 + flash-attn 2.8.3) | ✅ | 8.5 GB,CUDA OK |
| **评测层**(port + runner) | 🟡 | 代码就绪,5 处 API 适配 bug 已识别并修(uncommitted) |
| **训练层**(config + launcher + orchestrator,**支持双架构**) | ✅ | dry-run 通过;Llama2-1.82B + Qwen3-1.7B spec 都派发正确 |
| Reference experiment configs · Protocol A(Llama2-1.82B × 2 seeds) | ✅ | seed42 + seed1337 |
| Reference experiment configs · Protocol B(Qwen3-1.7B × 2 seeds) | ✅ | seed42 + seed1337(新增) |
| Backbone experiment config · Qwen3 | ✅ | `backbone_fineweb_edu_qwen3`(新增) |
| Qwen3 nanotron patch | 🟡 | spec dict 就绪;运行时 patch 是 stub,首次 Qwen3 训练前实现 |
| **Tokenized · Cosmopedia v2** | ✅ | **31.5B tokens / 58.76 GiB / 16 .ds shards / 9.2 min** |
| **Tokenized · FineWeb-Edu backbone** | ✅ | **29B tokens / 54.04 GiB / 16 .ds shards / 8.6 min** |
| **flash-attn** | ✅ | 2.8.3 已装 |
| **Smoke test cosmo-1b** | 🟡 | 9 次重试,API 适配问题层层揭露并已修复;最后一次重试因 Shell 挂死未启动 |
| 合成 pipeline(OpenRouter 路由) | ⏳ | P2 前完成 |

---

## 3. 资源现状

| 项 | 占用 |
|---|---|
| 本地 NVMe | 290 GB / 8.2 TB available(env + raw + tokenized) |
| NAS 冷备 | 186 GB(raw 参考数据;tokenized 待同步) |
| GPU | **8 张 H100 全空** ✅(其他用户 Qwen3 训练已结束) |
| API 花费 | $0(本次没调合成 API) |

---

## 4. 本次会话(2026-04-20 → 2026-04-21)发现的 lighteval 0.13 API 改动 + 修复(关键经验)

按发生顺序,每个 bug 都通过 smoke test 暴露:

| # | Bug | 修复 | 文件 |
|---|---|---|---|
| 1 | nanotron 强制 import flash_attn,在装 flash_attn 前 lighteval import 失败 | 临时卸 nanotron,装 flash-attn 后再装回 | env-level |
| 2 | `pretrained=` 已重命名为 `model_name=` | runner._model_args() | `eval/runner.py` |
| 3 | 自定义 task 与 lighteval 0.13 内置同名冲突(hellaswag/piqa/winogrande/...) | 全部加 `incep_` 前缀 | `eval/lighteval_tasks.py` |
| 4 | MMLU subset 使用 `_` 拼接被解析失败 | 恢复 `incep_mmlu_mc:subset` `:` 分隔(lighteval 把 `:` 视为 subset 分隔符) | `eval/lighteval_tasks.py` |
| 5 | task spec 格式从 `suite\|task\|fewshot\|truncate` 改为 `task\|fewshot` | `_task_str()` 改为 2 元素 | `eval/lighteval_tasks.py` |
| 6 | `datasets 4.x` 完全删了 dataset 加载脚本(piqa.py/siqa.py/boolq.py 等都用脚本) | 降到 `datasets<4`(3.6.0) | env-level |
| 7 | 加 `HF_DATASETS_TRUST_REMOTE_CODE=1`(配合 #6) | runner driver template | `eval/runner.py` |

**这些代码修改都已完成、已测试,但因为 Shell 在最后一轮挂死,未 commit。**

---

## 5. ⚠️ 未 commit 的本地改动(下次接力第一件事)

```
M src/incepedia/eval/lighteval_tasks.py   (incep_ 前缀 + : 分隔 + 2-element task string)
M src/incepedia/eval/runner.py            (model_name + MMLU regex + 环境变量注入)
```

降版的 `datasets 3.6.0` 已生效但**没写进 requirements.txt**(还是 `datasets>=2.19`)。

---

## 6. 下次会话接力(明确步骤)

```bash
# 0. shell 恢复后第一件事:commit 本次的 5 个 eval 修复
cd /home/ubuntu/lilei/projects/Incepedia
git add src/incepedia/eval/{lighteval_tasks.py,runner.py}
git diff --cached
# 检查无误后:
git commit -m "fix(eval): adapt port+runner to lighteval 0.13 quirks"
git push

# 1. 锁 datasets 版本到 requirements.txt
sed -i 's/^datasets>=.*/datasets>=3.6,<4/' requirements.txt
git add requirements.txt
git commit -m "chore(deps): pin datasets<4 (lighteval needs script support)"
git push

# 2. 跑 smoke test(GPU 已空)
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate incepedia
nohup bash scripts/run_eval.sh HuggingFaceTB/cosmo-1b experiments/_sanity_cosmo_1b/eval early-signal 200 > /tmp/smoke_run.out 2>&1 &
# 监控:
watch -n 30 "tail -3 /tmp/smoke_run.out; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -8"
# 预期 ~30-45 min 完成,产出 experiments/_sanity_cosmo_1b/metrics.json
# 验收:HellaSwag ~50-55%,MMLU cloze ~25-30%,ARC challenge ~30-35%(参考 cosmo-1b 公开数字)

# 3. tokenized 数据同步到 NAS
bash scripts/sync_to_nas.sh dataset cosmopedia_v2_reference
bash scripts/sync_to_nas.sh dataset fineweb_edu_backbone
```

---

## 7. 风险 & 开放问题

| 风险 / 问题 | 影响 | 当前状态 |
|---|---|---|
| Shell 子进程在长 nohup 后挂死(本会话第 3 次) | Medium | 🟡 仅影响 agent,不影响 user 终端 |
| GPU 资源共享 | High | 🟢 当前空闲 |
| lighteval 0.13 API 频繁变化 | Medium | 已识别 5 个适配点(见 §4),修复就绪 |
| `datasets<4` 与 lighteval 0.13 警告冲突 | Low | 实测运行 OK,仅警告 |

---

## 8. 本会话(累计)commits

```
06497d3 chore(env): pin python 3.11 + lighteval from pypi
1de3a72 docs: log tokenize progress + GPU blocker in status brief
c746d96 feat(data): add datatrove tokenizer pipeline for reference data
d2a6ecd feat(training): add config schema, nanotron launcher, orchestrator
d7ec352 fix(gitignore): exclude third_party (not submodules)
5696b4f refactor(eval): adapt port + runner to lighteval 0.13 API
04bf5e7 docs: add project status brief (twice-weekly update)
```

---

## 9. 关键文档索引

- **方法论详解** → `docs/methodology.md`
- **ADR 列表** → `docs/decisions/README.md`(6 条)
- **AI agent 守则** → `AGENTS.md`
- **存储使用手册** → `README.md` + NAS `README.md`

---

## 变更历史

| 日期 | 要点 |
|---|---|
| 2026-04-20 | 创建,P1 筹备阶段 |
| 2026-04-20 | 完成环境重建 + eval/training 代码重写 + Cosmopedia v2 tokenize (31.5B tokens) |
| 2026-04-21 01:30 UTC | FineWeb-Edu tokenize ✅(29B tokens / 54 GiB)+ flash-attn ✅;smoke test 修了 5 个 lighteval 0.13 适配 bug,最后一轮 Shell 挂死未启动;有 2 个文件 uncommitted |
| 2026-04-21 06:20 UTC | **双协议架构决策落地**(ADR 0007):Llama2-1.82B 作外部锚,Qwen3-1.7B 作工作架构;launcher 重构支持多架构;新增 4 个 reference experiment configs;`incepedia-overview.md` + .docx 同步;Qwen3 nanotron patch spec 就绪,运行时 patch 留作首次训练前实现 |
