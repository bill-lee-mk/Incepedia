# Incepedia

> 🇨🇳 中文(当前) · [🇬🇧 English](./README.en.md) · [📦 NAS 冷备份](/lambda/nfs/us-south-2/incepedia_exp_bak/README.md)

**目标**:在与 SmolLM/Cosmopedia 完全一致的 ablation 协议下,合成一份预训练数据集 **Incepedia**,跨多 benchmark 超越 Cosmopedia v1/v2,逼近 SmolLM2。

---

## 当前阶段(2026-04-23 快照)

🟢 **P1 · 复现** — train→convert→eval 端到端管线已打通并验证,seed42 30B 训练进行中

| 模块 | 状态 |
|------|------|
| 仓库骨架 + 三层存储 + NAS 同步 | ✅ |
| Conda env(Py 3.11 + torch 2.8 cu128 + nanotron + lighteval 0.13 + datatrove + flash-attn 2/3) | ✅ |
| Reference 数据(Cosmopedia v2 + FineWeb-Edu)tokenized,Mistral × Qwen tokenizer 双版本 | ✅ |
| 双协议架构(ADR 0007:Llama2-1.82B 锚 + Qwen3-1.7B 工作架构) | ✅ |
| nanotron 训练 launcher + auto-resume + 断网保护 | ✅ |
| nanotron Qwen2 → HF Qwen2 转换器(ADR 0009) | ✅ |
| lighteval `cosmopedia-full` 任务组(129 任务,与 Cosmopedia eval.slurm 逐字一致) | ✅ |
| Aim 实时训练曲线监控(sidecar 临时方案,T5 直连 TODO) | ✅ |
| INDEX.parquet 实验登记 + NAS 冷备 | ✅ |
| **R1 · seed42 30B 完整训练** | 🟡 **进行中**(~30h ETA) |
| seed1337 / Track 2 cooldown-fork / Protocol A Llama2 / Incepedia 生成器 / decontam | ⏳ |

详细进度见 [`docs/project-status.md`](./docs/project-status.md)(每周一/周四更新)。

---

## 一句话方法论

**用多生成器混合 + web-grounded RAG + 多 agent 自批改 + 三层去污染,在双协议(Llama2-1.82B 锚 + Qwen3-1.7B 工作)/ Cosmopedia-parity 评测下打过 Cosmopedia v1/v2,然后扩到 30B+ 公开发布。**

深度细节见:
- [`docs/methodology.md`](./docs/methodology.md) — 方法论与决策依据
- [`docs/decisions/`](./docs/decisions/) — 9 条 ADR(架构、协议、工具、评测策略)
- [`docs/incepedia-overview.md`](./docs/incepedia-overview.md) — 高层级项目叙事
- [`docs/project-status.md`](./docs/project-status.md) — 双周状态简报

---

## 端到端管线 — train → convert → eval → INDEX → NAS

```
                                    ┌─────── methodology.md / ADR 0004 双轨评测协议
                                    │
                                    ▼
              experiments/<exp_id>/config.yaml
              (单一可复现来源 — 谁有这个 yaml 就能完整复现)
                                    │
                                    ▼
            scripts/run_experiment.py(orchestrator)
                                    │
        ┌───────────────────────────┼─────────────────────────────┐
        │                           │                             │
        ▼                           ▼                             ▼
┌───────────────┐         ┌──────────────────┐          ┌─────────────────┐
│ train.log →   │  ←───── │ nanotron 训练    │ ─────→   │ ckpt/<step>/    │
│ Aim sidecar   │         │ (8×H100 dp,FA2)  │          │  config.yaml    │
│ (实时曲线)    │         │ trapezoidal lr   │          │  model/...      │
└───────────────┘         │ ~55h 30B/seed    │          │  optimizer/...  │
                          │ auto-resume 兜底  │          └─────────────────┘
                          └──────────────────┘                  │
                                                                ▼
                                            scripts/convert_nanotron_qwen2_to_hf.py
                                                          (ADR 0009)
                                                                │
                                                                ▼
                                                ┌──────────────────────────────┐
                                                │ hf_ckpt/                     │
                                                │   config.json (Qwen2ForCausalLM)│
                                                │   model.safetensors (3.5 GB) │
                                                │   tokenizer.json             │
                                                └──────────────────────────────┘
                                                                │
                                                                ▼
                                          lighteval (accelerate, cosmopedia-full)
                                          129 任务,full samples,~28 min on 8×H100
                                          (与 Cosmopedia eval.slurm 逐字一致 → 分数直接可比)
                                                                │
                                                                ▼
                                                ┌────────────────────────┐
                                                │ metrics.json           │
                                                │ INDEX.parquet (+1 row) │
                                                │ NAS 同步:config/ckpt/  │
                                                │   hf_ckpt/eval/        │
                                                └────────────────────────┘
```

**发布**(P3 末期):`hf_ckpt/` → `huggingface_hub.upload_folder` → HF Hub 公开仓库。

---

## 当前实测数字

| 指标 | 当前值(2026-04-23 19h+ 训练) |
|------|------------------------------|
| 训练吞吐 | 185 K tok/s 全局 / 23 K tok/s/卡 |
| MFU | 28%(H100 BF16,1.7B 模型健康水位) |
| 30B/seed 墙钟 | 实测 ~45-55 h |
| Eval 时间 | 28.5 min(cosmopedia-full 129 任务,full samples,8×H100) |
| Convert 时间 | ~30 s(CPU,150 行 Python) |
| Auto-resume 检测 | 自动从最后一个 valid ckpt 接续(`scripts/check_resume.py` 预览) |

---

## 三阶段路线图

| 阶段 | 目标 | 完成标准 |
|------|------|----------|
| **P1 · 复现** | Cosmopedia-parity baseline + 双协议双 seed 验证 | 与 SmolLM/FineWeb 公开数字 ±0.5pp 对齐 |
| **P2 · PoC** | Incepedia v0.1(3B token)在两轨持平或超过 Cosmopedia | 最小闭环:OpenRouter 生成 → tokenize → 训 → 转 → 评 |
| **P3 · 迭代+扩容** | Incepedia v1.0(10-30B token)显著超越 Cosmopedia | 跨 benchmark 优势 + HF Hub 发布 + 论文 |

详见 ADR 0004 / 0005 / 0007。

---

## 双协议架构(ADR 0007 核心决策)

| 协议 | 架构 | 作用 | 频率 |
|------|------|------|------|
| **A · Llama2-1.82B** | 全注意力,RoPE θ=10000,无 bias,Mistral tokenizer | **外部锚** — 与 SmolLM/Cosmopedia 公开数字 ±0.5pp 对齐 | 全项目仅 2 次(Cosmopedia v2 × 2 seeds) |
| **B · Qwen3-1.7B** | GQA(8 KV),RoPE θ=1e6,QKV bias,Qwen tokenizer 151k vocab | **工作架构** — 所有 Incepedia 版本 ablation + 发布 | 6 个 milestones × 2 seeds + 33 个 cooldown-fork |

不变项:nanotron(原生支持两架构)、lighteval cosmopedia-full(任务集相同)、训练超参(lr / schedule / seeds 严格相同)。

---

## 双轨评测协议(ADR 0004 核心)

| 轨道 | 协议 | 答什么问题 | 用途 |
|------|------|-----------|------|
| **Track 1 · Standalone** | 1.82B/1.7B 从头训 × 30B tokens × 2 seeds | "数据集 X 作为独立 pretraining corpus,比 Y 强吗?" | 版本 milestone 验收 |
| **Track 2 · Seasoning** | 共享 backbone(FineWeb-Edu × 20B)+ cooldown-fork × 6B | "数据集 X 作为 decay 阶段调料,比 Y 强吗?" | 日常 ablation |

每个 milestone 都跑 2 seeds 并平均(ADR 0004 噪声可降到 ±0.15pp)。

---

## 目录布局

```
Incepedia/
├── README.md / README.en.md         # 本文件
├── .env                              # API keys(已 gitignore)
├── docs/
│   ├── methodology.md                # 方法论与决策依据
│   ├── project-status.md             # 双周状态简报(最新进度)
│   ├── incepedia-overview.md         # 高层级项目叙事
│   ├── multi-machine-eval-setup.md   # 多机评测部署
│   └── decisions/                    # 9 条 ADR
├── configs/                          # topics.yaml / personas.yaml / generators.yaml
├── src/incepedia/
│   ├── training/                     # nanotron launcher + Pydantic configs
│   ├── eval/                         # lighteval 任务定义(cosmopedia-full)+ runner
│   ├── generation/                   # OpenRouter 异步批量生成器(P2 实现)
│   └── config.py                     # REPO_ROOT, DATA_DIR 等路径常量
├── scripts/
│   ├── run_experiment.py             # train→convert→eval→INDEX→NAS 端到端 orchestrator
│   ├── convert_nanotron_qwen2_to_hf.py  # nanotron Qwen2 → HF Qwen2 转换(ADR 0009)
│   ├── check_resume.py               # 启动前预览 auto-resume 状态
│   ├── tokenize_dataset.py           # datatrove 把 Parquet → nanotron .ds shards
│   ├── prefetch_eval_datasets.py     # 离线缓存 eval 数据集(避 HF 429)
│   ├── lint_nanotron_yaml.py         # 审计 launcher YAML 与 nanotron 默认值差异
│   ├── tail_train_log_to_aim.py      # train.log → Aim sidecar(临时;T5 替换)
│   ├── bench_fa2_vs_fa3.py           # FA2 vs FA3 数值/速度 benchmark
│   ├── stats_cosmopedia_v2.py        # 上游数据 format/audience/token_length 统计
│   ├── sync_to_nas.sh                # 事件驱动 rsync(ckpt/hf_ckpt/gen/eval/...)
│   ├── mount_lambda_nas.sh           # 在 164 A100 box 挂 NAS(virtiofs / sshfs / rsync 三选一)
│   └── bootstrap_env.sh              # 一键搭建 conda env + 装 patches
├── patches/                          # 4 条本地 nanotron patch(rotary/datatrove/cache/FA3)
│   └── nanotron_*.patch              # bootstrap 自动 idempotent apply
├── data/                             # 大数据(.gitignore)
│   ├── datasets/                     # tokenized .ds shards(训练热读)
│   ├── reference/                    # Cosmopedia v2 / FineWeb-Edu 原始 parquet
│   ├── hf_cache/                     # 评测数据集隔离 cache
│   └── raw_generations/              # 合成 batch parquet(P2)
├── experiments/                      # 每个实验一个目录
│   ├── exp_ref_cosmopedia_v2_qwen3_seed{42,1337}/   # Protocol B baseline
│   ├── exp_ref_cosmopedia_v2_seed{42,1337}/          # Protocol A baseline
│   ├── backbone_fineweb_edu_qwen3/                   # Track 2 共享 backbone
│   └── exp_<exp_id>/
│       ├── config.yaml               # 完全自包含,仅凭这个 yaml 就能复现
│       ├── nanotron.yaml             # launcher 渲染产物(.gitignore)
│       ├── ckpt/<step>/              # nanotron native ckpts(.gitignore,走 NAS)
│       ├── hf_ckpt/                  # HF Qwen2 转换产物(.gitignore,走 NAS)
│       ├── train.log                 # tee 自 nanotron stdout(.gitignore)
│       ├── eval/                     # lighteval 完整结果
│       ├── metrics.json              # 14-16 个 benchmark 分数(入 git)
│       └── README.md                 # 人话总结(可选)
├── aim/                              # Aim 运行时 DB(.gitignore)
└── INDEX.parquet                     # 所有实验聚合一览(入 git)
```

---

## 三层存储防护(ADR 0001)

| 层 | 位置 | 内容 | 副本 |
|---|---|---|---|
| **L1 · 热** | 本地 NVMe `/home/ubuntu/lilei/projects/Incepedia/` | 全量训/评/生成 single source of truth | 1 |
| **L2 · 冷镜像** | NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/`(virtiofs)| 事件驱动 rsync;3.2 PB 容量 | +1 |
| **L3 · 元数据兜底** | git remote(GitHub) | configs/ + experiments/*/config.yaml + metrics.json + INDEX.parquet 等 <100 MB 结构化产物 | +1 |

**最坏情况**:本地 SSD + NAS 同时挂 → git 救回所有实验配置与评测结论,只丢权重(用相同 config 重训即可)。

---

## 同步策略(事件驱动)

| 触发 | 同步对象 | 子命令 |
|------|---------|--------|
| ckpt 保存完成 | `experiments/<exp>/ckpt/<step>/` | `sync_to_nas.sh ckpt <exp_id>` |
| HF 转换完成(自动) | `experiments/<exp>/hf_ckpt/` | `sync_to_nas.sh hf_ckpt <exp_id>` |
| 一轮合成批次完成 | `data/raw_generations/<batch>/` | `sync_to_nas.sh gen <batch_id>` |
| 实验启动 | `config.yaml`(立即) | `sync_to_nas.sh config <exp_id>` |
| 评测完成 | `experiments/<exp>/eval/` | `sync_to_nas.sh eval <exp_id>` |
| tokenized 数据集落盘 | `data/datasets/<id>/` | `sync_to_nas.sh dataset <id>` |
| 每晚 03:00 | 全量兜底 | cron(待加) |

---

## 多机部署 — 165 (8×H100 训练) + 164 (8×A100 评测)

> **架构**:165 训 + 164 评,两边都通过 NAS 共享 ckpt/eval/INDEX。  
> **当前**:165 已上线;164 待 NAS 接通(virtiofs 需 Lambda 控制台 attach 或走 sshfs fallback)。详见 [`docs/multi-machine-eval-setup.md`](./docs/multi-machine-eval-setup.md)。

```
H100 (8×80G,host 165)        NAS (3.2 PB)         A100 (8×40G,host 164)
─────────────────────       ─────────────────       ─────────────────
nanotron 训练               experiments/             lighteval eval
       ↓ convert            ├─ <exp>/ckpt/           ↑ rsync HF ckpt
       ↓ sync               ├─ <exp>/hf_ckpt/        ↓ 写 metrics
                            ├─ <exp>/eval/           ↑ rsync 回
                            └─ INDEX.parquet
                              (单一真相源)
```

---

## 快速开始

### 一次性环境搭建

```bash
cd /home/ubuntu/lilei/projects/Incepedia
bash scripts/bootstrap_env.sh        # conda env + nanotron + datatrove + 4 条本地 patch
```

### 跑一个 reference 实验(端到端)

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate incepedia

# 启动前确认 GPU 状态
nvidia-smi --query-compute-apps=pid --format=csv,noheader

# 启动前预览 auto-resume 状态
python scripts/check_resume.py experiments/exp_ref_cosmopedia_v2_qwen3_seed42/config.yaml

# 启动(支持 nohup 断网保护;支持 INCEPEDIA_FRESH_START=1 强制全新跑)
export CUDA_DEVICE_MAX_CONNECTIONS=1
nohup python scripts/run_experiment.py \
    --config experiments/exp_ref_cosmopedia_v2_qwen3_seed42/config.yaml \
    > /tmp/seed42_train.out 2>&1 &

# 监控
tail -F experiments/exp_ref_cosmopedia_v2_qwen3_seed42/train.log | grep iteration:
# Aim Web UI(假设 SSH 隧道已开)
open http://localhost:43800/
```

### 单独评测一个已有 ckpt

```bash
python scripts/run_experiment.py \
    --config experiments/<exp>/config.yaml --eval-only
# 自动检测是否需要 convert,然后 lighteval cosmopedia-full → metrics.json → INDEX → NAS
```

### 重启崩溃的训练(auto-resume)

```bash
# 直接重新执行同一命令,launcher 自动从最后一个 valid ckpt 接续
nohup python scripts/run_experiment.py --config <...>/config.yaml > /tmp/resume.out 2>&1 &
```

---

## API key 配置

```bash
# 双处配置(全局 + 仓库级);仓库级优先,代码用 python-dotenv 加载
# 全局:~/.bash_profile 里 export OPENROUTER_API_KEY=sk-or-v1-...
# 仓库级:.env(已 gitignore)
grep OPENROUTER_API_KEY .env && echo OK
```

代码查找顺序:`OPENROUTER_API_KEY` → `OpenRouter_API_KEY` → `OPENAI_API_KEY`(已做兼容)。

---

## 实验命名规范

```
exp_{semantic_slug}_{seed?}
```

示例:
- `exp_ref_cosmopedia_v2_qwen3_seed42` — Protocol B Cosmopedia v2 baseline
- `exp_ref_cosmopedia_v2_seed1337` — Protocol A Cosmopedia v2 baseline 第二 seed
- `exp_inc_v01_qwen3_seed42` — Incepedia v0.1 在 Protocol B 第一次跑(P2)
- `backbone_fineweb_edu_qwen3` — Track 2 共享 backbone

每个实验目录必含 `config.yaml`(单一可复现来源)+ `track: 1|2` 字段(ADR 0004)。

---

## 给 agent / 后续接手人员的阅读顺序

| # | 文档 | 用途 |
|---|------|------|
| 1 | 本 README | 5 分钟掌握项目全貌 |
| 2 | [`docs/project-status.md`](./docs/project-status.md) | 当前进度、TODO 队列、风险表 |
| 3 | [`docs/methodology.md`](./docs/methodology.md) | 方法论详解、双轨评测、Incepedia 生成策略(C1-C16) |
| 4 | [`docs/incepedia-overview.md`](./docs/incepedia-overview.md) | 高层级项目叙事 |
| 5 | [`docs/decisions/README.md`](./docs/decisions/README.md) | 9 条 ADR 索引 |
| 6 | `INDEX.parquet`(`pandas.read_parquet`)| 查最新实验状态与 benchmark 分数 |
| 7 | 最近修改的 `experiments/exp_*/` | 了解正在做什么实验 |

agent 守则见 [`AGENTS.md`](./AGENTS.md);贡献规范见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

---

## 关键参考

- [Cosmopedia 博客](https://huggingface.co/blog/cosmopedia)
- [SmolLM 博客(v2 配方)](https://huggingface.co/blog/smollm)
- [FineWeb 论文(ablation 范式)](https://arxiv.org/abs/2406.17557)
- [Cosmopedia GitHub](https://github.com/huggingface/cosmopedia)(我们 vendor 在 `third_party/cosmopedia/`)
- [Phi-4 技术报告](https://arxiv.org/abs/2412.08905)
- [Hägele et al. 2024 cooldown-fork](https://arxiv.org/abs/2405.18392)(Track 2 协议依据)

---

© 2026 · Incepedia contributors · Apache-2.0
