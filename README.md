# Incepedia

> 🇨🇳 中文(当前) · [🇬🇧 English](./README.en.md) · [📦 NAS 冷备份](/lambda/nfs/us-south-2/incepedia_exp_bak/README.md)

基于 Cosmopedia 架构、目标在下游 benchmark 上**超越 Cosmopedia** 的合成预训练数据集项目。

---

## 一句话

**用多生成器混合 + web‑grounded RAG + 多 agent 自批改 + 三层去污染,在 1.82B Llama2 / 30B token ablation 上打过 Cosmopedia v2,然后扩到 10B+。**

## 当前阶段

🟡 **阶段 0 · 脚手架搭建**(本 README 即本阶段产物)

- [x] OpenRouter API key 就绪
- [x] 存储架构定版(本地全量 hot+cold · NAS 独立冷备 · git 元数据第三层)
- [x] NAS 性能基准(virtiofs,大文件 885 MB/s 写 · 并发 3.4 GB/s)
- [ ] 仓库目录骨架
- [ ] 依赖环境(nanotron / lighteval / datatrove)
- [ ] Reference baseline:复现 Cosmopedia v2 @ 30B token
- [ ] OpenRouter 异步批量生成器
- [ ] `INDEX.parquet` 实验登记与同步 hook

## 设计哲学(简版)

1. **数据质量 = 模型下游分数**。抛弃主观判断,一切用 ablation 训练说话。
2. **"web‑grounded synthetic" > 凭空生成**。所有生成强制 seed‑grounding 降幻觉。
3. **瓶颈按任务路由**:常识类用中档模型堆量;数学/推理/代码上 frontier 模型。
4. **多样性是一等公民**。多生成器 × persona × 难度分层 × 结构模板 × embedding 去重。
5. **三层去污染**:n‑gram + embedding 相似度 + LLM‑as‑judge。

深度讨论见 [`docs/methodology.md`](./docs/methodology.md)(待写)。

## 目录布局

```
Incepedia/
├── README.md / README.en.md         # 本文件
├── .env                              # API keys(已在 .gitignore)
├── docs/                             # 方法论、实验日志、决策记录(ADR)
├── configs/                          # topics.yaml / personas.yaml / prompt_templates/
├── src/
│   ├── generation/                   # OpenRouter 异步批量生成器
│   ├── dedup/                        # MinHash + embedding 去重
│   ├── decontam/                     # 三层去污染
│   ├── training/                     # nanotron 训练入口
│   └── eval/                         # lighteval 任务定义(从 cosmopedia 复刻)
├── scripts/
│   ├── sync_to_nas.sh                # 事件驱动 rsync
│   ├── index_experiment.py           # 维护 INDEX.parquet
│   └── ...
├── data/                             # 大数据(.gitignore)
│   ├── datasets/                     # tokenized shards(训练热读)
│   ├── raw_generations/              # 合成原始 Parquet
│   ├── reference/                    # Cosmopedia v2 / FineWeb‑Edu 副本
│   └── tokenizers/
├── experiments/                      # 所有 ablation(.gitignore ckpt/ 子目录,其余入 git)
│   └── exp_{YYYYMMDD}_{slug}_{variant}/
│       ├── config.yaml               # 完全自包含,仅凭这个 yaml 就能复现
│       ├── metrics.json              # 训练 loss / 每 2B token 评测分数
│       ├── eval/                     # lighteval 完整结果
│       ├── ckpt/                     # 模型快照(不入 git,走 NAS 同步)
│       └── README.md                 # 人话总结:变了什么,学到什么
├── aim/                              # Aim 运行时 DB(.gitignore)
└── INDEX.parquet                     # 所有实验聚合一览(入 git)
```

## 三层存储防护

| 层 | 位置 | 内容 | 副本数 |
|---|---|---|---|
| L1 · 热 | 本地 NVMe `/home/ubuntu/lilei/projects/Incepedia/` | 全量 hot + cold,训练 / 评测 / 生成的 single source of truth | 1 |
| L2 · 冷镜像 | NAS `/lambda/nfs/us-south-2/incepedia_exp_bak/` | 事件驱动 rsync 的冷备份,多成员/agent 共享入口 | +1(冷) |
| L3 · 元数据兜底 | git 远端 | `configs/`, `experiments/*/config.yaml`, `experiments/*/metrics.json`, `INDEX.parquet` 等 <1GB 结构化产物 | +1(元数据) |

**最坏情况**:本地 SSD + NAS 同时挂 → 仍能从 git 救回所有实验配置与评测结论,只丢模型权重,可用相同配置重训。

## 同步策略(事件驱动)

| 触发 | 同步对象 | 脚本 |
|---|---|---|
| ckpt 保存完成 | `experiments/exp_xxx/ckpt/step_N/` | `scripts/sync_to_nas.sh ckpt` |
| 一轮合成批次完成 | `data/raw_generations/batch_xxx/` | `scripts/sync_to_nas.sh gen` |
| 实验启动 | `config.yaml`(立即) | `scripts/sync_to_nas.sh config` |
| 评测完成 | `eval/` 目录 | `scripts/sync_to_nas.sh eval` |
| 每晚 03:00 | 全量兜底 | cron |

**排除清单**:`aim/` / `logs/` / `.env` / `.git` / `*.tmp` / `*.lock`。

## 快速开始

```bash
# 1. 克隆 & 进入
cd /home/ubuntu/lilei/projects/Incepedia

# 2. 确认 .env(key 已配,查一下命名)
grep OPENROUTER_API_KEY .env

# 3. 创建环境(待添加 requirements.txt)
conda create -n incepedia python=3.10 -y
conda activate incepedia
pip install -r requirements.txt   # 待生成

# 4. 确认 NAS 挂载可写
touch /lambda/nfs/us-south-2/incepedia_exp_bak/.probe && rm /lambda/nfs/us-south-2/incepedia_exp_bak/.probe && echo "NAS OK"

# 5. 跑第一个 reference 实验(待实现)
python scripts/run_experiment.py --config experiments/exp_001_reference_cosmopedia_v2/config.yaml
```

## API Key 配置

双处配置(全局 + 仓库级)。**仓库级优先**,代码里用 `python-dotenv` 加载。

- **全局**:`~/.bash_profile` 里 `export OpenRouter_API_KEY=sk-or-v1-...`
- **仓库级**:本目录 `.env`(已 `.gitignore`)

代码中按此顺序查找:`OPENROUTER_API_KEY` → `OpenRouter_API_KEY` → `OPENAI_API_KEY`(已做兼容)。

## 实验命名规范

```
exp_{YYYYMMDD}_{variant_slug}_{seed?}
```
- 示例:`exp_20260420_reference_cosmopedia_v2_seed42`
- 示例:`exp_20260425_incepedia_v0.1_multigen_seed1337`

## 三阶段路线图

| 阶段 | 目标 | 输出 |
|---|---|---|
| **P1 · 复现** | 复现 Cosmopedia v2 在 1.82B/30B token 上的 early‑signal 分数,±0.5pp 内对齐 SmolLM 公布数字 | reference baseline + 训练/评测 pipeline |
| **P2 · PoC** | 3B token Incepedia v0.1 在同设置下持平或超越 Cosmopedia v2 | 最小闭环:生成 → 训练 → 评测 → 对比 |
| **P3 · 迭代与扩容** | 基于 ablation 反馈,扩到 10B/15B/30B,跨多 benchmark 超越 Cosmopedia v2,逼近 SmolLM2 | Incepedia v1.0 发布候选 |

## 关键参考

- [Cosmopedia 博客](https://huggingface.co/blog/cosmopedia)
- [SmolLM 博客(v2 配方)](https://huggingface.co/blog/smollm)
- [FineWeb 论文(ablation 范式)](https://arxiv.org/abs/2406.17557)
- [Cosmopedia GitHub](https://github.com/huggingface/cosmopedia)
- [Phi‑4 技术报告](https://arxiv.org/abs/2412.08905)

## 给 agent / 后续接手人员

读完本 README 后请按以下顺序:
1. [`docs/methodology.md`](./docs/methodology.md)——方法论与决策依据(待写)
2. [`docs/decisions/`](./docs/decisions/)——ADR 决策记录(待写)
3. `INDEX.parquet`——查最新实验状态与最佳分数
4. 最近修改的 `experiments/exp_*/README.md`——了解正在做什么

有疑问先读 `docs/FAQ.md`(待写),再提 issue。

---

© 2026 · Incepedia contributors · Apache‑2.0
