# experiments/

每个 ablation run 一个子目录,命名规范:

```
exp_{YYYYMMDD}_{variant_slug}_seed{N}
```

## 目录结构(强制)

```
exp_XXX/
├── config.yaml       # ★ 完全自包含;仅凭这个文件就能复现整个实验
├── metrics.json      # ★ 训练 + 评测指标(随训练追加)
├── README.md         # ★ 人话总结:改了什么、学到什么、下一步
├── ckpt/             # 模型快照(.gitignore,走 NAS 同步)
├── eval/             # lighteval 原始输出(.gitignore,走 NAS 同步)
└── aim_run_id.txt    # (.gitignore) 指向 Aim run 的链接
```

**入 git 的**:`config.yaml` / `metrics.json` / `README.md` — 都是小文本,可 diff,可审计。

**不入 git 的**:`ckpt/` / `eval/` / `aim_run_id.txt` — 走 NAS + 本地 NVMe 双副本。

## 工作流

```bash
EXP=exp_$(date +%Y%m%d)_reference_cosmopedia_v2_seed42
mkdir -p experiments/$EXP
# 1. 写 config.yaml
# 2. 跑训练: python scripts/run_experiment.py --config experiments/$EXP/config.yaml  (TBD)
# 3. 跑评测: python scripts/eval_experiment.py --exp $EXP  (TBD)
# 4. 写 README.md(手动,人话总结)
# 5. 登记到索引
python scripts/index_experiment.py add $EXP
# 6. 同步到 NAS
bash scripts/sync_to_nas.sh config $EXP
bash scripts/sync_to_nas.sh ckpt   $EXP
bash scripts/sync_to_nas.sh eval   $EXP
```

## 分叉(基于历史 ckpt 开新 ablation)

见 [`../docs/FAQ.md`](../docs/FAQ.md#q-我想在历史-ckpt-上分叉一个新-ablation怎么做)。
