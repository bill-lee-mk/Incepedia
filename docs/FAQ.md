# FAQ

## Q: 我想在历史 ckpt 上分叉一个新 ablation,怎么做?
```bash
SRC_EXP=exp_20260425_incepedia_v0.1_multigen_seed1337
NEW_EXP=exp_$(date +%Y%m%d)_fork_from_${SRC_EXP}_seed42

# 1. 从 NAS 拉父实验
rsync -aP /lambda/nfs/us-south-2/incepedia_exp_bak/experiments/$SRC_EXP/ \
        experiments/$SRC_EXP/

# 2. 开新实验目录,复制 config
mkdir -p experiments/$NEW_EXP
cp experiments/$SRC_EXP/config.yaml experiments/$NEW_EXP/config.yaml

# 3. 编辑 config.yaml:改 exp_id、branch_from、要动的那一个变量
vim experiments/$NEW_EXP/config.yaml

# 4. 注册 + 跑
python scripts/index_experiment.py add $NEW_EXP
# ... launcher TBD ...
```

## Q: 我改了 configs/topics.yaml,怎么让它生效?
配置是引用式的 — 下一轮 `PromptAssembler` 启动时会读。已生成的数据不会受影响。

## Q: 为什么不用 WandB?
- 多 run 对比需要付费
- 所有 run metadata 托管在云端,对"实验历史可 git 可审计"的目标不友好
- 本地 Aim(SQLite)+ 仓库内 `INDEX.parquet` 已覆盖需求

## Q: NAS 为什么不做热读?
virtiofs 元数据延迟 ~9 ms/op,小文件创建 ~263 文件/秒。训练/评测热路径会被拖慢 10–50%。详见 `README.md → NAS 性能基准`。

## Q: 生成器档位怎么选?
`configs/generators.yaml` → `task_map` 决定默认 tier。具体 task 模板可覆盖。看这条 FAQ 前请先读 `docs/methodology.md § 3.1`。

## Q: 多久做一次全量 rsync 到 NAS?
平时走**事件驱动**同步(ckpt 存完立即、eval 完成立即、config 启动立即)。**每晚 03:00 走一次 `scripts/sync_to_nas.sh nightly` 兜底**。
