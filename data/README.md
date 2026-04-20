# data/

**此目录下的子目录内容不入 git**(见根 `.gitignore`)。

## 子目录

- `datasets/` — tokenized 训练数据(nanotron shard 格式)
- `raw_generations/` — 合成原始输出(Parquet)
- `reference/` — 下载的对照基线数据(Cosmopedia v2 / FineWeb-Edu 等)
- `tokenizers/` — tokenizer 缓存

## 同步到 NAS

```bash
# 同步一个数据集
bash scripts/sync_to_nas.sh dataset incepedia_v0.1

# 同步一轮生成批次
bash scripts/sync_to_nas.sh gen batch_20260425_persona_v1
```

## 原则

1. 大文件只走 Parquet / Arrow / binary shard,禁止几十万小文件
2. 下游训练读路径 **永远**从本地 NVMe 读,不走 NAS
