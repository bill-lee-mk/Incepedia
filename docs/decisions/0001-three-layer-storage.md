# 0001 · Three-layer storage defense

- **Status**: accepted
- **Date**: 2026-04-20
- **Deciders**: bill-lee-mk
- **Context**:
  - 单节点本地 NVMe(8.5 TB 可用)性能最高,但单点故障风险
  - Lambda virtiofs NAS(3.1 PB 可用)读写带宽尚可(大文件 885/592 MB/s),但小文件元数据延迟高
  - 仓库需被多成员 / 多 agent 共享,实验结果需长期可溯
  - 合成与训练成本高,不能丢失

- **Decision**:
  采用三层防护:
  1. **L1 本地 NVMe** 作为 single source of truth,所有 hot + cold 数据都在这里
  2. **L2 NAS** 作为事件驱动 rsync 的冷镜像,保证离机副本和团队共享
  3. **L3 git 远端** 作为 `<1GB` 结构化产物(config / metrics / INDEX.parquet)的兜底副本

- **Alternatives considered**:
  1. **NAS 作为主盘**:读 592 MB/s 会拖慢训练 10–50%,元数据操作慢 30–100×,放弃。
  2. **对象存储 (S3/MinIO)**:需要额外服务部署,目前没有必要。
  3. **仅本地 + git**:仓库 <1GB 限制下无法备份 ckpt / 数据,不可行。

- **Consequences**:
  - Positive:零性能影响;三副本覆盖绝大多数故障;新成员从 NAS 一键上手
  - Negative:需要维护事件驱动 rsync 脚本与排除清单;NAS 元数据慢 → 强制所有 metadata 聚合 Parquet
  - Follow-ups:实现 `scripts/sync_to_nas.sh`(已完成);写 cron 兜底(待办);团队访问规则文档化(NAS README 已覆盖)

- **Related**:
  - `scripts/sync_to_nas.sh`
  - `README.md § 三层存储防护`
  - `/lambda/nfs/us-south-2/incepedia_exp_bak/README.md`
