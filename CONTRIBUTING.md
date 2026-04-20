# Contributing to Incepedia

> 这份指南面向**人类贡献者**。AI agent 请先读 [`AGENTS.md`](./AGENTS.md)。

## Workflow

1. **拉最新**:`git pull --rebase`
2. **起分支**(可选,小改动直接 main 也行):`git checkout -b feat/<slug>`
3. **改代码**
4. **跑 check**:`bash scripts/check_setup.sh`
5. **commit**:Conventional Commits 风格
   ```
   <type>(<scope>): <summary>
   ```
   类型:`feat` / `fix` / `docs` / `chore` / `refactor` / `exp` / `data` / `eval`
6. **push**:`git push`
7. **(可选)** 发起 PR

## 代码风格

- Python:ruff(配置在 `pyproject.toml`),`ruff check . && ruff format .` 通过再 commit
- Shell:`shellcheck`(已纳入 `check_setup.sh`)
- YAML:保持 2 空格缩进,字段按语义分块

## 实验流程(ablation)

严格要求 → [`AGENTS.md` §6](./AGENTS.md#6-ablation-discipline) + [`docs/methodology.md`](./docs/methodology.md)

## 写 ADR(Architecture Decision Record)

有影响多个模块 / 难以回退的决定,要写一份 ADR:
```bash
cp docs/decisions/TEMPLATE.md docs/decisions/$(printf '%04d' $(( $(ls docs/decisions | grep -cE '^[0-9]{4}-') + 1 )))-<slug>.md
```

## 不要做的事

- ❌ 不要改全局 git config(`~/.gitconfig`)——这台机器多人共用
- ❌ 不要在任何输出里暴露 `.env` 的内容或 token
- ❌ 不要把 `data/` / `ckpt/` / `aim/` 加进 git
- ❌ 不要在 NAS 上直接跑训练 / 评测
- ❌ 不要改 `experiments/exp_*/config.yaml` 已经跑过的实验(新建一个 `branch_from` 分叉)

---

© 2026 · Incepedia contributors · Apache-2.0
