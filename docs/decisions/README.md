# Architecture Decision Records (ADR)

**What is an ADR?** A short markdown file recording a decision that's hard to reverse or affects multiple parts of the project.

## When to write one

- Changing data pipeline architecture
- Choosing between incompatible tools/frameworks
- Setting policies (naming, storage, evaluation)
- Abandoning a prior decision (supersede, not delete)

## Format

Copy `TEMPLATE.md`, name it `NNNN-short-slug.md` where `NNNN` is the next 4-digit index (zero-padded). Keep slugs lowercase, hyphen-separated.

## Status values

- `proposed` — under discussion, not yet adopted
- `accepted` — in effect
- `superseded-by 0xyz-...` — replaced
- `deprecated` — no longer relevant but kept for history

## Index

| # | Title | Status |
|---|---|---|
| [0001](./0001-three-layer-storage.md) | Three-layer storage defense (local + NAS + git) | accepted |
| [0002](./0002-ablation-as-quality-metric.md) | Ablation training as the sole data-quality metric | accepted (partially superseded by 0004) |
| [0003](./0003-multi-tier-generator-routing.md) | Multi-tier generator routing via OpenRouter | accepted |
| [0004](./0004-evaluation-protocol-dual-track-adaptive.md) | Dual-track adaptive evaluation protocol | accepted |
| [0005](./0005-project-scope-standalone-plus-seasoning.md) | Project scope — standalone corpus AND decay seasoning | accepted |
| [0006](./0006-evaluation-stack-policy.md) | Evaluation stack policy — pin to lighteval latest | accepted |
