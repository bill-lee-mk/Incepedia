#!/usr/bin/env bash
# Incepedia · 环境 / 凭证 / NAS / 依赖健康检查
# 用法: bash scripts/check_setup.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAS_ROOT="${INCEPEDIA_NAS_ROOT:-/lambda/nfs/us-south-2/incepedia_exp_bak}"
PASS=0; FAIL=0; WARN=0

ok()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $1"; FAIL=$((FAIL+1)); }
warn() { echo "  ⚠️  $1"; WARN=$((WARN+1)); }

echo "────── 1. 仓库结构 ──────"
[[ -f "$REPO_ROOT/README.md" ]] && ok "README.md 存在" || bad "README.md 缺失"
[[ -f "$REPO_ROOT/pyproject.toml" ]] && ok "pyproject.toml 存在" || bad "pyproject.toml 缺失"
[[ -f "$REPO_ROOT/requirements.txt" ]] && ok "requirements.txt 存在" || bad "requirements.txt 缺失"
[[ -d "$REPO_ROOT/src/incepedia" ]] && ok "src/incepedia/ 包存在" || bad "src/incepedia/ 缺失"

echo
echo "────── 2. Git 身份(必须 repo-local,不能污染全局) ──────"
LOCAL_NAME=$(git -C "$REPO_ROOT" config --local user.name 2>/dev/null || true)
LOCAL_EMAIL=$(git -C "$REPO_ROOT" config --local user.email 2>/dev/null || true)
[[ -n "$LOCAL_NAME"  ]] && ok "local user.name  = $LOCAL_NAME"   || bad "local user.name  未设置(会使用全局 — 危险)"
[[ -n "$LOCAL_EMAIL" ]] && ok "local user.email = $LOCAL_EMAIL"  || bad "local user.email 未设置(会使用全局 — 危险)"
LOCAL_CRED_HELPER=$(git -C "$REPO_ROOT" config --local credential.helper 2>/dev/null || true)
if [[ "$LOCAL_CRED_HELPER" == *".credentials-local"* ]]; then
  ok "local credential helper 指向仓库私有文件"
else
  warn "local credential helper 未配置为仓库私有文件:当前='${LOCAL_CRED_HELPER}'"
fi
if [[ -f "$REPO_ROOT/.git/.credentials-local" ]]; then
  MODE=$(stat -c '%a' "$REPO_ROOT/.git/.credentials-local")
  [[ "$MODE" == "600" ]] && ok ".git/.credentials-local 权限 600" || bad ".git/.credentials-local 权限=$MODE(应为 600)"
fi

echo
echo "────── 3. .env / API keys ──────"
if [[ -f "$REPO_ROOT/.env" ]]; then
  ok ".env 存在"
  ENV_MODE=$(stat -c '%a' "$REPO_ROOT/.env")
  [[ "$ENV_MODE" == "600" || "$ENV_MODE" == "640" ]] && ok ".env 权限=$ENV_MODE" || warn ".env 权限=$ENV_MODE(建议 600)"
  if grep -q '^OPENROUTER_API_KEY=sk-or-v1-' "$REPO_ROOT/.env"; then
    KEY_LEN=$(grep '^OPENROUTER_API_KEY=' "$REPO_ROOT/.env" | cut -d= -f2- | tr -d '\n' | wc -c)
    ok "OPENROUTER_API_KEY 格式正确,长度=$KEY_LEN"
  else
    bad "OPENROUTER_API_KEY 格式不正确"
  fi
  if grep -q '^GITHUB_PAT=ghp_' "$REPO_ROOT/.env"; then
    ok "GITHUB_PAT 格式正确"
  else
    warn "GITHUB_PAT 格式不正确(可能用了其它 token 类型,检查一下)"
  fi
else
  bad ".env 缺失(请从 .env.example 拷贝)"
fi
git -C "$REPO_ROOT" check-ignore -q .env && ok ".env 已 gitignore" || bad ".env 未 gitignore — 立即修复!"

echo
echo "────── 4. NAS 冷备份 ──────"
if [[ -d "$NAS_ROOT" ]]; then
  ok "NAS 目录存在: $NAS_ROOT"
  TESTFILE="$NAS_ROOT/.probe_$$"
  if touch "$TESTFILE" 2>/dev/null && rm -f "$TESTFILE" 2>/dev/null; then
    ok "NAS 可写"
  else
    bad "NAS 不可写 — 检查挂载和权限"
  fi
  [[ -f "$NAS_ROOT/README.md"    ]] && ok "NAS README.md 存在"    || warn "NAS README.md 缺失"
  [[ -f "$NAS_ROOT/README.en.md" ]] && ok "NAS README.en.md 存在" || warn "NAS README.en.md 缺失"
else
  bad "NAS 目录不存在: $NAS_ROOT"
fi

echo
echo "────── 5. Python 环境 ──────"
if command -v python >/dev/null; then
  PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  ok "python = $(command -v python) ($PY_VER)"
  python -c "import incepedia" 2>/dev/null && ok "incepedia 包可导入" || warn "incepedia 包未安装(跑 bootstrap_env.sh)"
  for mod in dotenv yaml pandas pyarrow httpx datasketch; do
    python -c "import $mod" 2>/dev/null && ok "dep ok: $mod" || warn "dep 缺失: $mod"
  done
else
  bad "python 未找到(conda 环境未激活?)"
fi

echo
echo "────── 汇总 ──────"
echo "  通过 $PASS  警告 $WARN  失败 $FAIL"
[[ $FAIL -eq 0 ]] || exit 1
