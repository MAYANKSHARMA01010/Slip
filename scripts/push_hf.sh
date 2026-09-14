#!/usr/bin/env bash
# scripts/push_hf.sh — Push current state to HuggingFace Space as a single orphan commit.
# This bypasses git history (which has old LFS binary files HF now rejects).
# Usage: bash scripts/push_hf.sh
# OR via alias: git pushhf

set -e

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
HF_REMOTE="hf"
HF_BRANCH="main"

echo "🚀 Deploying to HuggingFace Space..."

# 1. Create an orphan branch with only the current working state
git checkout --orphan _hf_deploy_tmp

# 2. Stage everything (respects .gitignore — pkl/vectorstore/venv etc. excluded)
git add -A

# 3. Commit as a single flat snapshot
git commit -m "deploy: $(git log -1 --format='%s' "$CURRENT_BRANCH" 2>/dev/null || echo 'update')" --quiet

# 4. Force-push the orphan to HF main (no history = no old LFS references)
git push "$HF_REMOTE" "_hf_deploy_tmp:$HF_BRANCH" --force

echo "✅ HuggingFace Space updated."

# 5. Cleanup: switch back to original branch and delete temp
git checkout "$CURRENT_BRANCH"
git branch -D _hf_deploy_tmp
