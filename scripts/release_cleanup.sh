#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || echo "${PWD}")
cd "$ROOT_DIR"

echo "Preparing repository for release: cleaning tracked envs, venvs, DBs, and logs"

# Remove common volatile files from git index but keep locally
git rm --cached -r .venv || true
git rm --cached -r fenix_env || true
git rm --cached -r venv || true
git rm --cached -r env || true
git rm --cached .env || true
git rm --cached fenix_trading.db || true
git rm --cached -r logs || true
git rm --cached -r llm_responses || true
git rm --cached -r backups || true

# Commit removal if changes exist
if ! git diff --cached --quiet; then
  git add .gitignore
  git commit -m "chore: prepare release - remove local envs, logs, dbs, and venvs from git index"
fi

echo "Searching for sensitive patterns in repository (no change yet):"
git grep -nE "BINANCE_API_KEY|BINANCE_API_SECRET|OPENAI_API_KEY|HUGGINGFACE_API_KEY|GROQ_API_KEY|_API_KEY|_SECRET|sk-|-----BEGIN PRIVATE KEY-----|BEGIN RSA PRIVATE KEY" || true

echo "If secrets are found above, revoke or rotate and re-run cleanup / use git filter-repo to purge history if needed." 

echo "Done."
