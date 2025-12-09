#!/usr/bin/env bash
set -euo pipefail

echo "Running markdown-link-check on top-level docs and docs/ directory..."
if ! command -v markdown-link-check >/dev/null 2>&1; then
  echo "markdown-link-check not installed, installing globally (requires npm)..."
  npm install -g markdown-link-check
fi

FILES=$(find . -type f -name "*.md" -not -path "./node_modules/*" -not -path "./frontend/node_modules/*" -not -path "./fenix_env/*" -not -path "./.git/*" -not -path "./.venv/*" -not -path "*/site-packages/*")

for f in $FILES; do
  if [[ -f "$f" ]]; then
    echo "Checking $f"
    markdown-link-check -q "$f" || echo "Link check failed for $f"
  fi
done

echo "Done link checking."
